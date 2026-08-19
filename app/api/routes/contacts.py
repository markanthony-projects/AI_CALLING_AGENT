"""Managing what gets dialled: campaigns, their contact queues, and the do-not-call list.

Mounted under the dashboard's prefix and behind the same session auth, but kept in its own
module because dashboard.py is the read side — metrics, lists, detail views — and everything
here writes. Anything that spends money or changes who gets called requires an admin session,
for the same reason the dial endpoint always has: a browser must never hold a spend-capable
credential, and within the dashboard the destructive surface has a named holder.

The queue itself is drained by the pump in app/services/dial_pump.py. Nothing in this module
places a call. Starting a campaign here means setting it ACTIVE and letting the pump find it,
which is what makes pausing work: there is no queue of scheduled jobs to hunt down and cancel.
"""

import uuid
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core import call_slots
from app.core.security import SessionClaims, require_admin, require_session
from app.models.db import (
    RETRIABLE_CONTACT_STATUSES,
    Campaign,
    CampaignStatus,
    Contact,
    ContactStatus,
    Project,
    Suppression,
)
from app.services.contact_import import ImportReport, load, parse
from app.services.dial_pump import MAX_DIAL_ATTEMPTS
from app.services.discovery import invalidate_project_cache
from app.utils.phone import to_e164
from app.utils.timeutils import utc_now

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["contacts"],
    dependencies=[Depends(require_session)],
)

MAX_PAGE_SIZE = 200

# A lead list is a few hundred KB. Ten is far past any real one and well under what would let
# an authenticated operator exhaust the droplet's memory by uploading in a loop.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm")


# --- shapes ---------------------------------------------------------------------------


class ContactOut(BaseModel):
    id: str
    phone_number: str
    name: Optional[str] = None
    status: str
    attempts: int
    last_attempt_at: Optional[str] = None
    next_attempt_at: Optional[str] = None
    last_outcome: Optional[str] = None
    source_row: Optional[int] = None


class ContactPage(BaseModel):
    items: List[ContactOut]
    total: int
    page: int
    page_size: int
    # Every status and its count for the whole campaign, not just this page. The operator's
    # first question is "how much is left", and a page of 50 cannot answer it.
    counts: dict
    max_attempts: int


class RowProblemOut(BaseModel):
    row: int
    reason: str
    value: str = ""


class ImportPreview(BaseModel):
    """What the file contains, and what importing it would do.

    The columns are reported back because a silent mis-mapping is the failure that matters:
    a name column read as the phone column dials nothing, and a phone column read as the name
    column makes the agent greet somebody as "9876543210".
    """

    batch_id: str
    committed: bool
    header_row: Optional[int] = None
    phone_column: Optional[str] = None
    name_column: Optional[str] = None
    total_rows: int
    dialable: int
    problems: List[RowProblemOut]
    sample: List[ContactOut]
    inserted: int = 0
    already_present: int = 0
    suppressed: int = 0


class CampaignWrite(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    project_id: uuid.UUID


class CampaignPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    status: Optional[CampaignStatus] = None


class ContactAction(BaseModel):
    """A bulk change to specific contacts, or to a whole status within one campaign.

    Either form is allowed because both are things an operator actually does: retry these
    four, and retry all 900 that rang out. `ids` wins when both are given.
    """

    action: str
    ids: Optional[List[uuid.UUID]] = None
    from_status: Optional[ContactStatus] = None
    reason: Optional[str] = Field(default=None, max_length=200)

    @field_validator("action")
    @classmethod
    def known_action(cls, v: str) -> str:
        allowed = {"retry", "skip", "dnd", "unskip"}
        if v not in allowed:
            raise ValueError(f"action must be one of {sorted(allowed)}")
        return v


class SuppressionOut(BaseModel):
    id: str
    phone_number: str
    reason: Optional[str] = None
    added_by: Optional[str] = None
    created_at: str


class SuppressionAdd(BaseModel):
    phone_numbers: List[str] = Field(min_length=1, max_length=1000)
    reason: Optional[str] = Field(default=None, max_length=200)


# --- helpers --------------------------------------------------------------------------


def _iso(value) -> Optional[str]:
    return value.isoformat() if value else None


def _out(contact: Contact) -> ContactOut:
    return ContactOut(
        id=str(contact.id),
        phone_number=contact.phone_number,
        name=contact.name,
        status=contact.status.value,
        attempts=contact.attempts or 0,
        last_attempt_at=_iso(contact.last_attempt_at),
        next_attempt_at=_iso(contact.next_attempt_at),
        last_outcome=contact.last_outcome,
        source_row=contact.source_row,
    )


async def _require_campaign(db: AsyncSession, campaign_id: uuid.UUID) -> Campaign:
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")
    return campaign


async def _status_counts(db: AsyncSession, campaign_id: uuid.UUID) -> dict:
    rows = await db.execute(
        select(Contact.status, func.count(Contact.id))
        .where(Contact.campaign_id == campaign_id)
        .group_by(Contact.status)
    )
    # Every status present, so the UI does not have to distinguish "zero" from "missing".
    counts = {s.value: 0 for s in ContactStatus}
    for contact_status, count in rows:
        counts[contact_status.value] = int(count)
    return counts


# --- campaigns ------------------------------------------------------------------------


@router.post(
    "/campaigns",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
async def create_campaign(
    body: CampaignWrite,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Create a campaign against a project.

    Deliberately PAUSED on creation. An ACTIVE campaign is one the pump draws from, so
    creating it ACTIVE would start dialling the moment the first contact was imported —
    before anybody had looked at the import preview.
    """
    if await db.get(Project, body.project_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    campaign = Campaign(
        project_id=body.project_id, name=body.name.strip(), status=CampaignStatus.PAUSED
    )
    db.add(campaign)
    await db.commit()
    logger.info(f"Campaign '{campaign.name}' created by {claims.email} (paused)")
    return {"id": str(campaign.id), "name": campaign.name, "status": campaign.status.value}


@router.patch("/campaigns/{campaign_id}/details", dependencies=[Depends(require_admin)])
async def update_campaign(
    campaign_id: uuid.UUID,
    body: CampaignPatch,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Rename a campaign, or start and stop it.

    Setting ACTIVE is what starts dialling: the pump picks up any ACTIVE campaign with
    eligible contacts on its next tick. Setting PAUSED stops it within that same tick, and
    calls already in flight are left to finish rather than cut off mid-sentence.
    """
    campaign = await _require_campaign(db, campaign_id)
    if body.name is not None:
        campaign.name = body.name.strip()
    if body.status is not None and body.status != campaign.status:
        campaign.status = body.status
        logger.info(f"Campaign {campaign_id} set {body.status.value} by {claims.email}")
    await db.commit()
    # discovery.py caches campaign -> project; a campaign whose state just changed must not
    # keep serving the old context to a live call.
    await invalidate_project_cache(str(campaign_id))
    return {"id": str(campaign.id), "name": campaign.name, "status": campaign.status.value}


@router.delete(
    "/campaigns/{campaign_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)],
)
async def delete_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Delete a campaign and its contact queue.

    Refused while it is ACTIVE. Deleting a campaign the pump is drawing from would remove
    rows mid-dial, and the calls already placed would arrive at a webhook whose campaign no
    longer exists. Pause it first — which is also a moment to reconsider.

    Calls and their transcripts survive: the foreign key is SET NULL, because losing the call
    history of a finished campaign is a far worse trade than keeping rows with no campaign.
    """
    campaign = await _require_campaign(db, campaign_id)
    if campaign.status == CampaignStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pause the campaign before deleting it",
        )
    await db.delete(campaign)
    await db.commit()
    await invalidate_project_cache(str(campaign_id))
    logger.warning(f"Campaign {campaign_id} deleted by {claims.email}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- contacts -------------------------------------------------------------------------


@router.get("/campaigns/{campaign_id}/contacts", response_model=ContactPage)
async def list_contacts(
    campaign_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    contact_status: Optional[ContactStatus] = Query(default=None, alias="status"),
    q: Optional[str] = Query(default=None, max_length=100),
    db: AsyncSession = Depends(get_db),
):
    """The queue for one campaign, filtered and paged, with whole-campaign counts alongside."""
    await _require_campaign(db, campaign_id)

    stmt = select(Contact).where(Contact.campaign_id == campaign_id)
    if contact_status is not None:
        stmt = stmt.where(Contact.status == contact_status)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(Contact.phone_number.ilike(like) | Contact.name.ilike(like))

    total = await db.scalar(
        select(func.count()).select_from(stmt.order_by(None).subquery())
    )
    rows = (
        await db.execute(
            # Same order the pump works in, so the list reads as the queue it is.
            stmt.order_by(Contact.created_at.asc(), Contact.source_row.asc())
            .limit(page_size)
            .offset((page - 1) * page_size)
        )
    ).scalars().all()

    return ContactPage(
        items=[_out(c) for c in rows],
        total=int(total or 0),
        page=page,
        page_size=page_size,
        counts=await _status_counts(db, campaign_id),
        max_attempts=MAX_DIAL_ATTEMPTS,
    )


@router.post(
    "/campaigns/{campaign_id}/contacts/import",
    response_model=ImportPreview,
    dependencies=[Depends(require_admin)],
)
async def import_contacts(
    campaign_id: uuid.UUID,
    file: UploadFile = File(...),
    commit: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Read a spreadsheet of leads. Writes nothing unless commit=true.

    Two calls rather than one upload held in server state: the operator uploads to see the
    preview, then uploads the same file to commit it. That costs a second upload of a few
    hundred KB and buys statelessness — no token to expire, no half-finished import sitting in
    Redis, and nothing that behaves differently because the preview was left open too long.
    Re-committing is harmless anyway; the unique constraint makes the write idempotent.
    """
    await _require_campaign(db, campaign_id)

    name = (file.filename or "").lower()
    if not name.endswith(_SPREADSHEET_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload an .xlsx file. Save a CSV as Excel and try again.",
        )

    # Read with a ceiling rather than trusting Content-Length, which the client sets.
    body = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
            f"Split it and import in parts.",
        )
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The file is empty"
        )

    import io

    try:
        report = parse(io.BytesIO(body))
    except Exception as e:
        # openpyxl raises a variety of things on a file that is not really a workbook.
        logger.warning(f"Could not read the uploaded lead list: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That file could not be read as a spreadsheet.",
        )

    if commit and report.contacts:
        report = await load(db, campaign_id, report)
        logger.info(
            f"{claims.email} imported {report.inserted} contact(s) into campaign {campaign_id}"
        )

    return _preview(report, committed=commit)


def _preview(report: ImportReport, *, committed: bool) -> ImportPreview:
    return ImportPreview(
        batch_id=str(report.batch_id),
        committed=committed,
        header_row=report.header_row,
        phone_column=report.phone_column,
        name_column=report.name_column,
        total_rows=report.total_rows,
        dialable=report.dialable,
        # Capped: a file with 3,000 bad rows must not put 3,000 of them in one response.
        problems=[
            RowProblemOut(row=p.row, reason=p.reason, value=p.value) for p in report.problems[:200]
        ],
        # Enough to see the columns were read the way the operator meant.
        sample=[
            ContactOut(
                id="",
                phone_number=c.phone_number,
                name=c.name,
                status=ContactStatus.PENDING.value,
                attempts=0,
                source_row=c.row,
            )
            for c in report.contacts[:10]
        ],
        inserted=report.inserted,
        already_present=report.already_present,
        suppressed=report.suppressed,
    )


@router.get("/contacts/import/template", dependencies=[Depends(require_admin)])
async def import_template():
    """A spreadsheet with the headers the importer looks for.

    Column detection is forgiving, but an operator who starts from this never finds out what
    it does not accept.
    """
    import io

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Leads"
    sheet.append(["Name", "Phone"])
    sheet.append(["Rahul Sharma", "9876543210"])
    sheet.append(["Priya Nair", "+919876543211"])
    # Text, so Excel does not turn a ten-digit number into 9.87654E+09 on the way back in —
    # which is a row the importer has to reject because the digits are genuinely gone.
    for row in sheet.iter_rows(min_col=2, max_col=2):
        for cell in row:
            cell.number_format = "@"

    buffer = io.BytesIO()
    workbook.save(buffer)
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="lead-list-template.xlsx"'},
    )


@router.post(
    "/campaigns/{campaign_id}/contacts/actions",
    dependencies=[Depends(require_admin)],
)
async def act_on_contacts(
    campaign_id: uuid.UUID,
    body: ContactAction,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Retry, skip, or mark do-not-call, for specific contacts or a whole status.

    DIALING is excluded from every action. A call in flight has taken a carrier slot and is
    being served; changing its queue entry underneath it would have the outcome written back
    over the operator's decision a minute later.
    """
    await _require_campaign(db, campaign_id)

    where = [Contact.campaign_id == campaign_id, Contact.status != ContactStatus.DIALING]
    if body.ids:
        where.append(Contact.id.in_(body.ids))
    elif body.from_status is not None:
        where.append(Contact.status == body.from_status)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name the contacts with ids, or a status to act on with from_status",
        )

    if body.action == "retry":
        # attempts is reset, not merely decremented: an operator retrying deliberately is
        # starting again, and leaving the count at three would exhaust it on the first pass.
        values = dict(
            status=ContactStatus.PENDING,
            attempts=0,
            next_attempt_at=None,
            last_outcome=f"retried by {claims.email}",
        )
    elif body.action == "skip":
        values = dict(status=ContactStatus.SKIPPED, last_outcome=f"skipped by {claims.email}")
    elif body.action == "unskip":
        where.append(Contact.status == ContactStatus.SKIPPED)
        values = dict(
            status=ContactStatus.PENDING, next_attempt_at=None, last_outcome="returned to the queue"
        )
    else:  # dnd
        values = dict(
            status=ContactStatus.DND,
            last_outcome=(body.reason or f"marked do-not-call by {claims.email}"),
        )

    affected = (
        await db.execute(update(Contact).where(*where).values(**values))
    ).rowcount or 0

    # Marking do-not-call has to outlive this campaign, or the next project's import undoes it.
    if body.action == "dnd":
        numbers = (
            await db.execute(
                select(Contact.phone_number).where(*where)
                if not affected
                else select(Contact.phone_number).where(
                    Contact.campaign_id == campaign_id, Contact.status == ContactStatus.DND
                )
            )
        ).scalars().all()
        await _suppress(db, numbers, body.reason, claims.email)

    await db.commit()
    logger.info(f"{claims.email} applied '{body.action}' to {affected} contact(s) on {campaign_id}")
    return {"action": body.action, "affected": affected, "counts": await _status_counts(db, campaign_id)}


@router.delete(
    "/campaigns/{campaign_id}/contacts/batches/{batch_id}",
    dependencies=[Depends(require_admin)],
)
async def delete_import_batch(
    campaign_id: uuid.UUID,
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Undo one import.

    The reason import_batch_id exists. An operator who uploads the wrong file needs it gone,
    and picking the rows out by phone number is not something anybody will do for 900 of them.

    Only rows never dialled are removed. A contact that has been called carries a call record
    and possibly a lead, and deleting it would orphan both to tidy up a paperwork mistake.
    """
    await _require_campaign(db, campaign_id)
    result = await db.execute(
        delete(Contact).where(
            Contact.campaign_id == campaign_id,
            Contact.import_batch_id == batch_id,
            Contact.attempts == 0,
            Contact.status != ContactStatus.DIALING,
        )
    )
    await db.commit()
    removed = result.rowcount or 0
    logger.warning(f"{claims.email} removed {removed} contact(s) from import batch {batch_id}")
    return {"removed": removed, "counts": await _status_counts(db, campaign_id)}


# --- suppression ----------------------------------------------------------------------


async def _suppress(db: AsyncSession, numbers, reason: Optional[str], by: Optional[str]) -> int:
    """Add numbers to the do-not-call list, ignoring any already on it."""
    from sqlalchemy.dialects.postgresql import insert

    unique = sorted({n for n in numbers if n})
    if not unique:
        return 0
    statement = (
        insert(Suppression)
        .values([
            {"id": uuid.uuid4(), "phone_number": n, "reason": reason, "added_by": by}
            for n in unique
        ])
        .on_conflict_do_nothing(constraint="uq_suppressions_phone")
        .returning(Suppression.id)
    )
    return len((await db.execute(statement)).scalars().all())


@router.get("/suppressions", response_model=List[SuppressionOut])
async def list_suppressions(
    q: Optional[str] = Query(default=None, max_length=40),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_SIZE),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Suppression).order_by(Suppression.created_at.desc()).limit(limit)
    if q:
        stmt = stmt.where(Suppression.phone_number.ilike(f"%{q.strip()}%"))
    rows = (await db.execute(stmt)).scalars().all()
    return [
        SuppressionOut(
            id=str(s.id),
            phone_number=s.phone_number,
            reason=s.reason,
            added_by=s.added_by,
            created_at=s.created_at.isoformat(),
        )
        for s in rows
    ]


@router.post("/suppressions", dependencies=[Depends(require_admin)])
async def add_suppressions(
    body: SuppressionAdd,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Add numbers to the do-not-call list, and stop them mid-campaign.

    Existing queue entries are marked DND in the same transaction. Adding to the list without
    that would honour the request for future imports while the numbers already queued behind
    it kept ringing.
    """
    normalised, rejected = [], []
    for raw in body.phone_numbers:
        try:
            normalised.append(to_e164(raw))
        except ValueError as e:
            rejected.append({"value": raw, "reason": str(e)})

    added = await _suppress(db, normalised, body.reason, claims.email)
    stopped = 0
    if normalised:
        stopped = (
            await db.execute(
                update(Contact)
                .where(
                    Contact.phone_number.in_(normalised),
                    Contact.status.notin_([ContactStatus.DIALING, ContactStatus.DND]),
                )
                .values(status=ContactStatus.DND, last_outcome="added to the do-not-call list")
            )
        ).rowcount or 0
    await db.commit()
    logger.warning(f"{claims.email} suppressed {added} number(s); stopped {stopped} queued")
    return {"added": added, "already_listed": len(normalised) - added, "stopped_in_queue": stopped, "rejected": rejected}


@router.delete(
    "/suppressions/{suppression_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)],
)
async def remove_suppression(
    suppression_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Take a number off the do-not-call list.

    Contacts already marked DND are left alone. Un-suppressing is for a number added by
    mistake, and silently returning somebody to a live dial queue is not a decision this
    endpoint should make on the operator's behalf — they can retry the contact explicitly.
    """
    entry = await db.get(Suppression, suppression_id)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not on the list")
    await db.delete(entry)
    await db.commit()
    logger.warning(f"{claims.email} un-suppressed {entry.phone_number}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- carrier capacity -----------------------------------------------------------------


@router.get("/dial-capacity")
async def dial_capacity(db: AsyncSession = Depends(get_db)):
    """What the carrier is carrying right now, and what is waiting.

    Read from Redis rather than a per-process counter, so it is the same number every
    container is enforcing against.
    """
    now = utc_now()
    waiting = await db.scalar(
        select(func.count(Contact.id))
        .join(Campaign, Campaign.id == Contact.campaign_id)
        .where(
            Campaign.status == CampaignStatus.ACTIVE,
            Contact.status.in_((ContactStatus.PENDING, *RETRIABLE_CONTACT_STATUSES)),
            Contact.attempts < MAX_DIAL_ATTEMPTS,
            (Contact.next_attempt_at.is_(None)) | (Contact.next_attempt_at <= now),
        )
    )
    from app.core.config import settings

    return {
        "in_flight": await call_slots.active(),
        "max_concurrent": settings.MAX_CONCURRENT_CALLS,
        "waiting_to_dial": int(waiting or 0),
        "call_sids": await call_slots.held(),
    }


@router.post("/dial-capacity/reset", dependencies=[Depends(require_admin)])
async def reset_dial_capacity(claims: SessionClaims = Depends(require_admin)):
    """Clear every held slot.

    For an operator who knows the carrier has no calls up. The slots expire on their own after
    the longest a call can run, so this only saves waiting that out — but on a three-slot
    account one leaked slot is a third of the throughput, so waiting it out is expensive.
    """
    released = await call_slots.reset()
    logger.warning(f"{claims.email} reset the carrier slot count ({released} were held)")
    return {"released": released, "in_flight": await call_slots.active()}
