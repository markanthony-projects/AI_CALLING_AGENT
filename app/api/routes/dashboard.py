"""Read APIs for the operations dashboard.

Every timestamp in this module crosses one of two boundaries, and mixing them up is the
bug that keeps recurring in this codebase:

  * calls.started_at / ended_at and leads.created_at hold **naive UTC**.
  * leads.site_visit_time / callback_time hold **IST wall-clock time**, because that is
    what a salesperson dials against.

So anything grouped or bucketed by day converts UTC to IST first (_IST_SHIFT), and
appointment columns are compared against IST now (not utc_now) — otherwise "today's site
visits" is wrong by five and a half hours, which at 09:00 IST means yesterday's list.

Responses are deliberately explicit Pydantic models rather than ORM objects: the Lead and
Call tables carry phone numbers and transcripts, and a serializer that reflects whatever
columns exist will happily leak the next one somebody adds.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field, computed_field
from sqlalchemy import Select, case, cast, delete, func, or_, select, text
from sqlalchemy import String as SAString
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core import call_slots
from app.api.routes.campaign import DialRequest
from app.core.ratelimit import window_keys
from app.core.security import SessionClaims, require_admin, require_session
from app.models.db import (
    Call,
    CallStatus,
    Campaign,
    CampaignStatus,
    Lead,
    LeadStatus,
    Project,
    Transcript,
)
from app.services import dial_queue
from app.services.discovery import invalidate_campaign_context, invalidate_project_everywhere
from app.utils.context_builder import spoken_facility
from app.utils.timeutils import to_ist, utc_now

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["dashboard"],
    dependencies=[Depends(require_session)],
)

# Added to a naive-UTC column to bucket it into the IST day the business reports on.
_IST_SHIFT = text("interval '5 hours 30 minutes'")

MAX_PAGE_SIZE = 100
MAX_WINDOW_DAYS = 365


def _ist_now() -> datetime:
    """IST wall-clock as a naive datetime, comparable with the appointment columns."""
    return to_ist(utc_now()).replace(tzinfo=None)


def _ist_midnight_as_utc() -> datetime:
    """Start of the current IST day, expressed in naive UTC.

    "Calls today" has to mean the business day. Comparing against UTC midnight would roll
    the counter over at 05:30 IST, so the morning's calls land on the previous day.
    """
    midnight_ist = to_ist(utc_now()).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_ist.astimezone(timezone.utc).replace(tzinfo=None)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so a pasted '%' or '_' is searched for, not matched with."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _as_float(value: Any) -> Optional[float]:
    """Numeric columns arrive as Decimal, which json cannot encode."""
    return None if value is None else float(value)


# --- Response models ------------------------------------------------------------------

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int

    @computed_field
    @property
    def pages(self) -> int:
        """Serialised, not just a Python property: the pager needs it to know the last page."""
        return max(1, -(-self.total // self.page_size))


class CallSummary(BaseModel):
    id: str
    call_sid: str
    campaign_id: str
    campaign_name: Optional[str] = None
    project_name: Optional[str] = None
    phone_number: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    lead_id: Optional[str] = None
    lead_status: Optional[str] = None
    customer_name: Optional[str] = None
    has_transcript: bool = False


class TranscriptTurn(BaseModel):
    """One line of the conversation, already split for the UI.

    Transcripts are stored as a single blob with 'Agent:' / 'Prospect:' prefixes. Parsing
    it in the browser would put the format in two places; a change to the agent's output
    would then silently break rendering with no test to catch it.
    """

    speaker: Literal["agent", "prospect", "unknown"]
    text: str


class LeadSummary(BaseModel):
    id: str
    customer_name: Optional[str] = None
    phone_number: Optional[str] = None
    status: str
    preferred_location: Optional[str] = None
    preferred_unit_type: Optional[str] = None
    budget: Optional[float] = None
    timeline: Optional[str] = None
    site_visit_time: Optional[datetime] = None
    callback_time: Optional[datetime] = None
    created_at: Optional[datetime] = None
    campaign_id: Optional[str] = None
    campaign_name: Optional[str] = None
    call_id: Optional[str] = None
    call_sid: Optional[str] = None


class CallDetail(CallSummary):
    transcript: List[TranscriptTurn] = Field(default_factory=list)
    transcript_text: Optional[str] = None
    lead: Optional[LeadSummary] = None


class CampaignSummary(BaseModel):
    id: str
    name: str
    status: str
    project_id: str
    project_name: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_leads_dialed: float = 0
    calls: int = 0
    completed_calls: int = 0
    leads: int = 0
    hot_leads: int = 0
    site_visits: int = 0


class UnitConfig(BaseModel):
    type: Optional[str] = None
    area: Optional[str] = None
    price: Optional[str] = None


class ProjectSummary(BaseModel):
    id: str
    name: str
    # Who the agent says it is calling from. Optional: without one the greeting names the
    # project, as it did before this field existed.
    developer_name: Optional[str] = None
    city: str
    locality: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    possession_status: Optional[str] = None
    rera_id: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    usps: List[str] = Field(default_factory=list)
    configs: List[UnitConfig] = Field(default_factory=list)
    # Flattened to one line per category by the same function that feeds the caller, so what
    # an operator reads here is what the agent says. The column also holds the richer shapes
    # scraped sources produce; those flatten to the same text, and anything that is not a
    # dict of categories reads as empty — which is exactly what the agent gets from it.
    nearby: Dict[str, str] = Field(default_factory=dict)
    campaigns: int = 0


class OverviewMetrics(BaseModel):
    window_days: int
    total_calls: int
    completed_calls: int
    failed_calls: int
    in_progress_calls: int
    connect_rate: float
    total_talk_seconds: float
    avg_duration_seconds: float
    total_leads: int
    hot_leads: int
    warm_leads: int
    cold_leads: int
    site_visits_booked: int
    callbacks_booked: int
    lead_conversion_rate: float
    active_campaigns: int
    calls_today: int
    leads_today: int


class TimeseriesPoint(BaseModel):
    date: str
    calls: int
    completed: int
    failed: int
    leads: int
    hot_leads: int
    talk_seconds: float


class FunnelStage(BaseModel):
    stage: str
    count: int


class LiveStatus(BaseModel):
    active_calls: int
    max_concurrent_calls: int
    dialed_this_minute: int
    dial_max_per_minute: int
    dialed_today: int
    dial_max_per_day: int
    in_progress: List[CallSummary]


class AppointmentItem(LeadSummary):
    kind: Literal["site_visit", "callback"]
    scheduled_at: datetime


# --- Shared query fragments -----------------------------------------------------------

_LEAD_COLUMNS = (
    Lead.id,
    Lead.customer_name,
    Lead.phone_number,
    Lead.status,
    Lead.preferred_location,
    Lead.preferred_unit_type,
    Lead.budget,
    Lead.timeline,
    Lead.site_visit_time,
    Lead.callback_time,
    Lead.created_at,
)


def _lead_from_row(row: Any) -> LeadSummary:
    return LeadSummary(
        id=str(row.id),
        customer_name=row.customer_name,
        phone_number=row.phone_number,
        status=row.status.value if row.status else LeadStatus.WARM.value,
        preferred_location=row.preferred_location,
        preferred_unit_type=row.preferred_unit_type,
        budget=_as_float(row.budget),
        timeline=row.timeline,
        site_visit_time=row.site_visit_time,
        callback_time=row.callback_time,
        created_at=row.created_at,
        campaign_id=str(row.campaign_id) if getattr(row, "campaign_id", None) else None,
        campaign_name=getattr(row, "campaign_name", None),
        call_id=str(row.call_id) if getattr(row, "call_id", None) else None,
        call_sid=getattr(row, "call_sid", None),
    )


async def _paginate(db: AsyncSession, stmt: Select, page: int, page_size: int):
    """Count and slice one statement.

    The count runs off a subquery of the same statement rather than a hand-written
    duplicate, so a filter can never be applied to the rows but forgotten in the total —
    which shows the operator "312 results" above a list that ends at 40.
    """
    total = await db.scalar(select(func.count()).select_from(stmt.order_by(None).subquery()))
    rows = (await db.execute(stmt.limit(page_size).offset((page - 1) * page_size))).all()
    return rows, int(total or 0)


def _window_start(days: int) -> datetime:
    return utc_now() - timedelta(days=days)


# --- Overview -------------------------------------------------------------------------


@router.get("/overview", response_model=OverviewMetrics)
async def overview(
    days: int = Query(default=30, ge=1, le=MAX_WINDOW_DAYS),
    campaign_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    since = _window_start(days)
    ist_midnight_utc = _ist_midnight_as_utc()

    call_filters = [Call.started_at >= since]
    if campaign_id:
        call_filters.append(Call.campaign_id == campaign_id)

    call_stats = (
        await db.execute(
            select(
                func.count(Call.id).label("total"),
                func.count(case((Call.status == CallStatus.COMPLETED, 1))).label("completed"),
                func.count(case((Call.status == CallStatus.FAILED, 1))).label("failed"),
                func.count(case((Call.status == CallStatus.IN_PROGRESS, 1))).label("in_progress"),
                func.coalesce(func.sum(Call.duration_seconds), 0).label("talk_seconds"),
                func.count(case((Call.started_at >= ist_midnight_utc, 1))).label("today"),
            ).where(*call_filters)
        )
    ).one()

    # Joined through Call so a campaign filter reaches leads, which carry no campaign of
    # their own. An inner join also excludes any orphan lead row from the totals.
    lead_filters = [Call.started_at >= since]
    if campaign_id:
        lead_filters.append(Call.campaign_id == campaign_id)

    lead_stats = (
        await db.execute(
            select(
                func.count(Lead.id).label("total"),
                func.count(case((Lead.status == LeadStatus.HOT, 1))).label("hot"),
                func.count(case((Lead.status == LeadStatus.WARM, 1))).label("warm"),
                func.count(case((Lead.status == LeadStatus.COLD, 1))).label("cold"),
                func.count(case((Lead.site_visit_time.isnot(None), 1))).label("site_visits"),
                func.count(case((Lead.callback_time.isnot(None), 1))).label("callbacks"),
                func.count(case((Lead.created_at >= ist_midnight_utc, 1))).label("today"),
            )
            .select_from(Lead)
            .join(Call, Call.lead_id == Lead.id)
            .where(*lead_filters)
        )
    ).one()

    active_campaigns = await db.scalar(
        select(func.count(Campaign.id)).where(Campaign.status == CampaignStatus.ACTIVE)
    )

    total = int(call_stats.total or 0)
    completed = int(call_stats.completed or 0)
    talk = float(call_stats.talk_seconds or 0)
    leads = int(lead_stats.total or 0)

    return OverviewMetrics(
        window_days=days,
        total_calls=total,
        completed_calls=completed,
        failed_calls=int(call_stats.failed or 0),
        in_progress_calls=int(call_stats.in_progress or 0),
        # Share of calls that produced a conversation at all. Denominator is every call
        # placed, so a run of failures is visible rather than divided away.
        connect_rate=round(completed / total * 100, 1) if total else 0.0,
        total_talk_seconds=round(talk, 1),
        avg_duration_seconds=round(talk / completed, 1) if completed else 0.0,
        total_leads=leads,
        hot_leads=int(lead_stats.hot or 0),
        warm_leads=int(lead_stats.warm or 0),
        cold_leads=int(lead_stats.cold or 0),
        site_visits_booked=int(lead_stats.site_visits or 0),
        callbacks_booked=int(lead_stats.callbacks or 0),
        lead_conversion_rate=round(leads / completed * 100, 1) if completed else 0.0,
        active_campaigns=int(active_campaigns or 0),
        calls_today=int(call_stats.today or 0),
        leads_today=int(lead_stats.today or 0),
    )


@router.get("/timeseries", response_model=List[TimeseriesPoint])
async def timeseries(
    days: int = Query(default=30, ge=1, le=MAX_WINDOW_DAYS),
    campaign_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    """Daily call and lead counts, bucketed by IST day.

    Days with no activity are filled in here rather than left out. A line chart that skips
    empty days draws a flat segment across an outage instead of a drop to zero.
    """
    since = _window_start(days)
    day = func.date(Call.started_at + _IST_SHIFT)

    filters = [Call.started_at >= since]
    if campaign_id:
        filters.append(Call.campaign_id == campaign_id)

    rows = (
        await db.execute(
            select(
                day.label("day"),
                func.count(Call.id).label("calls"),
                func.count(case((Call.status == CallStatus.COMPLETED, 1))).label("completed"),
                func.count(case((Call.status == CallStatus.FAILED, 1))).label("failed"),
                func.count(Lead.id).label("leads"),
                func.count(case((Lead.status == LeadStatus.HOT, 1))).label("hot"),
                func.coalesce(func.sum(Call.duration_seconds), 0).label("talk_seconds"),
            )
            .select_from(Call)
            .outerjoin(Lead, Call.lead_id == Lead.id)
            .where(*filters)
            .group_by(day)
        )
    ).all()

    by_day = {str(r.day): r for r in rows}
    today_ist = to_ist(utc_now()).date()
    points: List[TimeseriesPoint] = []
    for offset in range(days - 1, -1, -1):
        key = str(today_ist - timedelta(days=offset))
        row = by_day.get(key)
        points.append(
            TimeseriesPoint(
                date=key,
                calls=int(row.calls) if row else 0,
                completed=int(row.completed) if row else 0,
                failed=int(row.failed) if row else 0,
                leads=int(row.leads) if row else 0,
                hot_leads=int(row.hot) if row else 0,
                talk_seconds=round(float(row.talk_seconds), 1) if row else 0.0,
            )
        )
    return points


@router.get("/funnel", response_model=List[FunnelStage])
async def funnel(
    days: int = Query(default=30, ge=1, le=MAX_WINDOW_DAYS),
    campaign_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    since = _window_start(days)
    filters = [Call.started_at >= since]
    if campaign_id:
        filters.append(Call.campaign_id == campaign_id)

    row = (
        await db.execute(
            select(
                func.count(Call.id).label("dialed"),
                func.count(case((Call.status == CallStatus.COMPLETED, 1))).label("connected"),
                func.count(Call.lead_id).label("qualified"),
                func.count(case((Lead.status == LeadStatus.HOT, 1))).label("hot"),
                func.count(case((Lead.site_visit_time.isnot(None), 1))).label("site_visits"),
            )
            .select_from(Call)
            .outerjoin(Lead, Call.lead_id == Lead.id)
            .where(*filters)
        )
    ).one()

    return [
        FunnelStage(stage="Dialed", count=int(row.dialed or 0)),
        FunnelStage(stage="Connected", count=int(row.connected or 0)),
        FunnelStage(stage="Qualified", count=int(row.qualified or 0)),
        FunnelStage(stage="Hot", count=int(row.hot or 0)),
        FunnelStage(stage="Site visit booked", count=int(row.site_visits or 0)),
    ]


@router.get("/live", response_model=LiveStatus)
async def live(db: AsyncSession = Depends(get_db)):
    """What the system is doing right now, plus how much dial budget is left.

    active_calls comes from Redis, so it is the same figure every container enforces against
    and the same one the dial pump reads before placing a call. It used to be a module-level
    integer in whichever api process served this request, which with more than one replica
    was not a count of anything.
    """
    from app.core.config import settings

    minute_key, day_key = window_keys()
    dialed_minute = dialed_day = 0
    try:
        from app.core.queue import get_arq_pool

        redis = get_arq_pool()
        raw_minute, raw_day = await redis.mget(minute_key, day_key)
        dialed_minute = int(raw_minute or 0)
        dialed_day = int(raw_day or 0)
    except Exception as exc:  # noqa: BLE001 - a missing counter must not blank the page
        logger.warning(f"Dial quota counters unavailable for dashboard: {exc}")

    rows = (
        await db.execute(
            _call_summary_select().where(Call.status == CallStatus.IN_PROGRESS)
            .order_by(Call.started_at.desc())
            .limit(MAX_PAGE_SIZE)
        )
    ).all()

    return LiveStatus(
        active_calls=await call_slots.active(),
        max_concurrent_calls=settings.MAX_CONCURRENT_CALLS,
        dialed_this_minute=dialed_minute,
        dial_max_per_minute=settings.DIAL_MAX_PER_MINUTE,
        dialed_today=dialed_day,
        dial_max_per_day=settings.DIAL_MAX_PER_DAY,
        in_progress=[_call_from_row(r) for r in rows],
    )


# --- Calls ----------------------------------------------------------------------------


def _call_summary_select() -> Select:
    return (
        select(
            Call.id,
            Call.call_sid,
            Call.campaign_id,
            Call.phone_number,
            Call.status,
            Call.started_at,
            Call.ended_at,
            Call.duration_seconds,
            Call.lead_id,
            Campaign.name.label("campaign_name"),
            Project.name.label("project_name"),
            Lead.status.label("lead_status"),
            Lead.customer_name,
            Transcript.id.label("transcript_id"),
        )
        .select_from(Call)
        .outerjoin(Campaign, Campaign.id == Call.campaign_id)
        .outerjoin(Project, Project.id == Campaign.project_id)
        .outerjoin(Lead, Lead.id == Call.lead_id)
        .outerjoin(Transcript, Transcript.call_id == Call.id)
    )


def _call_from_row(row: Any) -> CallSummary:
    return CallSummary(
        id=str(row.id),
        call_sid=row.call_sid,
        campaign_id=str(row.campaign_id),
        campaign_name=row.campaign_name,
        project_name=row.project_name,
        phone_number=row.phone_number,
        status=row.status.value if row.status else CallStatus.IN_PROGRESS.value,
        started_at=row.started_at,
        ended_at=row.ended_at,
        duration_seconds=_as_float(row.duration_seconds),
        lead_id=str(row.lead_id) if row.lead_id else None,
        lead_status=row.lead_status.value if row.lead_status else None,
        customer_name=row.customer_name,
        has_transcript=row.transcript_id is not None,
    )


@router.get("/calls", response_model=Page[CallSummary])
async def list_calls(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=MAX_PAGE_SIZE),
    status_filter: Optional[CallStatus] = Query(default=None, alias="status"),
    campaign_id: Optional[uuid.UUID] = None,
    q: Optional[str] = Query(default=None, max_length=100),
    days: Optional[int] = Query(default=None, ge=1, le=MAX_WINDOW_DAYS),
    db: AsyncSession = Depends(get_db),
):
    stmt = _call_summary_select()
    if status_filter:
        stmt = stmt.where(Call.status == status_filter)
    if campaign_id:
        stmt = stmt.where(Call.campaign_id == campaign_id)
    if days:
        stmt = stmt.where(Call.started_at >= _window_start(days))
    if q:
        # ilike with an escaped needle: an operator pasting a number with a '%' in it
        # should search for that character, not match every row.
        needle = f"%{_escape_like(q.strip())}%"
        stmt = stmt.where(
            or_(
                Call.phone_number.ilike(needle, escape="\\"),
                Lead.customer_name.ilike(needle, escape="\\"),
                cast(Call.call_sid, SAString).ilike(needle, escape="\\"),
            )
        )

    stmt = stmt.order_by(Call.started_at.desc().nullslast())
    rows, total = await _paginate(db, stmt, page, page_size)
    return Page[CallSummary](
        items=[_call_from_row(r) for r in rows], total=total, page=page, page_size=page_size
    )


_SPEAKER_PREFIXES = {"agent": "agent", "assistant": "agent", "prospect": "prospect", "user": "prospect", "customer": "prospect"}


def _parse_transcript(raw: str) -> List[TranscriptTurn]:
    turns: List[TranscriptTurn] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        speaker: Any = "unknown"
        text_part = line
        if ":" in line:
            head, _, tail = line.partition(":")
            mapped = _SPEAKER_PREFIXES.get(head.strip().lower())
            if mapped:
                speaker, text_part = mapped, tail.strip()
        # A continuation line with no prefix belongs to whoever spoke last; starting a new
        # bubble for it would break a wrapped sentence into two speakers.
        if speaker == "unknown" and turns:
            turns[-1] = TranscriptTurn(speaker=turns[-1].speaker, text=f"{turns[-1].text} {text_part}".strip())
            continue
        turns.append(TranscriptTurn(speaker=speaker, text=text_part))
    return turns


@router.get("/calls/{call_id}", response_model=CallDetail)
async def get_call(call_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(_call_summary_select().where(Call.id == call_id))).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found")

    detail = CallDetail(**_call_from_row(row).model_dump())

    transcript_text = await db.scalar(select(Transcript.full_text).where(Transcript.call_id == call_id))
    if transcript_text:
        detail.transcript_text = transcript_text
        detail.transcript = _parse_transcript(transcript_text)

    if row.lead_id:
        lead_row = (
            await db.execute(
                select(
                    *_LEAD_COLUMNS,
                    Call.campaign_id,
                    Call.id.label("call_id"),
                    Call.call_sid,
                    Campaign.name.label("campaign_name"),
                )
                .select_from(Lead)
                .join(Call, Call.lead_id == Lead.id)
                .outerjoin(Campaign, Campaign.id == Call.campaign_id)
                .where(Lead.id == row.lead_id)
            )
        ).first()
        if lead_row:
            detail.lead = _lead_from_row(lead_row)

    return detail


# --- Leads ----------------------------------------------------------------------------


def _lead_select() -> Select:
    return (
        select(
            *_LEAD_COLUMNS,
            Call.campaign_id,
            Call.id.label("call_id"),
            Call.call_sid,
            Campaign.name.label("campaign_name"),
        )
        .select_from(Lead)
        # Outer: a lead whose call row was lost is still a person sales must ring back.
        .outerjoin(Call, Call.lead_id == Lead.id)
        .outerjoin(Campaign, Campaign.id == Call.campaign_id)
    )


@router.get("/leads", response_model=Page[LeadSummary])
async def list_leads(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=MAX_PAGE_SIZE),
    status_filter: Optional[LeadStatus] = Query(default=None, alias="status"),
    campaign_id: Optional[uuid.UUID] = None,
    q: Optional[str] = Query(default=None, max_length=100),
    has_site_visit: Optional[bool] = None,
    has_callback: Optional[bool] = None,
    days: Optional[int] = Query(default=None, ge=1, le=MAX_WINDOW_DAYS),
    db: AsyncSession = Depends(get_db),
):
    stmt = _lead_select()
    if status_filter:
        stmt = stmt.where(Lead.status == status_filter)
    if campaign_id:
        stmt = stmt.where(Call.campaign_id == campaign_id)
    if has_site_visit is not None:
        stmt = stmt.where(Lead.site_visit_time.isnot(None) if has_site_visit else Lead.site_visit_time.is_(None))
    if has_callback is not None:
        stmt = stmt.where(Lead.callback_time.isnot(None) if has_callback else Lead.callback_time.is_(None))
    if days:
        stmt = stmt.where(Lead.created_at >= _window_start(days))
    if q:
        needle = f"%{_escape_like(q.strip())}%"
        stmt = stmt.where(
            or_(
                Lead.customer_name.ilike(needle, escape="\\"),
                Lead.phone_number.ilike(needle, escape="\\"),
                Lead.preferred_location.ilike(needle, escape="\\"),
            )
        )

    stmt = stmt.order_by(Lead.created_at.desc().nullslast())
    rows, total = await _paginate(db, stmt, page, page_size)
    return Page[LeadSummary](
        items=[_lead_from_row(r) for r in rows], total=total, page=page, page_size=page_size
    )


@router.get("/leads/{lead_id}", response_model=LeadSummary)
async def get_lead(lead_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(_lead_select().where(Lead.id == lead_id))).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    return _lead_from_row(row)


class LeadUpdate(BaseModel):
    status: LeadStatus


@router.patch("/leads/{lead_id}", response_model=LeadSummary)
async def update_lead(lead_id: uuid.UUID, req: LeadUpdate, db: AsyncSession = Depends(get_db),
                      claims: SessionClaims = Depends(require_session)):
    """Let a rep correct a lead's temperature after actually speaking to them.

    Open to any signed-in user, not just admins: reclassifying a lead is the sales job this
    dashboard exists for, and it spends nothing.
    """
    lead = await db.get(Lead, lead_id)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    previous = lead.status
    lead.status = req.status
    await db.commit()
    logger.info(f"Lead {lead_id} status {previous} -> {req.status} by {claims.email}")

    row = (await db.execute(_lead_select().where(Lead.id == lead_id))).first()
    return _lead_from_row(row)


@router.get("/appointments", response_model=List[AppointmentItem])
async def appointments(
    days: int = Query(default=14, ge=1, le=90),
    include_past: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Upcoming site visits and callbacks.

    Compared against IST now, not utc_now: these two columns hold IST wall-clock time, so a
    UTC comparison would keep showing appointments that ended five and a half hours ago.
    """
    now = _ist_now()
    lower = now - timedelta(days=days) if include_past else now
    upper = now + timedelta(days=days)

    rows = (
        await db.execute(
            _lead_select().where(
                or_(
                    Lead.site_visit_time.between(lower, upper),
                    Lead.callback_time.between(lower, upper),
                )
            )
        )
    ).all()

    items: List[AppointmentItem] = []
    for row in rows:
        base = _lead_from_row(row).model_dump()
        # One lead can owe both a visit and a callback; each is its own diary entry.
        for kind, when in (("site_visit", row.site_visit_time), ("callback", row.callback_time)):
            if when is not None and lower <= when <= upper:
                items.append(AppointmentItem(**base, kind=kind, scheduled_at=when))

    items.sort(key=lambda item: item.scheduled_at)
    return items


# --- Campaigns ------------------------------------------------------------------------

_CAMPAIGN_GROUP_BY = (
    Campaign.id, Campaign.name, Campaign.status, Campaign.project_id,
    Campaign.start_date, Campaign.end_date, Campaign.total_leads_dialed,
    Project.name, Project.city, Project.locality,
)


def _campaign_rollup_select() -> Select:
    """One campaign per row, with its call and lead counts already aggregated.

    A grouped left join rather than a query per row: at 25 rows a page the N+1 version
    issued 51 round trips to a managed database in another datacentre. Shared with the
    detail route so a card and the page it opens can never disagree on the numbers.
    """
    return (
        select(
            Campaign.id,
            Campaign.name,
            Campaign.status,
            Campaign.project_id,
            Campaign.start_date,
            Campaign.end_date,
            Campaign.total_leads_dialed,
            Project.name.label("project_name"),
            Project.city,
            Project.locality,
            func.count(Call.id).label("calls"),
            func.count(case((Call.status == CallStatus.COMPLETED, 1))).label("completed_calls"),
            func.count(Call.lead_id).label("leads"),
            func.count(case((Lead.status == LeadStatus.HOT, 1))).label("hot_leads"),
            func.count(case((Lead.site_visit_time.isnot(None), 1))).label("site_visits"),
        )
        .select_from(Campaign)
        .outerjoin(Project, Project.id == Campaign.project_id)
        .outerjoin(Call, Call.campaign_id == Campaign.id)
        .outerjoin(Lead, Lead.id == Call.lead_id)
        .group_by(*_CAMPAIGN_GROUP_BY)
    )


@router.get("/campaigns", response_model=Page[CampaignSummary])
async def list_campaigns(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=MAX_PAGE_SIZE),
    status_filter: Optional[CampaignStatus] = Query(default=None, alias="status"),
    project_id: Optional[uuid.UUID] = None,
    q: Optional[str] = Query(default=None, max_length=100),
    db: AsyncSession = Depends(get_db),
):
    """Campaigns with their call and lead rollups."""
    stmt = _campaign_rollup_select()

    if status_filter:
        stmt = stmt.where(Campaign.status == status_filter)
    if project_id:
        stmt = stmt.where(Campaign.project_id == project_id)
    if q:
        needle = f"%{_escape_like(q.strip())}%"
        stmt = stmt.where(or_(Campaign.name.ilike(needle, escape="\\"), Project.name.ilike(needle, escape="\\")))

    stmt = stmt.order_by(Campaign.start_date.desc().nullslast())
    rows, total = await _paginate(db, stmt, page, page_size)

    return Page[CampaignSummary](
        items=[_campaign_from_row(r) for r in rows], total=total, page=page, page_size=page_size
    )


def _campaign_from_row(row: Any) -> CampaignSummary:
    return CampaignSummary(
        id=str(row.id),
        name=row.name,
        status=row.status.value if row.status else CampaignStatus.ACTIVE.value,
        project_id=str(row.project_id),
        project_name=row.project_name,
        city=row.city,
        locality=row.locality,
        start_date=row.start_date,
        end_date=row.end_date,
        total_leads_dialed=_as_float(row.total_leads_dialed) or 0,
        calls=int(row.calls or 0),
        completed_calls=int(row.completed_calls or 0),
        leads=int(row.leads or 0),
        hot_leads=int(row.hot_leads or 0),
        site_visits=int(row.site_visits or 0),
    )


@router.get("/campaigns/{campaign_id}", response_model=CampaignSummary)
async def get_campaign(campaign_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(_campaign_rollup_select().where(Campaign.id == campaign_id))).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")
    return _campaign_from_row(row)


class CampaignUpdate(BaseModel):
    status: CampaignStatus


@router.patch("/campaigns/{campaign_id}", response_model=CampaignSummary, dependencies=[Depends(require_admin)])
async def update_campaign(campaign_id: uuid.UUID, req: CampaignUpdate, db: AsyncSession = Depends(get_db)):
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")
    campaign.status = req.status
    await db.commit()
    # discovery.py caches campaign -> project for a day in Redis and 5 minutes in-process.
    # A campaign whose state just changed must not keep serving the old context to a live call.
    await invalidate_campaign_context(str(campaign_id))
    return await get_campaign(campaign_id, db)


@router.post("/campaigns/{campaign_id}/dial", dependencies=[Depends(require_admin)])
async def dial_campaign(
    campaign_id: uuid.UUID,
    req: DialRequest,
    db: AsyncSession = Depends(get_db),
    claims: SessionClaims = Depends(require_admin),
):
    """Add numbers to a campaign's dial queue.

    This used to place the calls itself, one background task per number, with the concurrency
    cap checked later when each media websocket opened — after Vobiz had dialled, billed us
    and rung a real person, who then had the line closed on them with no Call row written. On
    a three-slot account a list of twenty meant seventeen people were called, charged for, and
    hung up on invisibly.

    So it enqueues instead. The pump in app/services/dial_pump.py places the calls, taking a
    carrier slot before each one, and this returns as soon as the rows are written. What the
    operator gives up is the illusion that the calls went out immediately; what they get is
    that every number in the list is eventually dialled exactly once.

    Kept alongside the spreadsheet import because pasting a handful of numbers is a real thing
    to want — a callback, a number a colleague just passed on — and opening Excel for three of
    them is not.
    """
    report = await dial_queue.enqueue(
        db, campaign_id, req.phone_numbers, requested_by=claims.email
    )
    return report.as_response()


# --- Projects -------------------------------------------------------------------------


@router.get("/projects", response_model=Page[ProjectSummary])
async def list_projects(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=MAX_PAGE_SIZE),
    city: Optional[str] = Query(default=None, max_length=100),
    q: Optional[str] = Query(default=None, max_length=100),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            Project.id, Project.name, Project.developer_name, Project.city, Project.locality,
            Project.min_price, Project.max_price, Project.possession_status,
            Project.rera_id, Project.amenities, Project.usps, Project.config_json,
            Project.nearby_facilities,
            func.count(Campaign.id).label("campaigns"),
        )
        .select_from(Project)
        .outerjoin(Campaign, Campaign.project_id == Project.id)
        .group_by(
            Project.id, Project.name, Project.developer_name, Project.city, Project.locality,
            Project.min_price, Project.max_price, Project.possession_status,
            Project.rera_id, Project.amenities, Project.usps, Project.config_json,
            # jsonb has an equality operator, so it groups; plain json would not.
            Project.nearby_facilities,
        )
    )
    if city:
        stmt = stmt.where(Project.city.ilike(f"%{_escape_like(city)}%", escape="\\"))
    if q:
        needle = f"%{_escape_like(q.strip())}%"
        stmt = stmt.where(
            or_(
                Project.name.ilike(needle, escape="\\"),
                Project.locality.ilike(needle, escape="\\"),
                Project.rera_id.ilike(needle, escape="\\"),
            )
        )

    stmt = stmt.order_by(Project.name.asc())
    rows, total = await _paginate(db, stmt, page, page_size)
    return Page[ProjectSummary](
        items=[_project_from_row(r) for r in rows], total=total, page=page, page_size=page_size
    )


def _nearby_from_column(value: Any) -> Dict[str, str]:
    """The location facts the agent actually has, as text.

    Not a dict of categories means the agent reads nothing from this column — see the
    isinstance guard in context_builder — so showing it as empty is the honest answer rather
    than a lossy one. It also gives the operator an empty editor to fill in, which is the
    repair for a project in that state.
    """
    if not isinstance(value, dict):
        return {}
    return {str(k): spoken_facility(v) for k, v in value.items() if spoken_facility(v)}


def _project_from_row(row: Any) -> ProjectSummary:
    raw_configs = row.config_json if isinstance(row.config_json, list) else []
    return ProjectSummary(
        id=str(row.id),
        name=row.name,
        developer_name=row.developer_name,
        city=row.city,
        locality=row.locality,
        min_price=_as_float(row.min_price),
        max_price=_as_float(row.max_price),
        possession_status=row.possession_status,
        rera_id=row.rera_id,
        amenities=list(row.amenities or []),
        usps=list(row.usps or []),
        configs=[UnitConfig(**{k: c.get(k) for k in ("type", "area", "price")}) for c in raw_configs if isinstance(c, dict)],
        nearby=_nearby_from_column(row.nearby_facilities),
        campaigns=int(row.campaigns or 0),
    )


@router.get("/projects/{project_id}", response_model=ProjectSummary)
async def get_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    campaigns = await db.scalar(
        select(func.count(Campaign.id)).where(Campaign.project_id == project_id)
    )
    # The ORM object and a query row expose the same attribute names, so one mapper serves
    # both — the count is the only field the entity does not carry.
    return _project_from_row(
        SimpleNamespace(
            **{
                name: getattr(project, name)
                for name in (
                    "id", "name", "developer_name", "city", "locality",
                    "min_price", "max_price",
                    "possession_status", "rera_id", "amenities", "usps", "config_json",
                    "nearby_facilities",
                )
            },
            campaigns=campaigns,
        )
    )


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    developer_name: Optional[str] = Field(default=None, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    locality: str = Field(min_length=1, max_length=100)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    possession_status: Optional[str] = Field(default=None, max_length=200)
    rera_id: Optional[str] = Field(default=None, max_length=100)
    amenities: List[str] = Field(default_factory=list)
    usps: List[str] = Field(default_factory=list)
    # Typed rather than List[Any]. These three keys are the whole of what reaches the caller
    # — context_builder reads type, area and price and nothing else — and the agent speaks
    # this list out loud, so an unvalidated shape here becomes a wrong sentence on a call
    # rather than an error anybody sees.
    config_json: Optional[List[UnitConfig]] = None
    nearby_facilities: Optional[Dict[str, str]] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    developer_name: Optional[str] = Field(default=None, max_length=200)
    city: Optional[str] = Field(default=None, min_length=1, max_length=100)
    locality: Optional[str] = Field(default=None, min_length=1, max_length=100)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    possession_status: Optional[str] = Field(default=None, max_length=200)
    rera_id: Optional[str] = Field(default=None, max_length=100)
    amenities: Optional[List[str]] = None
    usps: Optional[List[str]] = None
    config_json: Optional[List[UnitConfig]] = None
    nearby_facilities: Optional[Dict[str, str]] = None


@router.post("/projects", response_model=ProjectSummary, dependencies=[Depends(require_admin)])
async def create_project(req: ProjectCreate, db: AsyncSession = Depends(get_db)):
    project = Project(
        name=req.name,
        developer_name=req.developer_name,
        city=req.city,
        locality=req.locality,
        min_price=req.min_price,
        max_price=req.max_price,
        possession_status=req.possession_status,
        rera_id=req.rera_id,
        amenities=req.amenities or [],
        usps=req.usps or [],
        # Plain dicts, because the column is JSONB and the ORM will not serialise a model.
        config_json=[c.model_dump() for c in req.config_json] if req.config_json is not None else None,
        nearby_facilities=req.nearby_facilities,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return _project_from_row(SimpleNamespace(
        id=project.id, name=project.name, developer_name=project.developer_name,
        city=project.city, locality=project.locality,
        min_price=project.min_price, max_price=project.max_price,
        possession_status=project.possession_status, rera_id=project.rera_id,
        amenities=project.amenities, usps=project.usps, config_json=project.config_json,
        nearby_facilities=project.nearby_facilities,
        campaigns=0,
    ))


@router.patch("/projects/{project_id}", response_model=ProjectSummary, dependencies=[Depends(require_admin)])
async def update_project(project_id: uuid.UUID, req: ProjectUpdate, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    for field, value in req.model_dump(exclude_unset=True).items():
        setattr(project, field, value)

    await db.commit()
    # Every campaign selling this project, not the project id. The cache is keyed by
    # campaign, so passing the project id here matched nothing and an edited price kept
    # being spoken on live calls for another day.
    cleared = await invalidate_project_everywhere(db, project_id)
    logger.info(f"Project {project_id} edited; cleared context for {cleared} campaign(s)")
    return await get_project(project_id, db)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
async def delete_project(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    campaign_count = await db.scalar(
        select(func.count(Campaign.id)).where(Campaign.project_id == project_id)
    )
    if campaign_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project has {campaign_count} campaign(s); delete or reassign them first",
        )

    # Resolved before the delete: afterwards the campaigns are gone and there is nothing
    # left to look up. The 409 above means there should be none, and clearing zero is the
    # correct outcome — but this must not be the place that assumes it.
    cleared = await invalidate_project_everywhere(db, project_id)
    await db.execute(delete(Project).where(Project.id == project_id))
    await db.commit()
    if cleared:
        logger.info(f"Project {project_id} deleted; cleared context for {cleared} campaign(s)")


# --- Campaign delete ------------------------------------------------------------------


@router.delete("/campaigns/{campaign_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
async def delete_campaign(campaign_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")

    call_count = await db.scalar(
        select(func.count(Call.id)).where(Call.campaign_id == campaign_id)
    )
    if call_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Campaign has {call_count} call record(s); archive it instead of deleting",
        )

    await db.execute(delete(Campaign).where(Campaign.id == campaign_id))
    await db.commit()
    await invalidate_campaign_context(str(campaign_id))
