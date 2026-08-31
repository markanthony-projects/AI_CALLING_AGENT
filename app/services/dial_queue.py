"""The one way a number gets onto a dial queue.

There were two ways to ask this system to call someone, and only one of them was safe.

The dashboard route enqueues contacts and lets the pump place each call after reserving a
carrier slot. The API-key route at /api/v1/campaigns/{id}/dial/vobiz did not: it created one
background task per number, up to five hundred of them, and asked Vobiz to dial every one
immediately. The concurrency cap was still enforced — but at the media websocket, which
opens after the carrier has dialled, billed us and rung a real person. On a three-slot
account that meant everyone past the third was called, charged for, and hung up on with no
Call row written to say it had happened.

That is the same failure the dashboard route was rebuilt to remove, still live on the other
door. Worse, the deployment guide told operators to use that door: its "place a test call"
instructions were curl commands against it.

So both routes now call this. Not because the duplication was ugly, but because two
independent copies of "who may be dialled" drift, and the version that drifts is the one
nobody is looking at. There is exactly one implementation, and it always goes through the
queue.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Protocol, Sequence

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ratelimit import reserve_dial_quota, reserve_llm_headroom
from app.models.db import Campaign, CampaignStatus, Contact, ContactStatus, Suppression
from app.services.dial_pump import dial_forecast, eligible
from app.utils.timeutils import utc_now


class DialTarget(Protocol):
    """One number to call. Both routes parse into a model with these two fields."""

    number: str
    name: object


@dataclass
class QueueReport:
    """What queueing these numbers actually did.

    `queued` counts rows written, which is not the same as calls that will happen: a number
    already on the campaign keeps the status it has, so re-queueing somebody who has spoken
    to the agent writes nothing and dials nobody. `will_dial` is the count that answers the
    only question the caller of this function is asking.
    """

    queued: int = 0
    already_queued: int = 0
    suppressed: int = 0
    total_numbers: int = 0
    will_dial: int = 0
    held_back: Dict[str, int] = field(default_factory=dict)

    def as_response(self) -> dict:
        return {
            "status": "queued",
            "queued": self.queued,
            "already_queued": self.already_queued,
            "suppressed": self.suppressed,
            "total_numbers": self.total_numbers,
            # The two fields worth reading. will_dial is how many calls this actually causes.
            "will_dial": self.will_dial,
            "held_back": self.held_back,
        }


async def enqueue(
    db: AsyncSession,
    campaign_id: uuid.UUID,
    targets: Sequence[DialTarget],
    *,
    requested_by: str,
) -> QueueReport:
    """Put these numbers on the campaign's queue. Places no calls itself.

    The pump in app/services/dial_pump.py dials them, taking a carrier slot before each one.
    What the caller gives up is the illusion that the calls went out immediately; what they
    get is that every number is dialled exactly once and never past the carrier's limit.
    """
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found")
    if campaign.status != CampaignStatus.ACTIVE:
        # Pausing a campaign has to actually stop it, or the button means nothing. Queueing
        # against a paused campaign would look like it worked and then never dial.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Campaign is {campaign.status.value}; set it ACTIVE before dialing",
        )

    # Reserved here even though the pump is what spends the money. These are daily and
    # per-minute ceilings on how much may be queued in one go, and refusing at the point the
    # request arrived is the only place the caller can act on it.
    await reserve_dial_quota(len(targets))
    # The separate question of whether the LLM can answer a greeting at all.
    await reserve_llm_headroom()

    numbers = [t.number for t in targets]
    blocked = set(
        (
            await db.execute(
                select(Suppression.phone_number).where(Suppression.phone_number.in_(numbers))
            )
        )
        .scalars()
        .all()
    )

    rows = [
        {
            "id": uuid.uuid4(),
            "campaign_id": campaign_id,
            "phone_number": target.number,
            "name": target.name,
            "status": ContactStatus.DND if target.number in blocked else ContactStatus.PENDING,
            "last_outcome": "on the do-not-call list" if target.number in blocked else None,
        }
        for target in targets
    ]
    # A number already on this campaign's queue keeps whatever state it has. Sending the same
    # list twice must not reset somebody who has already been called back to PENDING.
    added = len(
        (
            await db.execute(
                pg_insert(Contact)
                .values(rows)
                .on_conflict_do_nothing(constraint="uq_contacts_campaign_phone")
                .returning(Contact.id)
            )
        )
        .scalars()
        .all()
    )
    await db.commit()

    # Writing a row is not the same as queueing a call, and the difference is invisible from
    # outside. A number that has already spoken to the agent, or used up its attempts, or been
    # marked do-not-call, keeps that status — the insert conflicts and does nothing, and the
    # pump will never pick it up. Reporting only what was inserted let this answer "queued"
    # for a list that would place no calls at all, which is precisely the case an operator
    # cannot tell apart from a broken dialer.
    #
    # Asked with the pump's own predicate rather than a copy of it, so the two cannot drift
    # into disagreeing about whether a number is going to be called.
    verdicts = (
        await db.execute(
            select(Contact.status, eligible(utc_now()).label("dialable")).where(
                Contact.campaign_id == campaign_id, Contact.phone_number.in_(numbers)
            )
        )
    ).all()
    will_dial, held_back = dial_forecast(verdicts)

    report = QueueReport(
        queued=added,
        already_queued=len(rows) - added,
        suppressed=len(blocked),
        total_numbers=len(rows),
        will_dial=will_dial,
        held_back=held_back,
    )
    logger.info(
        f"{requested_by} queued {report.queued} number(s) on campaign {campaign_id} "
        f"({report.already_queued} already queued, {report.suppressed} suppressed); "
        f"{report.will_dial} of {report.total_numbers} will be dialled"
        + (f", held back: {report.held_back}" if report.held_back else "")
    )
    return report
