"""Draw from the contact queue at exactly the rate the carrier can carry.

This replaces firing every number in a dial request at once. That loop placed as many calls
as the request contained, and the concurrency cap was checked when each media websocket
opened — after Vobiz had dialled, billed us, and rung a real person's phone. On a three-slot
account a list of twenty meant seventeen people were called, charged for, and hung up on with
no record that it had happened.

A pull model rather than a push one. Nothing schedules a number for a particular moment;
instead this runs on a short timer, asks how many slots are free, and takes that many. That
matters for the failure cases more than for the happy path:

  a worker dies mid-run   Nothing is lost. The rows are still PENDING in Postgres and the
                          next tick picks them up. A push model would have to remember what
                          it had already handed out.

  two workers run         SELECT ... FOR UPDATE SKIP LOCKED means each row is claimed by
                          exactly one of them, without either waiting on the other.

  the carrier is full     Nothing is dialled and nothing is marked. The queue simply does not
                          move, which is the correct behaviour and needs no special case.

  a campaign is paused    It stops being selected. No queue to drain, no jobs to cancel.

Business hours are enforced here rather than at import, because a list uploaded at 9 PM
should dial in the morning, not be rejected.
"""

import uuid
from collections import Counter
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from loguru import logger
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import call_slots
from app.core.database import AsyncSessionLocal
from app.models.db import (
    RETRIABLE_CONTACT_STATUSES,
    Campaign,
    CampaignStatus,
    Contact,
    ContactStatus,
    Suppression,
)
from app.services.call_context import remember_customer_name, remember_dialed_number
from app.services.dialer import trigger_vobiz_call
from app.utils.timeutils import is_within_business_hours, to_ist, utc_now

# Dials placed per contact before it is EXHAUSTED. Very few people answer a first attempt
# from an unknown number; stopping at one throws away most of a reachable list.
MAX_DIAL_ATTEMPTS = 3

# Waits before each retry, indexed by the attempt just completed. Widening on purpose: a
# second try two hours later catches someone who was driving, and the third the next day
# catches someone who was travelling. Calling back in five minutes catches nobody and reads
# as harassment.
RETRY_BACKOFF = (timedelta(hours=2), timedelta(days=1))


def next_attempt_after(attempts: int, now=None) -> Optional[object]:
    """When a contact with this many attempts should be tried again, or None if never.

    None means EXHAUSTED. Returned rather than raising because the caller is deciding a
    status, and "no further attempt" is an ordinary outcome, not an error.
    """
    if attempts >= MAX_DIAL_ATTEMPTS or attempts < 1:
        return None
    gap = RETRY_BACKOFF[min(attempts - 1, len(RETRY_BACKOFF) - 1)]
    return (now or utc_now()) + gap


def eligible(now) -> object:
    """The SQL condition for a contact the pump may dial right now."""
    return and_(
        or_(
            Contact.status == ContactStatus.PENDING,
            Contact.status.in_(RETRIABLE_CONTACT_STATUSES),
        ),
        # Null means "as soon as you get to it", which is what a fresh import wants.
        or_(Contact.next_attempt_at.is_(None), Contact.next_attempt_at <= now),
        Contact.attempts < MAX_DIAL_ATTEMPTS,
    )


def dial_forecast(verdicts: Sequence[Tuple[ContactStatus, object]]) -> Tuple[int, Dict[str, int]]:
    """How many of these contacts the pump will call, and why it will not call the rest.

    Takes rows of (status, dialable) where dialable is `eligible()` evaluated by the database
    for that row, so this cannot disagree with what the pump selects.

    It exists because adding a contact and queueing a call are different things, and from
    outside they look identical. A number that has already spoken to the agent, or used up
    its attempts, or been marked do-not-call, keeps that status when it is queued again — the
    insert conflicts, nothing changes, and no call is ever placed. An operator reading
    "queued" cannot tell that apart from a dialer that has stopped working.

    Anything the database could not judge counts as held back rather than dialable. Guessing
    the optimistic way here would restore the exact silence this replaces.
    """
    will_dial = sum(1 for _, dialable in verdicts if dialable)
    held_back = Counter(status.value for status, dialable in verdicts if not dialable)
    return will_dial, dict(held_back)


async def claim(db: AsyncSession, campaign_id, limit: int, now) -> List[Contact]:
    """Take up to `limit` contacts off the queue and mark them DIALING.

    SKIP LOCKED is what makes this safe to run in more than one worker: a row another
    transaction has claimed is passed over rather than waited for, so two pumps never dial
    the same number and neither blocks.

    The status is flipped inside the same transaction that selected them. Leaving them
    PENDING until the dial returned would let the next tick — five seconds later — pick them
    up again and dial everyone twice.
    """
    if limit <= 0:
        return []

    rows = (
        await db.execute(
            select(Contact)
            .where(Contact.campaign_id == campaign_id, eligible(now))
            # Oldest first, so a list is worked in the order it was uploaded and a retry does
            # not jump the queue ahead of numbers never tried at all.
            .order_by(Contact.next_attempt_at.asc().nullsfirst(), Contact.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
    ).scalars().all()

    for contact in rows:
        contact.status = ContactStatus.DIALING
        contact.attempts = (contact.attempts or 0) + 1
        contact.last_attempt_at = now
    return list(rows)


async def suppressed(db: AsyncSession, numbers: List[str]) -> set:
    """Which of these numbers must never be dialled, by any campaign.

    Checked at dial time and not only at import, because a request to stop being called
    arrives during a run and has to take effect on the numbers already queued behind it.
    """
    if not numbers:
        return set()
    rows = await db.execute(
        select(Suppression.phone_number).where(Suppression.phone_number.in_(numbers))
    )
    return {r[0] for r in rows}


async def active_campaign_ids(db: AsyncSession) -> List[uuid.UUID]:
    """Campaigns with work outstanding, fewest remaining first.

    Ordered so a campaign close to finishing is cleared rather than left with a long tail
    while a newly uploaded list soaks up every slot. With one campaign running — the usual
    case — the order costs nothing.
    """
    now = utc_now()
    remaining = func.count(Contact.id).label("remaining")
    rows = await db.execute(
        select(Campaign.id, remaining)
        .join(Contact, Contact.campaign_id == Campaign.id)
        .where(Campaign.status == CampaignStatus.ACTIVE, eligible(now))
        .group_by(Campaign.id)
        .order_by(remaining.asc())
    )
    return [r[0] for r in rows]


async def dial_due_contacts() -> int:
    """One tick. Returns how many calls were placed.

    Every failure path here has to leave the queue in a state the next tick can recover from,
    because this runs unattended every few seconds and nobody reads its return value.
    """
    if not is_within_business_hours(to_ist(utc_now())):
        return 0

    slots = await call_slots.free_slots()
    if slots <= 0:
        return 0

    placed = 0
    async with AsyncSessionLocal() as db:
        for campaign_id in await active_campaign_ids(db):
            if slots <= 0:
                break
            now = utc_now()
            claimed = await claim(db, campaign_id, slots, now)
            if not claimed:
                continue

            # One query for the whole batch rather than one per contact.
            blocked = await suppressed(db, [c.phone_number for c in claimed])
            for contact in claimed:
                if contact.phone_number in blocked:
                    contact.status = ContactStatus.DND
                    contact.last_outcome = "on the suppression list"
                    # The attempt increment in claim() was not a dial. Undo it so the number
                    # is not recorded as having been called.
                    contact.attempts = max(0, (contact.attempts or 1) - 1)
                    contact.last_attempt_at = None

            to_dial = [c for c in claimed if c.status == ContactStatus.DIALING]
            # Committed before a single dial goes out. If this process dies during the dials
            # below, the rows already say DIALING and the reaper will time them out; the
            # alternative is a crash that loses the claim and dials everyone twice.
            await db.commit()

            for contact in to_dial:
                if not await _place(db, contact):
                    continue
                placed += 1
                slots -= 1
                if slots <= 0:
                    break
            await db.commit()

    if placed:
        logger.info(f"Dial pump placed {placed} call(s)")
    return placed


async def _place(db: AsyncSession, contact: Contact) -> bool:
    """Reserve a slot and ask Vobiz to dial. False if the call was not placed.

    The slot is taken before the dial and released by whatever ends the call. Taking it after
    would reopen the hole this module exists to close.
    """
    call_sid = str(uuid.uuid4())
    if not await call_slots.acquire(call_sid):
        # The carrier filled up between the count and here. Put it back untouched — the
        # attempt was never made, so it must not be charged one.
        contact.status = ContactStatus.PENDING
        contact.attempts = max(0, (contact.attempts or 1) - 1)
        contact.last_attempt_at = None
        return False

    await remember_dialed_number(call_sid, contact.phone_number)
    await remember_customer_name(call_sid, contact.name)
    # Carried through Redis so the answer webhook can attribute the call to this row without
    # matching on a phone number that may sit in several campaigns.
    await remember_contact(call_sid, str(contact.id))

    try:
        ok = await trigger_vobiz_call(contact.phone_number, str(contact.campaign_id), call_sid)
    except Exception as e:
        ok = False
        logger.error(f"[{call_sid}] Dial raised for {contact.phone_number}: {e}")

    if not ok:
        await call_slots.release(call_sid)
        contact.status = ContactStatus.FAILED
        contact.last_outcome = "the carrier refused the dial"
        contact.next_attempt_at = next_attempt_after(contact.attempts or 1)
        if contact.next_attempt_at is None:
            contact.status = ContactStatus.EXHAUSTED
        return False

    logger.info(
        f"[{call_sid}] Dialling {contact.phone_number} "
        f"(attempt {contact.attempts} of {MAX_DIAL_ATTEMPTS})"
    )
    return True


# --- writing the outcome back ---------------------------------------------------------


async def record_outcome(contact_id: Optional[str], call_status, *, answered_words: int = 0) -> None:
    """Move a contact out of DIALING once its call has finished.

    Called from the same place that finalises the Call row, so the queue and the call history
    cannot disagree. Silent when there is no contact: calls placed by hand, and every call
    made before the queue existed, have none.
    """
    if not contact_id:
        return

    from app.models.db import CallStatus

    async with AsyncSessionLocal() as db:
        contact = await db.get(Contact, uuid.UUID(str(contact_id)))
        if contact is None:
            logger.warning(f"Call finished for contact {contact_id}, which no longer exists")
            return
        # An operator may have marked it DND or SKIPPED while the call was up. That decision
        # outranks the outcome of a call already in flight.
        if contact.status not in (ContactStatus.DIALING,):
            return

        if call_status is CallStatus.COMPLETED and answered_words > 0:
            contact.status = ContactStatus.COMPLETED
            contact.last_outcome = "spoke with the agent"
            contact.next_attempt_at = None
        else:
            # A COMPLETED call with nothing said is a pickup that produced no conversation,
            # which is the same thing to a dial list as a ring-out.
            contact.status = ContactStatus.NO_ANSWER
            contact.last_outcome = (
                "answering machine" if call_status is CallStatus.MACHINE
                else "no conversation" if call_status is CallStatus.COMPLETED
                else "the call did not complete"
            )
            contact.next_attempt_at = next_attempt_after(contact.attempts or 1)
            if contact.next_attempt_at is None:
                contact.status = ContactStatus.EXHAUSTED
                contact.last_outcome = f"{contact.last_outcome}; no attempts left"
        await db.commit()


def stale_dial_verdict(attempts: int, now=None) -> Tuple[ContactStatus, Optional[object], str]:
    """What a contact whose dial never connected becomes: status, next attempt, and why.

    Separated from the query so it can be tested for the two things that were wrong with it,
    neither of which a database is needed to show: that a ring-no-answer waits the same
    widening gap as any other no-answer, and that one out of attempts ends as EXHAUSTED
    rather than sitting at NO_ANSWER for ever, never dialled but shown as still coming.
    """
    when = next_attempt_after(attempts or 1, now)
    if when is None:
        return ContactStatus.EXHAUSTED, None, "the dial never connected; no attempts left"
    return ContactStatus.NO_ANSWER, when, "the dial never connected"


async def release_stale_dialing(older_than: timedelta = timedelta(minutes=20)) -> int:
    """Move contacts stuck in DIALING out of it, on the same terms as any other no-answer.

    A dial whose call never opened a websocket — nobody picked up, the number was
    unreachable, the process died — leaves the row DIALING for ever, and DIALING is not
    eligible, so that number is silently never called again while its row looks busy rather
    than broken.

    Two things this used to get wrong, and both mattered because this is the path a genuine
    ring-no-answer takes. A call nobody picks up never opens a media stream, so it does not
    reach record_outcome at all; almost every no-answer in a campaign is finalised here.

    It set next_attempt_at to now, making the contact eligible immediately. Detection already
    takes twenty minutes plus up to fifteen more for this to run, so the widening backoff —
    two hours, then a day — was replaced for the commonest outcome by a redial roughly half
    an hour later. The number that did not answer got called back three times inside a
    morning, which is the behaviour the backoff exists to prevent.

    And it wrote NO_ANSWER regardless of how many attempts were left. A contact out of
    attempts stayed NO_ANSWER for ever: never dialled again, because eligible() stops at the
    cap, but shown on the dashboard as though it were still coming. record_outcome has always
    marked that EXHAUSTED. Two paths disagreeing about the same end state is how a queue
    starts lying about itself.

    Row by row rather than one UPDATE because the next attempt depends on how many that
    contact has already had. The volume here is whatever got stuck, which is small by
    definition.
    """
    cutoff = utc_now() - older_than
    exhausted = 0
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(Contact).where(
                    Contact.status == ContactStatus.DIALING, Contact.last_attempt_at < cutoff
                )
            )
        ).scalars().all()

        for contact in rows:
            status, when, why = stale_dial_verdict(contact.attempts)
            contact.status = status
            contact.next_attempt_at = when
            contact.last_outcome = why
            exhausted += status is ContactStatus.EXHAUSTED
        await db.commit()

    if rows:
        logger.warning(
            f"Returned {len(rows)} contact(s) stuck mid-dial to the queue"
            + (f"; {exhausted} had no attempts left" if exhausted else "")
        )
    return len(rows)


# --- carrying the contact id from the dial to the call --------------------------------

_CONTACT_TTL_SECONDS = 3600


def _contact_key(call_sid: str) -> str:
    return f"call_contact:{call_sid}"


async def remember_contact(call_sid: str, contact_id: str) -> None:
    from app.services.discovery import get_redis_client

    try:
        client = await get_redis_client()
        await client.setex(_contact_key(call_sid), _CONTACT_TTL_SECONDS, contact_id)
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not record the contact behind this dial: {e}")


async def recall_contact(call_sid: str) -> Optional[str]:
    from app.services.discovery import get_redis_client

    try:
        client = await get_redis_client()
        value = await client.get(_contact_key(call_sid))
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not recall the contact behind this dial: {e}")
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)
