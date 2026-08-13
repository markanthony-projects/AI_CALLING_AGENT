"""Close out Call rows whose session ended without anyone writing the ending down.

A Call row is created when the websocket opens and updated in the `finally` block of
_handle_call. Anything that stops that `finally` from running — the container being
replaced mid-call, an OOM kill, a wedged pipeline that outlives the shutdown grace period —
leaves the row saying IN_PROGRESS for ever. Nothing in the system ever looked at those rows
again, so the dashboard kept showing a call that had been over for days, and the operator
could not tell that from a call genuinely in progress right now.

Both of those happened on the same afternoon: Sarvam ran out of credits, the pipeline
failed to tear itself down, and the box was restarted by hand to stop it.

Age is the discriminator rather than process state. Reaping "everything IN_PROGRESS at
startup" is only correct while exactly one process serves calls; the moment there are two,
restarting one would mark the other's live calls as failed. A row older than any call could
possibly be is safe to close no matter who is running.

Note this only corrects the record. It cannot hang up a phone leg or free a concurrency
slot — those live in the process that owns the websocket, which by then is usually gone.
The duration cap in app/services/agent.py is what prevents the leg from running on.
"""

from datetime import timedelta

from loguru import logger
from sqlalchemy import select, update

from app.core.database import AsyncSessionLocal
from app.models.db import Call, CallStatus
from app.utils.timeutils import utc_now

# Comfortably past MAX_CALL_DURATION_SECS (10 minutes) so a call the duration cap is about
# to end is never reaped out from under the process still holding it — that would race the
# real ending and could overwrite a COMPLETED with a FAILED.
STALE_AFTER = timedelta(minutes=30)

# Runs on a timer as well as at startup, because the common cause is a process that died
# and therefore is not around to clean up after itself on the way back.
SWEEP_EVERY_SECONDS = 900


async def reap_stale_calls(stale_after: timedelta = STALE_AFTER) -> int:
    """Mark long-abandoned IN_PROGRESS rows as FAILED. Returns how many were closed.

    ended_at is set to now rather than left null so duration_seconds is not silently wrong,
    and so the dashboard sorts these alongside real endings. The duration it produces is an
    upper bound on the real one, which is the honest direction to be wrong in.
    """
    cutoff = utc_now() - stale_after
    async with AsyncSessionLocal() as db:
        stale = (
            await db.execute(
                select(Call.id, Call.call_sid, Call.started_at).where(
                    Call.status == CallStatus.IN_PROGRESS, Call.started_at < cutoff
                )
            )
        ).all()
        if not stale:
            return 0

        ended_at = utc_now()
        for row in stale:
            await db.execute(
                update(Call)
                .where(Call.id == row.id)
                .values(
                    status=CallStatus.FAILED,
                    ended_at=ended_at,
                    duration_seconds=(ended_at - row.started_at).total_seconds(),
                )
            )
        await db.commit()

    for row in stale:
        logger.warning(
            f"[{row.call_sid}] Call was left IN_PROGRESS since {row.started_at:%Y-%m-%d %H:%M} "
            f"and never finalised; recording it as FAILED"
        )
    logger.warning(f"Reaped {len(stale)} abandoned call record(s)")
    return len(stale)
