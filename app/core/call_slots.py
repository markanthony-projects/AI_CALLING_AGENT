"""How many calls may be in flight at once, counted where every process can see it.

The account this dials through allows three simultaneous calls. Two things went wrong with
counting them in a module-level integer.

The first is that it was the wrong place to check. The counter was read when the media
websocket opened, which is after Vobiz has dialled, billed us, and rung a real person's
phone. They answered and got the line closed on them, and because the check returned before
the Call row was written there was no record that it had happened at all. Dialling twenty
numbers on a three-slot account meant seventeen people were called, charged for, and hung up
on invisibly. The gate has to sit in front of the dial, not in front of the audio.

The second is that an integer in one process is not a count of anything. Two api containers
would each admit three calls, and a restart resets it to zero while the carrier still has
calls up.

So the count lives in Redis, as a sorted set scored by the moment each call was admitted.
Scoring it that way makes it self-healing, which matters more here than it sounds: a process
that dies mid-call cannot run its own cleanup, and a leaked slot is permanent otherwise —
three of them and the system stops dialling forever with nothing in the logs to say why.
Entries older than the longest a call can possibly run are dropped on every read, so the
worst case for a crash is that a slot is unavailable for the length of one call.

The check and the insert run as one Lua script rather than as separate commands. That is not
an optimisation: read-then-write across two round trips lets two processes both see the last
free slot, which puts four calls on a three-call account.

Redis being unavailable is treated as no capacity rather than unlimited capacity. It makes a
Redis outage look like a stalled campaign, which is recoverable, instead of a flood of calls
nobody is metering, which is not.
"""

from typing import Optional

from loguru import logger

from app.core.config import settings
from app.services.discovery import get_redis_client

_KEY = "calls:inflight"


def _stale_before(now: float) -> float:
    """Scores at or below this belong to calls that cannot still be running.

    MAX_CALL_DURATION_SECS is the hard cap the pipeline enforces on itself, so a slot older
    than that plus a margin was leaked by a process that died. The margin covers the gap
    between a slot being taken and the call actually starting — the dial, the carrier ringing
    the phone, and the answer webhook.
    """
    from app.services.agent import MAX_CALL_DURATION_SECS

    return now - (MAX_CALL_DURATION_SECS + 120)


# One round trip, and atomic.
#
# This was five separate awaits — TIME, ZREMRANGEBYSCORE, ZSCORE, ZCARD, ZADD — with a comment
# claiming they were one. They were not, and the gap between the ZCARD and the ZADD is a real
# race: two processes both read two-of-three and both add, putting four calls on a three-call
# account. That is exactly the failure this module exists to prevent, so it cannot be left to
# chance about interleaving.
#
# EVAL runs the whole thing inside Redis with nothing able to interleave. redis.call('TIME')
# is used rather than a clock passed in, for the same reason the Python version called
# client.time(): several containers write these scores, and a few seconds of drift between
# their clocks would expire live slots or keep dead ones.
_ACQUIRE = """
local now = tonumber(redis.call('TIME')[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[2]))
if redis.call('ZSCORE', KEYS[1], ARGV[1]) == false
   and redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
    return 0
end
redis.call('ZADD', KEYS[1], now, ARGV[1])
return 1
"""

# Same sweep, no write. Separate from _ACQUIRE so a read cannot accidentally take a slot.
_ACTIVE = """
local now = tonumber(redis.call('TIME')[1])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[1]))
return redis.call('ZCARD', KEYS[1])
"""


def _stale_after() -> float:
    """How old a slot may get before it is assumed leaked.

    MAX_CALL_DURATION_SECS is the hard cap the pipeline enforces on itself, so a slot older
    than that plus a margin was left behind by a process that died. The margin covers the gap
    between a slot being taken and the call actually starting — the dial, the carrier ringing
    the phone, and the answer webhook.
    """
    from app.services.agent import MAX_CALL_DURATION_SECS

    return MAX_CALL_DURATION_SECS + 120


async def acquire(call_sid: str) -> bool:
    """Take a slot for this call, or return False if the carrier is at capacity.

    Idempotent per call_sid: re-acquiring for a call that already holds a slot refreshes its
    score and stays within the same slot, so a retried answer webhook cannot consume two.
    """
    try:
        client = await get_redis_client()
        taken = await client.eval(
            _ACQUIRE, 1, _KEY, call_sid, _stale_after(), settings.MAX_CONCURRENT_CALLS
        )
        return bool(int(taken))
    except Exception as e:
        # No capacity, not unlimited capacity. A stalled campaign is recoverable; an
        # unmetered flood of billed calls is not.
        logger.error(f"[{call_sid}] Could not reserve a call slot, refusing to dial: {e}")
        return False


async def release(call_sid: str) -> None:
    """Give the slot back. Safe to call for a call that never held one."""
    try:
        client = await get_redis_client()
        await client.zrem(_KEY, call_sid)
    except Exception as e:
        # The score-based expiry below is what stops this becoming permanent.
        logger.warning(f"[{call_sid}] Could not release the call slot: {e}")


async def active() -> int:
    """Calls in flight right now, across every process."""
    try:
        client = await get_redis_client()
        return int(await client.eval(_ACTIVE, 1, _KEY, _stale_after()))
    except Exception as e:
        logger.warning(f"Could not read the in-flight call count: {e}")
        # Reported as full for the same reason acquire refuses: whatever reads this is
        # deciding whether to spend money.
        return settings.MAX_CONCURRENT_CALLS


async def free_slots() -> int:
    """How many more calls may be placed right now. Never negative."""
    return max(0, settings.MAX_CONCURRENT_CALLS - await active())


async def held(limit: int = 50) -> list:
    """The call_sids currently holding a slot, oldest first. For diagnostics only."""
    try:
        client = await get_redis_client()
        rows = await client.zrange(_KEY, 0, limit - 1)
    except Exception:
        return []
    return [r.decode() if isinstance(r, bytes) else str(r) for r in rows]


async def reset() -> Optional[int]:
    """Drop every slot. For an operator who knows the carrier has no calls up.

    Exists because the self-healing expiry is bounded by the longest possible call, and an
    operator who is certain the line is clear should not have to wait it out.
    """
    try:
        client = await get_redis_client()
        count = int(await client.zcard(_KEY))
        await client.delete(_KEY)
        logger.warning(f"Call slots reset by hand; {count} were held")
        return count
    except Exception as e:
        logger.error(f"Could not reset call slots: {e}")
        return None
