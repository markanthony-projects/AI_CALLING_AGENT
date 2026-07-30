"""Ceilings on outbound dialing.

The API key is a bearer secret: whoever holds it can dial. MAX_CALLS caps how many media
streams run at once, but Vobiz bills at dial time and that check only runs when the stream
opens — so a runaway loop or a leaked key could place thousands of billed calls of which
eight got audio. These ceilings sit in front of the dial instead.

Counters live in Redis so one limit holds across every replica; a per-process counter would
silently multiply by the number of workers. Redis being unavailable fails the request
closed, because refusing to dial costs a retry while dialing with no ceiling costs money.
"""

import time
from datetime import timedelta
from typing import Optional, Tuple

from fastapi import HTTPException, status
from loguru import logger
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.queue import get_arq_pool
from app.utils.timeutils import to_ist, utc_now

_MINUTE_TTL = 120
_DAY_TTL = 172_800


def window_keys() -> Tuple[str, str]:
    """The current minute, and the current day in IST — the day sales actually bills against."""
    minute = int(time.time()) // 60
    day = to_ist(utc_now()).date().isoformat()
    return f"dial:minute:{minute}", f"dial:day:{day}"


def seconds_to_ist_midnight() -> int:
    now = to_ist(utc_now())
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


async def _reserve(redis, key: str, count: int, limit: int, ttl: int) -> bool:
    """Fixed-window counter. Over-limit reservations are handed back, never left charged."""
    used = await redis.incrby(key, count)
    if used == count:
        await redis.expire(key, ttl)
    if used > limit:
        await redis.decrby(key, count)
        return False
    return True


async def _charge(count: int) -> Optional[Tuple[str, int]]:
    """Charge both windows. Returns None when the batch fits, else (reason, retry_after)."""
    redis = get_arq_pool()
    minute_key, day_key = window_keys()

    if not await _reserve(redis, minute_key, count, settings.DIAL_MAX_PER_MINUTE, _MINUTE_TTL):
        return f"Dial rate limit reached ({settings.DIAL_MAX_PER_MINUTE}/minute)", 60

    if not await _reserve(redis, day_key, count, settings.DIAL_MAX_PER_DAY, _DAY_TTL):
        # The minute window already accepted this batch; give that budget back or a caller
        # who hits the daily ceiling also loses the rest of their minute.
        await redis.decrby(minute_key, count)
        return (
            f"Daily dial ceiling reached ({settings.DIAL_MAX_PER_DAY}/day)",
            seconds_to_ist_midnight(),
        )

    return None


async def reserve_dial_quota(count: int) -> None:
    """Charge `count` numbers against both ceilings, or reject the batch whole.

    Counts numbers rather than requests: one request carries up to MAX_DIAL_BATCH of them, so
    a per-request limit would bound nothing. A batch that does not fit is rejected entirely
    and consumes no quota, so a rejected caller cannot drain the window for everyone else.
    """
    try:
        verdict = await _charge(count)
    except (RedisError, RuntimeError, OSError) as exc:
        logger.error(f"Dial quota unavailable ({exc}); refusing to dial without a ceiling")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dial quota unavailable; try again shortly",
        ) from exc

    if verdict is None:
        return

    reason, retry_after = verdict
    logger.warning(f"Dial rejected: {reason} (batch of {count})")
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=reason,
        headers={"Retry-After": str(retry_after)},
    )
