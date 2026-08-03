"""What is left of the LLM's per-minute token allowance, shared across the process.

Every Groq response carries the answer in its headers, but only the call that received it
could see them. So the dialer had no idea it was about to start a conversation the account
could not pay for — and on a throttled account the very first thing that stalls is the
opening line, before the prospect has heard who is calling.

The budget refills continuously, so a reading goes stale within seconds. It is stored with
the moment it was taken and extrapolated forward at the provider's own refill rate rather
than trusted as-is; treating a snapshot as current would refuse dials for a minute after
one busy call.
"""

import time
from typing import Optional

from loguru import logger
from redis.exceptions import RedisError

from app.core.queue import get_arq_pool

_KEY = "llm:budget"
# Long enough to survive a gap between calls, short enough that a reading nobody has
# refreshed stops being used. Past this the dialer treats the budget as unknown and allows
# the dial, which is the right default: we only ever had this information by accident.
_TTL = 300


async def record_budget(remaining: int, limit: int) -> None:
    """Publish a rate-limit reading. Never raises — this runs inside a live call."""
    try:
        redis = get_arq_pool()
        await redis.hset(_KEY, mapping={"remaining": remaining, "limit": limit, "at": time.time()})
        await redis.expire(_KEY, _TTL)
    except (RedisError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug(f"Could not publish the LLM budget ({exc}); dialing will not gate on it")


async def tokens_available() -> Optional[float]:
    """Best estimate of the tokens left this minute, or None when we do not know.

    None is not zero. No reading means no basis to refuse, and refusing to dial on missing
    telemetry would take the whole campaign down the first time Redis blinked.
    """
    try:
        raw = await get_arq_pool().hgetall(_KEY)
    except (RedisError, RuntimeError, OSError, AttributeError) as exc:
        logger.debug(f"LLM budget unreadable ({exc}); dialing without it")
        return None
    if not raw:
        return None

    def _get(key: str) -> Optional[float]:
        value = raw.get(key) or raw.get(key.encode())
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    remaining, limit, at = _get("remaining"), _get("limit"), _get("at")
    if remaining is None or limit is None or at is None or limit <= 0:
        return None

    # The provider's bucket refills at limit/60 per second. Without this the reading only
    # ever falls, and one busy call would block dialing until the key expired.
    refilled = remaining + max(0.0, time.time() - at) * (limit / 60.0)
    return min(limit, refilled)
