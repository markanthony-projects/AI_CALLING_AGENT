"""What is left of the LLM's per-minute allowance, shared across the process.

Every response carries the answer in its headers, but only the call that received it could
see them. So the dialer had no idea it was about to start a conversation the account could
not pay for — and on a throttled account the very first thing that stalls is the opening
line, before the prospect has heard who is calling.

Both ceilings are tracked because providers bind on different ones. Groq's free tier runs
out of tokens (12,000/minute against a 3,400-token request); Cerebras's runs out of
requests first (5/minute, against a call that needs six to ten). Watching only tokens would
have declared Cerebras healthy right up until the greeting stalled.

The allowance refills continuously, so a reading goes stale within seconds. It is stored
with the moment it was taken and extrapolated forward at the provider's own refill rate
rather than trusted as-is; treating a snapshot as current would refuse dials for a minute
after one busy call.
"""

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from redis.exceptions import RedisError

from app.core.queue import get_arq_pool

_KEY = "llm:budget"
# Long enough to survive a gap between calls, short enough that a reading nobody has
# refreshed stops being used. Past this the dialer treats the budget as unknown and allows
# the dial, which is the right default: we only ever had this information by accident.
_TTL = 300

_REDIS_FAULTS = (RedisError, RuntimeError, OSError, AttributeError)


@dataclass(frozen=True)
class Headroom:
    """Estimated room left this minute. None on a field means the provider does not report it."""

    tokens: Optional[float]
    requests: Optional[float]


async def record_budget(
    tokens: int,
    token_limit: int,
    requests: Optional[int] = None,
    request_limit: Optional[int] = None,
) -> None:
    """Publish a rate-limit reading. Never raises — this runs inside a live call."""
    payload = {"tokens": tokens, "token_limit": token_limit, "at": time.time()}
    if requests is not None and request_limit:
        payload["requests"] = requests
        payload["request_limit"] = request_limit
    try:
        redis = get_arq_pool()
        await redis.hset(_KEY, mapping=payload)
        await redis.expire(_KEY, _TTL)
    except _REDIS_FAULTS as exc:
        logger.debug(f"Could not publish the LLM budget ({exc}); dialing will not gate on it")


def _refill(remaining: Optional[float], limit: Optional[float], elapsed: float) -> Optional[float]:
    """Carry a reading forward at the provider's own per-minute refill rate."""
    if remaining is None or not limit:
        return None
    return min(limit, remaining + max(0.0, elapsed) * (limit / 60.0))


async def headroom() -> Headroom:
    """Best estimate of what is left this minute.

    None is not zero. No reading means no basis to refuse, and refusing to dial on missing
    telemetry would take the whole campaign down the first time Redis blinked.
    """
    try:
        raw = await get_arq_pool().hgetall(_KEY)
    except _REDIS_FAULTS as exc:
        logger.debug(f"LLM budget unreadable ({exc}); dialing without it")
        return Headroom(None, None)
    if not raw:
        return Headroom(None, None)

    def _get(key: str) -> Optional[float]:
        value = raw.get(key) if key in raw else raw.get(key.encode())
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    at = _get("at")
    if at is None:
        return Headroom(None, None)
    elapsed = time.time() - at
    return Headroom(
        tokens=_refill(_get("tokens"), _get("token_limit"), elapsed),
        requests=_refill(_get("requests"), _get("request_limit"), elapsed),
    )


async def tokens_available() -> Optional[float]:
    """Kept as its own name because that is the question the dial gate asks first."""
    return (await headroom()).tokens
