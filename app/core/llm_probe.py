"""Can the configured models actually answer? Asked with a completion, not a listing.

On 10 Sep 2026 Cerebras returned 404 model_not_found for gemma-4-31b on every completion
while models.list() still returned it. Every production call failed, and the first sign was
a prospect being told "Sorry, I missed that" twice and hung up on. Four days later the
local .env still named that model — and named a fallback Groq had shut down on 16 August.
Two dead models, and nothing in the process that could have said so before a real person
answered the phone.

So this asks each configured endpoint for one token, at startup and on a timer, and
publishes the verdict where the dialer can read it. "Listed" is not "available"; a
completion is the only test that means anything, and it costs a fraction of a paisa.

The verdict feeds one decision: may a dial be placed? A call needs SOME model to answer
its turns. If the primary is gone and the fallback can carry the call, dialing continues
and the log says so; if neither can, every dial would bill the carrier leg, ring a real
person, and hang up on them — so the pump stops until a probe passes. Unknown is not a
refusal: a probe that has not run, or a Redis that cannot be read, must not take the
campaign down. That is the same stance app/core/llm_budget.py takes, for the same reason.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)
from redis.exceptions import RedisError

from app.core.queue import get_arq_pool

if TYPE_CHECKING:
    from app.services.llm_provider import LLMEndpoint

_KEY = "llm:probe"
# A verdict older than this is stale — the periodic probe has stopped, or Redis restarted —
# and stale is treated as unknown, which permits dialing. Sized to two missed intervals.
_TTL = 900

_REDIS_FAULTS = (RedisError, RuntimeError, OSError, AttributeError)

# One token is enough to prove the model exists and takes our request shape. Ten seconds
# is far longer than any healthy first token and short enough that a dead provider does
# not hold startup for a minute.
_PROBE_TIMEOUT_SECS = 10.0

OK = "ok"
MODEL_NOT_FOUND = "model_not_found"
BAD_REQUEST = "bad_request"
UNAUTHORIZED = "unauthorized"
RATE_LIMITED = "rate_limited"
UNREACHABLE = "unreachable"

# Verdicts on which a call can still be served. A throttled endpoint is a working one that
# is busy; the per-turn fallover already handles that on a live call.
_SERVICEABLE = frozenset({OK, RATE_LIMITED})


@dataclass(frozen=True)
class Verdict:
    """What one endpoint said when asked for a token."""

    role: str  # "primary" or "fallback"
    endpoint: str  # "cerebras/gpt-oss-120b"
    status: str
    detail: str = ""

    @property
    def serviceable(self) -> bool:
        return self.status in _SERVICEABLE


def _client(endpoint: "LLMEndpoint") -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        max_retries=0,
        timeout=_PROBE_TIMEOUT_SECS,
    )


async def probe_endpoint(endpoint: "LLMEndpoint", role: str) -> Verdict:
    """Ask for one token, with exactly the extra parameters a real turn would carry.

    The extra parameters matter: a reasoning_effort this model does not take is a 400 on
    every turn, and a probe that omitted it would pass a configuration that fails live.
    """
    client = _client(endpoint)
    try:
        await client.chat.completions.create(
            model=endpoint.model,
            messages=[{"role": "user", "content": "."}],
            max_tokens=1,
            **endpoint.extra_params,
        )
        return Verdict(role, str(endpoint), OK)
    except NotFoundError as e:
        return Verdict(role, str(endpoint), MODEL_NOT_FOUND, str(e))
    except AuthenticationError as e:
        return Verdict(role, str(endpoint), UNAUTHORIZED, str(e))
    except BadRequestError as e:
        return Verdict(role, str(endpoint), BAD_REQUEST, str(e))
    except RateLimitError as e:
        return Verdict(role, str(endpoint), RATE_LIMITED, str(e))
    except (APIConnectionError, APITimeoutError) as e:
        return Verdict(role, str(endpoint), UNREACHABLE, str(e))
    except Exception as e:  # noqa: BLE001 — a probe must classify, never raise
        return Verdict(role, str(endpoint), UNREACHABLE, f"{type(e).__name__}: {e}")
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass


async def probe_all(settings) -> List[Verdict]:
    """The primary, and the fallback when one is configured."""
    # Imported here, not at module scope: llm_provider pulls in the Pipecat runtime, and
    # the dial pump — which reads this module's verdict — runs in the worker, which is kept
    # free of that runtime on purpose. See the note in app/services/dialer.py.
    from app.services.llm_provider import fallback_endpoint, primary_endpoint

    verdicts = [await probe_endpoint(primary_endpoint(settings), "primary")]
    fallback = fallback_endpoint(settings)
    if fallback is not None:
        verdicts.append(await probe_endpoint(fallback, "fallback"))
    return verdicts


def can_serve(verdicts: List[Verdict]) -> bool:
    """True when at least one endpoint can answer a turn."""
    return any(v.serviceable for v in verdicts)


def report(verdicts: List[Verdict]) -> None:
    """Say what was found, at the level the finding deserves.

    A dead primary with a live fallback is an ERROR that dialing survives. Nothing live is
    an ERROR that stops it, and the line says so in words, because the next person to read
    the log is the one wondering why the queue is not moving.
    """
    for v in verdicts:
        if v.status == OK:
            logger.info(f"LLM probe: {v.role} {v.endpoint} answers")
        else:
            logger.error(
                f"LLM probe: {v.role} {v.endpoint} cannot answer ({v.status})"
                + (f": {v.detail[:200]}" if v.detail else "")
            )
    if not can_serve(verdicts):
        logger.error(
            "LLM probe: no configured model can answer a call. Dialing is stopped until one "
            "does — fix LLM_MODEL / LLM_FALLBACK_MODEL in .env and restart."
        )
    elif verdicts and not verdicts[0].serviceable:
        logger.error(
            f"LLM probe: every call will run on the fallback ({verdicts[-1].endpoint}) "
            f"until the primary is fixed."
        )


async def publish(verdicts: List[Verdict]) -> None:
    """Record the verdict for the dialer. Never raises."""
    payload = {
        "serviceable": "1" if can_serve(verdicts) else "0",
        "at": time.time(),
    }
    for v in verdicts:
        payload[v.role] = v.status
    try:
        redis = get_arq_pool()
        await redis.hset(_KEY, mapping=payload)
        await redis.expire(_KEY, _TTL)
    except _REDIS_FAULTS as exc:
        logger.debug(f"Could not publish the LLM probe ({exc}); dialing will not gate on it")


async def run_and_publish(settings) -> List[Verdict]:
    """One full cycle: probe, log, publish. What startup and the timer both call."""
    verdicts = await probe_all(settings)
    report(verdicts)
    await publish(verdicts)
    return verdicts


async def status() -> dict:
    """The last published verdict, for /health. Empty when nothing has been published."""
    try:
        raw = await get_arq_pool().hgetall(_KEY)
    except _REDIS_FAULTS:
        return {}
    if not raw:
        return {}
    decoded = {}
    for key, value in raw.items():
        k = key.decode() if isinstance(key, bytes) else key
        decoded[k] = value.decode() if isinstance(value, bytes) else value
    return decoded


async def serviceable() -> bool:
    """May a dial be placed, as far as the LLM is concerned?

    Unknown is yes. No probe yet, a Redis that cannot be read, or a verdict past its TTL
    all mean "no basis to refuse" — refusing on missing telemetry would take the campaign
    down every time Redis blinked, and the per-call fallover still stands behind this.
    """
    verdict = await status()
    if not verdict:
        return True
    return verdict.get("serviceable", "1") != "0"


def summarise(verdicts: Optional[List[Verdict]]) -> str:
    """One line for a startup log or a test assertion."""
    if not verdicts:
        return "no probe"
    return ", ".join(f"{v.role}={v.status}" for v in verdicts)
