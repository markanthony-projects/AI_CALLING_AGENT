"""The LLM service, given a live phone call's tolerance for waiting.

A call spent fourteen seconds in silence and the log said `groq=14605ms` with no error.
Nothing was slow. Groq's per-minute token ceiling had been reached, it answered 429 with
"please try again in Ns", and the OpenAI SDK — which defaults to ``max_retries=2`` and
honours Retry-After inside the same ``await`` — simply slept and tried again. Pipecat
starts its TTFB timer before that await and stops it on the first token, so the whole
sleep was billed to the model. A throttle and a slow model are indistinguishable in the
logs, which is why this went unnoticed across several calls.

Three things follow from that, and this module is all three:

  ``max_retries=0``     Silence is the worst thing a phone call can do. A retry that takes
                        longer than a caller will wait is not a recovery.
  a budget watcher      The rate-limit headers were on every single response the whole
                        time. Reading them turns an invisible failure into a warning.
  a fallback endpoint   Every provider here speaks the OpenAI wire format, so a 429 can be
                        answered by asking somebody else instead of hanging up.
"""

import re
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, RateLimitError
from pipecat.services.groq.llm import GroqLLMService

from app.core.llm_budget import record_budget

# Groq words a per-minute throttle and an exhausted daily allowance almost identically:
#
#   ...on tokens per minute (TPM): Limit 12000, Used 11500. Please try again in 3.5s.
#   ...on tokens per day (TPD):    Limit 100000, Used 99884. Please try again in 28m37.6s.
#
# Both trip is_quota_error(). The only thing separating a hiccup from a dead account is the
# number of seconds, so that is what gets parsed. Reading them as the same thing means
# hanging up on a caller over a three-second wait.
# Milliseconds are matched before minutes would swallow them: without the lookahead "500ms"
# parses as 500 minutes, and without the ms group at all it parses as nothing — which lands
# in is_quota_error() and hangs up the call over half a second.
_TRY_AGAIN = re.compile(
    r"try\s+again\s+in\s*"
    r"(?:(?P<h>\d+(?:\.\d+)?)\s*h)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)\s*m(?!s))?"
    r"(?:(?P<ms>\d+(?:\.\d+)?)\s*ms)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s)?",
    re.I,
)

# Longer than this and there is no point waiting: the caller is listening to nothing, and
# an apology now beats an answer they have already hung up on. Deliberately shorter than
# any plausible reply — a turn that takes 8s has already failed by conversational standards.
MAX_THROTTLE_WAIT_SECS = 8.0


def retry_after_seconds(message: Optional[str]) -> Optional[float]:
    """How long the provider asked us to wait, in seconds. None when it did not say."""
    if not message:
        return None
    match = _TRY_AGAIN.search(str(message))
    if not match:
        return None
    parts = {k: float(v) for k, v in match.groupdict().items() if v is not None}
    if not parts:
        return None
    return (
        parts.get("h", 0.0) * 3600
        + parts.get("m", 0.0) * 60
        + parts.get("s", 0.0)
        + parts.get("ms", 0.0) / 1000
    )


def is_transient_throttle(message: Optional[str]) -> bool:
    """True for a rate limit that clears on its own within a turn.

    Distinguishes "you are going too fast this minute" from "your account is out of
    tokens for the day". Both arrive as 429 with near-identical wording; only the first
    is worth staying on the line for.
    """
    delay = retry_after_seconds(message)
    return delay is not None and delay <= MAX_THROTTLE_WAIT_SECS


@dataclass(frozen=True)
class LLMEndpoint:
    """One OpenAI-wire-format provider the agent can talk to."""

    name: str
    api_key: str
    base_url: str
    model: str


class BudgetWatcher:
    """Reads the rate-limit headers that were on every response all along.

    Every Groq response carries x-ratelimit-remaining-tokens. Nothing looked at it, so the
    account crossed its ceiling mid-call with no warning anywhere — the first symptom was a
    caller saying "Hello?" into silence. This logs the approach rather than the arrival.
    """

    def __init__(self, call_sid: str, warn_below: int):
        self._call_sid = call_sid
        self._warn_below = warn_below
        self._warned = False

    async def __call__(self, response: httpx.Response) -> None:
        # Headers only. Touching the body here would consume the stream the pipeline is
        # about to read.
        remaining = response.headers.get("x-ratelimit-remaining-tokens")
        if remaining is None:
            return
        try:
            left = int(float(remaining))
        except ValueError:
            return

        limit = response.headers.get("x-ratelimit-limit-tokens")
        if limit:
            # Shared so the dialer can decline to start a call this account cannot pay for.
            # Best-effort by design: it must never interfere with the call in progress.
            try:
                await record_budget(left, int(float(limit)))
            except Exception:  # noqa: BLE001 - telemetry must not break a live call
                pass

        if left >= self._warn_below:
            self._warned = False
            return
        if self._warned:
            return
        self._warned = True
        logger.warning(
            f"[{self._call_sid}] LLM token budget low: {left} left of "
            f"{response.headers.get('x-ratelimit-limit-tokens', '?')} per minute, "
            f"resets in {response.headers.get('x-ratelimit-reset-tokens', '?')}. "
            f"The next turn will be throttled and the caller will hear the wait."
        )


class ResilientLLMService(GroqLLMService):
    """Groq, but it never leaves the caller listening to silence.

    The SDK's own retry is disabled on purpose. It is built for batch work, where waiting
    out a Retry-After is exactly right; on a phone call the same behaviour spends the
    caller's patience invisibly and reports the delay as model latency.
    """

    # Pipecat names a processor after its class, and that name is what the latency
    # observer prints. Left alone every log line would read `resilientllm=376ms`, breaking
    # continuity with every call log recorded so far for no gain.
    LOG_NAME = "GroqLLMService"

    def __init__(self, *, call_sid: str = "-", fallback: Optional[LLMEndpoint] = None, **kwargs):
        # Set before super().__init__, which calls create_client() on its last line.
        self._call_sid = call_sid
        kwargs.setdefault("name", self.LOG_NAME)
        self._fallback = fallback
        self._fallback_client: Optional[AsyncOpenAI] = None
        # One request's worth of headroom. Below this the next turn is the one that stalls.
        self._watcher = BudgetWatcher(call_sid, warn_below=4000)
        super().__init__(**kwargs)
        if fallback:
            self._fallback_client = AsyncOpenAI(
                api_key=fallback.api_key, base_url=fallback.base_url, max_retries=0
            )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            # The whole point. Two silent retries honouring Retry-After is how a 429 turned
            # into 14.6 seconds of dead air that the logs attributed to the model.
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                ),
                event_hooks={"response": [self._watcher]},
            ),
        )

    async def get_chat_completions(self, context):
        """Ask Groq; on a rate limit, ask the fallback rather than give up the turn."""
        try:
            return await super().get_chat_completions(context)
        except RateLimitError as exc:
            if self._fallback_client is None or self._fallback is None:
                raise
            waited = retry_after_seconds(str(exc))
            logger.warning(
                f"[{self._call_sid}] Groq rate-limited"
                f"{f' (asked for {waited:.1f}s)' if waited else ''}; "
                f"switching this turn to {self._fallback.name}/{self._fallback.model}"
            )
            return await self._complete_on_fallback(context)

    async def _complete_on_fallback(self, context):
        """Same request, different provider.

        Built through the service's own adapter rather than by hand so tools, tool_choice
        and message conversion stay identical to the primary path — a fallback that quietly
        drops the end_call tool would strand every call it rescued.
        """
        adapter = self.get_llm_adapter()
        params = self.build_chat_completion_params(
            adapter.get_llm_invocation_params(
                context, convert_developer_to_user=not self.supports_developer_role
            )
        )
        params["model"] = self._fallback.model
        return await self._fallback_client.chat.completions.create(**params)

    async def stop(self, frame):
        await super().stop(frame)
        if self._fallback_client is not None:
            await self._fallback_client.close()


def build_llm_service(call_sid: str, model: str, settings) -> ResilientLLMService:
    """Assemble the call's LLM service from configuration.

    The fallback stays off unless both its key and its model are set. A half-configured
    fallback is worse than none: it would look like insurance and fail at the one moment
    it was needed.
    """
    fallback = None
    if settings.llm_fallback_enabled:
        fallback = LLMEndpoint(
            name=settings.LLM_FALLBACK_NAME,
            api_key=settings.LLM_FALLBACK_API_KEY,
            base_url=settings.LLM_FALLBACK_BASE_URL,
            model=settings.LLM_FALLBACK_MODEL,
        )

    return ResilientLLMService(
        call_sid=call_sid,
        fallback=fallback,
        api_key=settings.GROQ_API_KEY,
        settings=GroqLLMService.Settings(model=model),
    )
