"""The LLM service, given a live phone call's tolerance for waiting.

A call spent fourteen seconds in silence and the log said `groq=14605ms` with no error.
Nothing was slow. Groq's per-minute token ceiling had been reached, it answered 429 with
"please try again in Ns", and the OpenAI SDK — which defaults to ``max_retries=2`` and
honours Retry-After inside the same ``await`` — simply slept and tried again. Pipecat
starts its TTFB timer before that await and stops it on the first token, so the whole
sleep was billed to the model. A throttle and a slow model are indistinguishable in the
logs, which is why this went unnoticed across several calls.

Four things follow, and this module is all four:

  ``max_retries=0``     Silence is the worst thing a phone call can do. A retry that takes
                        longer than a caller will wait is not a recovery.
  a budget watcher      The rate-limit headers were on every response the whole time.
                        Reading them turns an invisible failure into a warning, and lets
                        the dialer decline a call the account cannot pay for.
  a fallback endpoint   Every provider here speaks the OpenAI wire format, so a 429 can be
                        answered by asking somebody else instead of hanging up.
  no provider in code   Groq deprecated the model this agent was built on with six weeks'
                        notice. Which provider is serving traffic is configuration, not a
                        class name, so the next deprecation is an env change.

Providers agree on all the concepts here and on none of the spellings. Groq says
``x-ratelimit-remaining-tokens``; Cerebras says ``x-ratelimit-remaining-tokens-minute``.
Groq puts the wait in English prose in the body; Cerebras sends a ``Retry-After`` header
and no prose at all. Both were measured, and both are handled here so that nothing
downstream has to know which one it is talking to.
"""

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import httpx
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, RateLimitError
from pipecat.services.openai.llm import OpenAILLMService

from app.core.llm_budget import record_budget

# --- how long the provider wants us to wait ------------------------------------------
#
# Groq words a per-minute throttle and an exhausted daily allowance almost identically:
#
#   ...on tokens per minute (TPM): Limit 12000, Used 11500. Please try again in 3.5s.
#   ...on tokens per day (TPD):    Limit 100000, Used 99884. Please try again in 28m37.6s.
#
# Both trip is_quota_error(). The only thing separating a hiccup from a dead account is the
# number of seconds. Cerebras says neither — "Tokens per minute limit exceeded - too many
# tokens processed." — and puts the number in a Retry-After header instead. So the header is
# preferred where it exists and the prose is parsed where it does not.

# Milliseconds are matched before minutes would swallow them: without the lookahead "500ms"
# parses as 500 minutes, and without the ms group at all it parses as nothing.
_TRY_AGAIN = re.compile(
    r"try\s+again\s+in\s*"
    r"(?:(?P<h>\d+(?:\.\d+)?)\s*h)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)\s*m(?!s))?"
    r"(?:(?P<ms>\d+(?:\.\d+)?)\s*ms)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s)?",
    re.I,
)

# Longer than this and there is no point waiting: the caller is listening to nothing, and an
# apology now beats an answer they have already hung up on. Deliberately shorter than any
# plausible reply — a turn that takes 8s has already failed by conversational standards.
MAX_THROTTLE_WAIT_SECS = 8.0


def retry_after_seconds(message: Optional[str]) -> Optional[float]:
    """How long the provider asked us to wait, read out of its prose. None if it did not say."""
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


def retry_after_header(headers) -> Optional[float]:
    """The standard Retry-After header, in seconds. The HTTP-date form is not used by any
    provider here and is deliberately not guessed at."""
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def throttle_delay(message: Optional[str] = None, headers=None) -> Optional[float]:
    """The wait the provider is asking for, from whichever channel it used.

    Header first: it is the HTTP standard, it is unambiguous, and Cerebras sends nothing
    else. Prose second, because Groq sends no header.
    """
    from_header = retry_after_header(headers)
    return from_header if from_header is not None else retry_after_seconds(message)


def is_transient_throttle(message: Optional[str], delay: Optional[float] = None) -> bool:
    """True for a rate limit that clears within a turn, so the call is worth keeping.

    `delay` is the authoritative value when the caller has one (read from Retry-After);
    otherwise the provider's prose is parsed. Distinguishes "you are going too fast this
    minute" from "your account is finished", which arrive looking almost identical.
    """
    seconds = delay if delay is not None else retry_after_seconds(message)
    return seconds is not None and seconds <= MAX_THROTTLE_WAIT_SECS


# --- rate-limit headers ---------------------------------------------------------------
#
# Both providers report tokens per minute; only the spelling differs.
_REMAINING_TOKENS: Sequence[str] = (
    "x-ratelimit-remaining-tokens",  # Groq, OpenAI
    "x-ratelimit-remaining-tokens-minute",  # Cerebras
)
_LIMIT_TOKENS: Sequence[str] = (
    "x-ratelimit-limit-tokens",
    "x-ratelimit-limit-tokens-minute",
)
# Requests are NOT symmetrical, and reading them wrong would be worse than not reading them.
# Groq's unsuffixed x-ratelimit-limit-requests is a DAILY figure (1000 on the free tier);
# Cerebras's -minute suffix means what it says (5). Only the explicit per-minute spelling is
# used, so Groq simply reports no RPM and the request check does not run for it.
_REMAINING_REQUESTS: Sequence[str] = ("x-ratelimit-remaining-requests-minute",)
_LIMIT_REQUESTS: Sequence[str] = ("x-ratelimit-limit-requests-minute",)


def _first_number(headers, names: Sequence[str]) -> Optional[int]:
    for name in names:
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class LLMEndpoint:
    """One OpenAI-wire-format provider the agent can talk to."""

    name: str
    api_key: str
    base_url: str
    model: str

    def __str__(self) -> str:
        return f"{self.name}/{self.model}"


class BudgetWatcher:
    """Reads the rate-limit headers that were on every response all along.

    Nothing looked at them, so the account crossed its ceiling mid-call with no warning
    anywhere — the first symptom was a caller saying "Hello?" into silence. This logs the
    approach rather than the arrival, and publishes the reading so the dialer can decline
    to start a call that would stall on its own greeting.
    """

    def __init__(self, call_sid: str, warn_below: int):
        self._call_sid = call_sid
        self._warn_below = warn_below
        self._warned = False

    async def __call__(self, response: httpx.Response) -> None:
        # Headers only. Touching the body here would consume the stream the pipeline is
        # about to read.
        headers = response.headers
        tokens = _first_number(headers, _REMAINING_TOKENS)
        if tokens is None:
            return

        limit = _first_number(headers, _LIMIT_TOKENS)
        requests_left = _first_number(headers, _REMAINING_REQUESTS)
        requests_limit = _first_number(headers, _LIMIT_REQUESTS)

        if limit:
            # Best-effort by design: telemetry must never interfere with a live call.
            try:
                await record_budget(tokens, limit, requests_left, requests_limit)
            except Exception:  # noqa: BLE001
                pass

        if tokens >= self._warn_below:
            self._warned = False
            return
        if self._warned:
            return
        self._warned = True
        logger.warning(
            f"[{self._call_sid}] LLM token budget low: {tokens} left of {limit or '?'} per "
            f"minute. The next turn will be throttled and the caller will hear the wait."
        )


class ResilientLLMService(OpenAILLMService):
    """Any OpenAI-compatible provider, with a phone call's tolerance for waiting.

    The SDK's own retry is disabled on purpose. It is built for batch work, where waiting
    out a Retry-After is exactly right; on a phone call the same behaviour spends the
    caller's patience invisibly and reports the delay as model latency.
    """

    # Pipecat names a processor after its class, and that name is what the latency observer
    # prints. Left alone every log line would read `resilientllm=376ms`, breaking continuity
    # with every call log recorded so far for no gain.
    LOG_NAME = "GroqLLMService"

    def __init__(
        self,
        *,
        endpoint: LLMEndpoint,
        call_sid: str = "-",
        fallback: Optional[LLMEndpoint] = None,
        warn_below: int = 4000,
        **kwargs,
    ):
        # Set before super().__init__, which calls create_client() on its last line.
        self._call_sid = call_sid
        self._endpoint = endpoint
        self._fallback = fallback
        self._fallback_client: Optional[AsyncOpenAI] = None
        self._last_throttle_delay: Optional[float] = None
        self._watcher = BudgetWatcher(call_sid, warn_below=warn_below)
        kwargs.setdefault("name", self.LOG_NAME)
        super().__init__(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            settings=OpenAILLMService.Settings(model=endpoint.model),
            **kwargs,
        )
        if fallback:
            self._fallback_client = AsyncOpenAI(
                api_key=fallback.api_key, base_url=fallback.base_url, max_retries=0
            )

    @property
    def endpoint(self) -> LLMEndpoint:
        return self._endpoint

    @property
    def last_throttle_delay(self) -> Optional[float]:
        """Seconds the provider last asked us to wait, taken from its Retry-After header.

        Exists because the only thing the pipeline hands an error handler is a string, and
        Cerebras puts the number in a header rather than in the message. Without this the
        agent would have to guess whether a 429 was a three-second hiccup or a dead account,
        and guessing wrong means either hanging up on a live caller or making them repeat
        themselves into an account that cannot answer.
        """
        return self._last_throttle_delay

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
        """Ask the primary; on a rate limit, ask the fallback rather than lose the turn."""
        try:
            completions = await super().get_chat_completions(context)
        except RateLimitError as exc:
            self._last_throttle_delay = throttle_delay(
                str(exc), getattr(getattr(exc, "response", None), "headers", None)
            )
            if self._fallback_client is None or self._fallback is None:
                raise
            waited = self._last_throttle_delay
            logger.warning(
                f"[{self._call_sid}] {self._endpoint} rate-limited"
                f"{f' (asked for {waited:.1f}s)' if waited is not None else ''}; "
                f"switching this turn to {self._fallback}"
            )
            return await self._complete_on_fallback(context)
        else:
            # Cleared on success, so a delay from an earlier turn cannot be read as the
            # explanation for a later, unrelated failure.
            self._last_throttle_delay = None
            return completions

    async def _complete_on_fallback(self, context):
        """The same request, sent to a different provider.

        Built through the service's own adapter rather than by hand so tools, tool_choice
        and message conversion stay identical to the primary path — a fallback that quietly
        dropped the end_call tool would rescue the turn and then strand the call.
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


def primary_endpoint(settings) -> LLMEndpoint:
    """Where the agent's turns go.

    The key is resolved per provider by Settings.llm_api_key, so pointing LLM_BASE_URL at a
    different vendor cannot silently send it the previous vendor's credential.
    """
    return LLMEndpoint(
        name=settings.LLM_PROVIDER_NAME,
        api_key=settings.llm_api_key,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
    )


def fallback_endpoint(settings) -> Optional[LLMEndpoint]:
    """The second provider, or None.

    Off unless both its key and its model are set. A half-configured fallback is worse than
    none: it looks like insurance and fails at the one moment it is needed.
    """
    if not settings.llm_fallback_enabled:
        return None
    return LLMEndpoint(
        name=settings.LLM_FALLBACK_NAME,
        api_key=settings.LLM_FALLBACK_API_KEY,
        base_url=settings.LLM_FALLBACK_BASE_URL,
        model=settings.LLM_FALLBACK_MODEL,
    )


def build_llm_service(call_sid: str, settings) -> ResilientLLMService:
    """Assemble the call's LLM service from configuration alone."""
    return ResilientLLMService(
        call_sid=call_sid,
        endpoint=primary_endpoint(settings),
        fallback=fallback_endpoint(settings),
        warn_below=settings.LLM_MIN_TOKENS_TO_DIAL,
    )
