"""Rate limits, and why a call spent fourteen seconds in silence.

From a live call, five consecutive turns:

    LATENCY turn 5:  2792ms  |  groq=2579ms   sarvam=206ms
    LATENCY turn 6:  5793ms  |  groq=5500ms   sarvam=285ms
    LATENCY turn 7:  7891ms  |  groq=7650ms   sarvam=228ms
    LATENCY turn 8: 13776ms  |  groq=13506ms  sarvam=248ms
    LATENCY turn 9: 14861ms  |  groq=14605ms  sarvam=249ms

No errors. Sarvam flat throughout. Turn 4 had run at groq=376ms, so nothing was slow — the
account's per-minute token ceiling had been reached (12,000/min measured against a request
that cost 4,121 tokens, i.e. under three turns a minute). Groq answered 429 with
"try again in Ns" and the OpenAI SDK, which defaults to max_retries=2 and honours
Retry-After inside the same await, slept and retried. Pipecat's TTFB timer starts before
that await, so every second of waiting was billed to the model and no error was ever
raised. The caller said "Hello?" twice into the silence.
"""

import ast
import asyncio
import inspect

import pytest

from app.core.config import Settings
from app.services import agent
from app.services.agent import LLM_BUSY_LINE, LLM_RECOVERY_LINE, is_quota_error
from app.services.llm_provider import (
    MAX_THROTTLE_WAIT_SECS,
    BudgetWatcher,
    LLMEndpoint,
    ResilientLLMService,
    build_llm_service,
    is_transient_throttle,
    primary_endpoint,
    retry_after_header,
    retry_after_seconds,
    throttle_delay,
)

GROQ = LLMEndpoint(name="groq", api_key="k", base_url="https://api.groq.com/openai/v1", model="m")


def _service(**kw):
    return ResilientLLMService(call_sid="t", endpoint=GROQ, **kw)

# Verbatim shapes from Groq. The two differ only in the unit of the limit and the size of
# the number — which is exactly the problem.
TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_01k` service tier `on_demand` on tokens "
    "per minute (TPM): Limit 12000, Used 11500, Requested 4200. Please try again in 3.5s.', "
    "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_01k` service tier `on_demand` on tokens "
    "per day (TPD): Limit 100000, Used 98928, Requested 3060. Please try again in "
    "28m37.631999999s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)

BASE = dict(
    API_KEY="k" * 32,
    CALL_TOKEN_SECRET="s" * 32,
    DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
    OPENAI_API_KEY="x",
    SARVAM_API_KEY="x",
    # Settings now refuses to construct without a key for the configured provider, so this
    # has to be supplied rather than leaked in from the developer's own .env.
    CEREBRAS_API_KEY="csk-test",
    GROQ_API_KEY="gsk-test",
)


@pytest.fixture(autouse=True)
def _restore_environment():
    """_settings() below strips the LLM variables out of os.environ so these assert against the
    defaults rather than against whatever the machine has. It used to strip them permanently.

    Every test that ran after this file then saw a process with no CEREBRAS_API_KEY, and the
    ones that build their own Settings could not satisfy the "the LLM has a key" validator.
    Nothing noticed, because .env was still being read as a last resort and quietly put the key
    back. The moment the harness stopped reading .env — which is the whole point of a harness —
    five unrelated tests failed in a full run and passed on their own.
    """
    import os

    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _settings(**over):
    """A Settings built from arguments alone.

    load_dotenv() has already pushed .env into os.environ by the time this module is
    imported, so both the env vars and the env file have to be shut out or these assert
    against whatever is on the machine.
    """
    import os

    for key in (
        "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_MODEL", "TURN_SETTLE_SECS",
        "LLM_PROVIDER_NAME", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
        "CEREBRAS_API_KEY", "GROQ_API_KEY",
    ):
        os.environ.pop(key, None)
    return Settings(**{**BASE, **over}, _env_file=None)


# --- reading the delay --------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Please try again in 3.5s.", 3.5),
        ("Please try again in 28m37.631999999s.", 28 * 60 + 37.631999999),
        ("please try again in 1h2m3s", 3723.0),
        ("Please try again in 500ms", 0.5),
        ("Please try again in 45s", 45.0),
    ],
)
def test_the_provider_delay_is_read_out_of_the_message(message, expected):
    assert retry_after_seconds(message) == pytest.approx(expected)


def test_milliseconds_are_not_read_as_minutes():
    """Without the lookahead "500ms" parses as 500 minutes — 30,000x too long, and the call
    gets hung up on rather than waiting half a second."""
    assert retry_after_seconds("try again in 500ms") < 1.0


@pytest.mark.parametrize("message", [None, "", "Connection reset by peer", "Error code: 500"])
def test_a_message_with_no_delay_reads_as_none(message):
    assert retry_after_seconds(message) is None


# --- the distinction the whole change rests on --------------------------------------


def test_a_per_minute_throttle_is_transient():
    assert is_transient_throttle(TPM)


def test_an_exhausted_daily_allowance_is_not():
    assert not is_transient_throttle(TPD)


def test_both_look_identical_to_the_quota_check():
    """The reason is_transient_throttle has to exist at all.

    is_quota_error matches on "rate limit reached", which Groq puts in front of both. While
    the SDK swallowed 429s this never mattered; with max_retries=0 they surface, and without
    this distinction the first per-minute throttle would hang up on the caller with
    "I'm having some trouble on this line" over a three-second wait.
    """
    assert is_quota_error(TPM) and is_quota_error(TPD), "both trip the quota check"
    assert is_transient_throttle(TPM) != is_transient_throttle(TPD), "yet only one is fatal"


@pytest.mark.parametrize(
    "delay,transient",
    [(0.5, True), (MAX_THROTTLE_WAIT_SECS - 0.1, True), (MAX_THROTTLE_WAIT_SECS, True),
     (MAX_THROTTLE_WAIT_SECS + 0.1, False), (60.0, False)],
)
def test_the_boundary_is_how_long_a_caller_will_wait(delay, transient):
    assert is_transient_throttle(f"try again in {delay}s") is transient


def test_an_unparseable_rate_limit_is_treated_as_terminal():
    """Conservative on purpose: staying on a line we cannot serve is worse than a clean
    sign-off, and a provider that will not say how long has not promised it is short."""
    assert not is_transient_throttle("Error code: 429 - rate_limit_exceeded")


# --- the client ---------------------------------------------------------------------


def test_the_sdk_does_not_retry_behind_our_back():
    """The single most important line in the module. Two silent retries honouring
    Retry-After is precisely how a 429 reached the logs as groq=14605ms."""
    assert _service()._client.max_retries == 0


def test_the_processor_still_logs_itself_as_groq():
    """LatencyObserver prints the processor name. A subclass would silently rename every
    latency line to `resilientllm=` and break continuity with every call log so far."""
    from app.utils.latency import _short

    assert _short(_service().name + "#0") == "groq"


# --- the budget watcher -------------------------------------------------------------


class _Response:
    def __init__(self, remaining):
        self.headers = {
            "x-ratelimit-remaining-tokens": str(remaining),
            "x-ratelimit-limit-tokens": "12000",
            "x-ratelimit-reset-tokens": "4.2s",
        }


def _warnings_from(watcher, *remainings):
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(m), level="WARNING")
    try:
        for r in remainings:
            asyncio.run(watcher(_Response(r)))
    finally:
        logger.remove(sink)
    return seen


def test_the_approaching_ceiling_is_reported():
    """These headers were on every response for the whole call and nothing read them, so
    the account crossed its limit mid-conversation with no warning anywhere."""
    assert len(_warnings_from(BudgetWatcher("sid", warn_below=4000), 500)) == 1


def test_a_healthy_budget_says_nothing():
    assert _warnings_from(BudgetWatcher("sid", warn_below=4000), 11000) == []


def test_it_does_not_repeat_itself_every_turn():
    """One warning per excursion. Once per request would bury the call's own log lines."""
    assert len(_warnings_from(BudgetWatcher("sid", warn_below=4000), 500, 400, 300)) == 1


def test_it_re_arms_after_the_budget_recovers():
    """The bucket refills between turns, so a second excursion is a second real event."""
    assert len(_warnings_from(BudgetWatcher("sid", warn_below=4000), 500, 9000, 400)) == 2


def test_a_response_without_the_headers_is_ignored():
    class Bare:
        headers = {}

    assert _warnings_from(BudgetWatcher("sid", warn_below=4000), *[]) == []
    asyncio.run(BudgetWatcher("sid", warn_below=4000)(Bare()))  # must not raise


# --- the fallback -------------------------------------------------------------------


def test_the_fallback_is_off_until_it_is_fully_configured():
    """Half-configured insurance fails at the one moment it is needed."""
    assert build_llm_service("sid", _settings())._fallback is None
    assert build_llm_service("sid", _settings(LLM_FALLBACK_API_KEY="k"))._fallback is None
    assert build_llm_service("sid", _settings(LLM_FALLBACK_MODEL="gpt-4o-mini"))._fallback is None


def test_a_fully_configured_fallback_is_wired():
    svc = build_llm_service(
        "sid", _settings(LLM_FALLBACK_API_KEY="k", LLM_FALLBACK_MODEL="gpt-4o-mini")
    )
    assert isinstance(svc._fallback, LLMEndpoint)
    assert svc._fallback.model == "gpt-4o-mini"
    assert svc._fallback_client is not None
    assert svc._fallback_client.max_retries == 0, "the fallback must not stall either"


def test_the_fallback_carries_the_tools_over():
    """A fallback that drops end_call would rescue the turn and then strand the call with
    no way to hang up. Built through the service's own adapter for exactly this reason."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    async def end_call(params: dict, closing_line: str):
        """Hangs up."""

    svc = build_llm_service(
        "sid", _settings(LLM_FALLBACK_API_KEY="k", LLM_FALLBACK_MODEL="gpt-4o-mini")
    )
    sent = {}

    class _Completions:
        async def create(self, **params):
            sent.update(params)

    svc._fallback_client.chat.completions = _Completions()
    ctx = LLMContext(messages=[{"role": "system", "content": "x"}], tools=[end_call])
    asyncio.run(svc._complete_on_fallback(ctx))

    assert sent["model"] == "gpt-4o-mini", "the fallback's own model, not Groq's"
    assert [t["function"]["name"] for t in sent["tools"]] == ["end_call"]
    assert sent["messages"][0]["content"] == "x"


# --- how the agent reacts -----------------------------------------------------------


def _handler_src(name: str) -> str:
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == name
    )
    return ast.unparse(node)


def test_a_throttle_does_not_ask_the_caller_to_repeat_themselves():
    """They said nothing wrong. LLM_RECOVERY_LINE blames them for our rate limit."""
    src = _handler_src("on_llm_error")
    branch = src[src.index("is_transient_throttle"):]
    head = branch.split("is_quota_error")[0]
    assert "LLM_BUSY_LINE" in head
    assert "LLM_RECOVERY_LINE" not in head


def test_the_throttle_branch_is_checked_before_the_quota_branch():
    """The other way round and every per-minute throttle ends the call, because
    is_quota_error matches the throttle message too."""
    src = _handler_src("on_llm_error")
    assert src.index("is_transient_throttle") < src.index("is_quota_error")


def test_the_busy_line_is_its_own_line():
    assert LLM_BUSY_LINE != LLM_RECOVERY_LINE
    assert LLM_BUSY_LINE.strip()


def test_the_agent_builds_the_resilient_service_not_a_bare_groq_one():
    """Constructing GroqLLMService directly restores the SDK default of two silent retries,
    which is the whole bug."""
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    built = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "build_llm_service" in built
    assert "GroqLLMService" not in built

    # Checked against the parsed module too: leaving the import in place is the one step
    # between this passing and someone reinstating the bare service.
    module = ast.parse(inspect.getsource(agent))
    imported = {
        alias.name
        for n in ast.walk(module)
        if isinstance(n, ast.ImportFrom)
        for alias in n.names
    }
    assert "GroqLLMService" not in imported


# --- turn fragmentation -------------------------------------------------------------


def test_the_turn_stop_strategy_is_named_rather_than_defaulted():
    """Pipecat's default is a Smart Turn ONNX model. On PSTN it ruled "Maybe around in 2" a
    finished turn, so "months." arrived as a second turn and each half cost a full request."""
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "UserTurnStrategies"
    )
    stop = next(kw.value for kw in call.keywords if kw.arg == "stop")
    assert isinstance(stop, ast.List) and len(stop.elts) == 1


def test_the_settle_window_is_configurable():
    src = inspect.getsource(agent.run_voice_agent)
    start = src.index("SpeechTimeoutUserTurnStopStrategy(")
    assert "settings.TURN_SETTLE_SECS" in src[start : src.index(")", start)]


def test_the_settle_window_is_short_enough_to_be_worth_paying_on_every_turn():
    """It used to need 0.75s to cover the two observed splits (~50ms and ~739ms apart),
    because a split turn put two replies on the line. TurnFinalityGate holds the stale half
    back now, so a split costs one extra inference and the caller hears nothing wrong — and
    this window went back to being pure latency. It is silence on every single turn, and it
    is invisible in the LATENCY log lines, which start counting after it has elapsed.
    """
    assert _settings().TURN_SETTLE_SECS <= 0.5


def test_the_window_is_still_long_enough_for_a_breath():
    """Zero would put us back to VAD_STOP_SECS behaviour, splitting on every pause and
    paying for an inference on each half."""
    assert _settings().TURN_SETTLE_SECS >= 0.3


def test_lowering_it_is_only_safe_because_the_gate_exists():
    """Names the dependency, so removing the gate cannot quietly leave a 0.4s window that
    was only ever justified by it.

    Checked by resolving the real class and confirming the agent constructs it, not by
    looking for the name in the source: a stub assignment keeps the string and drops the
    behaviour, which is exactly the regression this is here to catch.
    """
    from app.services import agent as agent_module
    from app.utils.turn_gate import TurnFinalityGate

    assert agent_module.TurnFinalityGate is TurnFinalityGate, (
        "the agent is not using the real gate; a short settle window is unsafe without it"
    )
    tree = ast.parse(inspect.getsource(agent_module.run_voice_agent).lstrip())
    built = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "TurnFinalityGate" in built, "the gate is imported but never constructed"


def test_vad_stop_secs_was_not_used_for_this():
    """It cannot be. Pipecat waits max(0, stt_p99 - stop_secs) for transcripts, so raising
    it past Deepgram's 0.35 collapses that window and turn detection runs blind — which is
    what running at 0.6 did, and why the settle window lives on the stop strategy instead."""
    from pipecat.services.deepgram.stt import DEEPGRAM_TTFS_P99

    assert _settings().VAD_STOP_SECS < DEEPGRAM_TTFS_P99


# --- abandoning a superseded turn ---------------------------------------------------


def test_a_resumed_prospect_abandons_the_inflight_turn():
    """Both halves finishing is how a caller heard a stale "What time on Sunday?" followed
    straight by the goodbye."""
    src = _handler_src("on_user_turn_started")
    assert "_llm_in_flight" in src
    assert "InterruptionWorkerFrame" in src


def test_the_flag_is_cleared_once_the_answer_starts_arriving():
    """Otherwise an ordinary barge-in during playback would raise a second interruption on
    top of the one Pipecat already raises."""
    assert "_llm_in_flight = False" in _handler_src("on_assistant_turn_started")


def test_the_flag_is_set_when_inference_is_triggered():
    assert "_llm_in_flight = True" in _handler_src("on_user_turn_inference_triggered")


@pytest.mark.parametrize(
    "event,owner",
    [
        ("on_user_turn_inference_triggered", "user_agg"),
        ("on_assistant_turn_started", "assistant_agg"),
    ],
)
def test_the_events_relied_on_actually_exist(event, owner):
    """A misspelled event name registers a handler that is never called, and the guard
    silently does nothing for the rest of the call."""
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMAssistantAggregator,
        LLMUserAggregator,
    )

    cls = LLMUserAggregator if owner == "user_agg" else LLMAssistantAggregator
    assert f'"{event}"' in inspect.getsource(cls), f"{cls.__name__} registers no {event}"


# --- providers agree on the concepts and on none of the spellings --------------------
#
# Measured against both live APIs on 2026-08-03:
#
#   Groq      x-ratelimit-remaining-tokens: 8038
#             body: "...Please try again in 3.5s."          no Retry-After header
#   Cerebras  x-ratelimit-remaining-tokens-minute: 10538
#             x-ratelimit-limit-requests-minute: 5
#             body: "Tokens per minute limit exceeded - too many tokens processed."
#             Retry-After: 56                                no number in the body at all
#
# Reading only Groq's spellings meant the budget warning and the dial gate would have gone
# silently blind on Cerebras, and a 429 there would have fallen past both the throttle and
# the quota branch into "Sorry, I missed that. Could you say it once more?"

CEREBRAS_429 = "Error code: 429 - {'message': 'Tokens per minute limit exceeded - too many tokens processed.', 'type': 'too_many_requests_error', 'code': 'request_quota_exceeded'}"


class _Headers(dict):
    """httpx headers are case-insensitive; a plain dict is not, and the real ones arrive
    lowercased. Both spellings are exercised so neither is assumed."""


@pytest.mark.parametrize(
    "header,expected",
    [
        ({"retry-after": "56"}, 56.0),
        ({"Retry-After": "3"}, 3.0),
        ({"retry-after": "not a number"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_the_standard_retry_after_header_is_read(header, expected):
    assert retry_after_header(header) == expected


def test_the_header_wins_over_the_prose():
    """Both can be present. The header is the HTTP standard and unambiguous; the prose is a
    fallback for providers that send nothing else."""
    assert throttle_delay("try again in 3.5s", {"retry-after": "56"}) == 56.0


def test_prose_is_used_when_there_is_no_header():
    assert throttle_delay("Please try again in 3.5s.", {}) == pytest.approx(3.5)


def test_cerebras_429_carries_no_number_in_its_body():
    """Which is why the header had to be read at all — on the words alone this message
    yields nothing to classify on."""
    assert retry_after_seconds(CEREBRAS_429) is None


def test_a_long_cerebras_throttle_signs_off_rather_than_blaming_the_caller():
    """56 seconds is a dead account as far as the person on the line is concerned. Without
    the delay, none of LLM_QUOTA_MARKERS appears in this message either, so it would reach
    the generic path and ask them to repeat themselves twice before hanging up anyway."""
    assert not any(m in CEREBRAS_429.lower() for m in agent.LLM_QUOTA_MARKERS), (
        "if this message ever starts matching a marker, this test is no longer the guard"
    )
    assert not is_transient_throttle(CEREBRAS_429, 56.0)
    assert is_quota_error(CEREBRAS_429, 56.0)


def test_a_short_cerebras_throttle_keeps_the_call():
    assert is_transient_throttle(CEREBRAS_429, 2.0)
    assert not is_quota_error(CEREBRAS_429, 2.0)


@pytest.mark.parametrize(
    "headers,tokens",
    [
        ({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-limit-tokens": "12000"}, 500),
        ({"x-ratelimit-remaining-tokens-minute": "500", "x-ratelimit-limit-tokens-minute": "30000"}, 500),
    ],
)
def test_both_providers_spellings_of_the_token_headers_are_read(headers, tokens):
    """Groq on the left, Cerebras on the right. One name only would have gone blind."""
    class R:
        pass

    r = R()
    r.headers = headers
    assert len(_warnings_from_response(r)) == 1


def _warnings_from_response(response):
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        asyncio.run(BudgetWatcher("sid", warn_below=4000)(response))
    finally:
        logger.remove(sink)
    return [w for w in seen if "budget low" in w]


def test_groqs_request_header_is_not_read_as_a_per_minute_figure():
    """Groq's x-ratelimit-limit-requests is a DAILY figure (1000 on the free tier);
    Cerebras's -minute suffix means what it says (5). Treating Groq's as per-minute would
    let the dial gate believe there were a thousand requests of headroom every minute."""
    from app.services.llm_provider import _LIMIT_REQUESTS, _REMAINING_REQUESTS

    assert all(n.endswith("-minute") for n in _REMAINING_REQUESTS + _LIMIT_REQUESTS), (
        "only the explicitly per-minute spelling may be used for RPM"
    )


# --- the provider is configuration, not a class name ---------------------------------


def test_the_default_is_the_model_that_was_measured():
    """Chosen by replaying a real call against every candidate and scoring the rules live
    calls had broken — 503ms, end_call through the tool channel, the prospect's name in six
    replies out of six — not by tokens per second."""
    endpoint = primary_endpoint(_settings())
    assert "cerebras.ai" in endpoint.base_url
    assert endpoint.model == "gemma-4-31b"


def test_the_key_follows_the_provider_not_a_single_fallback():
    """The failure this prevents: a blanket `LLM_API_KEY or GROQ_API_KEY` sends the Groq
    credential to Cerebras the moment the default endpoint moves, and it arrives as a
    dropped call rather than as a configuration error."""
    assert primary_endpoint(_settings()).api_key == "csk-test"
    groq = _settings(
        LLM_PROVIDER_NAME="groq", LLM_BASE_URL="https://api.groq.com/openai/v1",
        LLM_MODEL="llama-3.3-70b-versatile",
    )
    assert primary_endpoint(groq).api_key == "gsk-test"


def test_an_explicit_key_always_wins():
    assert primary_endpoint(_settings(LLM_API_KEY="explicit")).api_key == "explicit"


def test_a_provider_with_no_key_is_refused_at_startup():
    """Without this the first symptom is a caller hearing silence: the greeting is queued,
    the completion 401s, and the recovery path apologises on our behalf. A process that
    cannot make an LLM call should not accept a websocket."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="No API key"):
        _settings(LLM_BASE_URL="https://api.cerebras.ai/v1", CEREBRAS_API_KEY="")


def test_an_unrecognised_endpoint_needs_its_key_named_explicitly():
    """A self-hosted or new vendor cannot be guessed at, and guessing would hand it
    somebody else's credential."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="No API key"):
        _settings(LLM_BASE_URL="https://llm.internal/v1")
    assert primary_endpoint(
        _settings(LLM_BASE_URL="https://llm.internal/v1", LLM_API_KEY="k")
    ).api_key == "k"


def test_switching_provider_is_configuration_only():
    """The point of the whole exercise: Groq deprecated the model this agent was built on
    with six weeks' notice, and the next such notice should cost an env change."""
    endpoint = primary_endpoint(
        _settings(
            LLM_PROVIDER_NAME="cerebras",
            LLM_BASE_URL="https://api.cerebras.ai/v1",
            LLM_API_KEY="csk-x",
            LLM_MODEL="gemma-4-31b",
        )
    )
    assert (endpoint.base_url, endpoint.model, endpoint.api_key) == (
        "https://api.cerebras.ai/v1", "gemma-4-31b", "csk-x"
    )
    assert str(endpoint) == "cerebras/gemma-4-31b"


def test_the_service_talks_to_the_endpoint_it_was_given():
    cerebras = LLMEndpoint(
        name="cerebras", api_key="csk-x", base_url="https://api.cerebras.ai/v1", model="gemma-4-31b"
    )
    svc = ResilientLLMService(call_sid="t", endpoint=cerebras)
    assert str(svc._client.base_url).rstrip("/") == "https://api.cerebras.ai/v1"
    assert svc.endpoint.model == "gemma-4-31b"


def test_no_provider_class_is_hardcoded_in_the_agent():
    """A GroqLLMService constructed anywhere restores both the SDK's silent retries and a
    base URL that configuration can no longer move."""
    tree = ast.parse(inspect.getsource(agent))
    built = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    imported = {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert "GroqLLMService" not in built and "GroqLLMService" not in imported


def test_the_delay_is_cleared_after_a_good_turn():
    """Otherwise a throttle from turn 2 explains an unrelated failure on turn 7, and the
    call signs off for a reason that is no longer true."""
    src = inspect.getsource(ResilientLLMService.get_chat_completions)
    assert "_last_throttle_delay = None" in src


def test_the_error_handler_uses_the_delay_the_provider_sent():
    """The pipeline hands an error handler only a string, and Cerebras puts the number in a
    header. Without reading it back off the service there is nothing to classify on."""
    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def on_llm_error"):]
    assert "last_throttle_delay" in handler
    assert "is_transient_throttle(error.error, throttled_for)" in handler
    assert "is_quota_error(error.error, throttled_for)" in handler
