"""The primary's model is gone, and the call must not go with it.

Live call c15a5b2a, 10 Sep 2026. Cerebras answered every request for gemma-4-31b with

    404 — Model does not exist or you do not have access to it.  (model_not_found)

while still listing the model. The prospect said "Yeah, but regarding what", was told
"Sorry, I missed that", said it again, and was hung up on with "trouble on this line".

Two things let that happen. The warm-up fired before the opening line and saw the 404 — and
logged it at DEBUG, which the log filter drops. And the fallback only catches RateLimitError,
so a NotFoundError fell through it to the generic "turn failed" path, which asks the caller
to repeat themselves into a model that does not exist.
"""

import asyncio

import httpx
import pytest
from openai import BadRequestError, NotFoundError, RateLimitError
from pipecat.processors.aggregators.llm_context import LLMContext

from app.services.llm_provider import LLMEndpoint, ResilientLLMService

PRIMARY = LLMEndpoint(
    name="cerebras", api_key="k", base_url="https://api.cerebras.ai/v1", model="gemma-4-31b"
)
FALLBACK = LLMEndpoint(
    name="groq", api_key="k2", base_url="https://api.groq.com/openai/v1", model="llama-3.3-70b"
)


def _404() -> NotFoundError:
    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(404, request=request)
    return NotFoundError(
        "Error code: 404 - {'message': 'Model does not exist or you do not have access to it.'}",
        response=response,
        body={"code": "model_not_found"},
    )


def _429() -> RateLimitError:
    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "3"})
    return RateLimitError("Error code: 429", response=response, body=None)


class _Completions:
    """Stands in for client.chat.completions. Raises what it is told to, counts the calls."""

    def __init__(self, raise_=None, result="ok"):
        self.raise_ = raise_
        self.result = result
        self.calls = 0

    async def create(self, **params):
        self.calls += 1
        if self.raise_ is not None:
            raise self.raise_
        return self.result


def _service(fallback=None, primary_raises=None, fallback_result="from-fallback"):
    svc = ResilientLLMService(call_sid="t", endpoint=PRIMARY, fallback=fallback)
    svc._client.chat.completions = _Completions(raise_=primary_raises)
    if fallback is not None:
        svc._fallback_client.chat.completions = _Completions(result=fallback_result)
    return svc


def _ctx():
    return LLMContext(messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])


def _capture(logger_):
    lines = []
    handle = logger_.add(lambda m: lines.append(m), level="DEBUG")
    return lines, handle


# --- the warm-up says so, out loud -----------------------------------------------------------


def test_a_404_on_warm_up_is_an_error_not_a_debug_line():
    from loguru import logger

    lines, handle = _capture(logger)
    try:
        svc = _service(primary_raises=_404())
        asyncio.run(svc.warm_up())
    finally:
        logger.remove(handle)
    errors = [str(l) for l in lines if l.record["level"].name == "ERROR"]
    assert len(errors) == 1, [str(l) for l in lines]
    assert "has no model 'gemma-4-31b'" in errors[0]
    assert "Every turn will fail" in errors[0]


def test_the_warm_up_error_says_whether_there_is_a_way_out():
    from loguru import logger

    for fallback, expected in ((FALLBACK, "using groq/llama-3.3-70b instead"), (None, "No LLM_FALLBACK_* configured")):
        lines, handle = _capture(logger)
        try:
            asyncio.run(_service(fallback=fallback, primary_raises=_404()).warm_up())
        finally:
            logger.remove(handle)
        errors = [str(l) for l in lines if l.record["level"].name == "ERROR"]
        assert expected in errors[0], errors


def test_a_404_on_warm_up_marks_the_primary_as_gone():
    svc = _service(primary_raises=_404())
    assert svc._primary_unusable is False
    asyncio.run(svc.warm_up())
    assert svc._primary_unusable is True


def test_any_other_warm_up_failure_stays_quiet_and_marks_nothing():
    """A slow handshake is what the warm-up exists to absorb. It must not be promoted."""
    from loguru import logger

    lines, handle = _capture(logger)
    try:
        svc = _service(primary_raises=RuntimeError("connection reset"))
        asyncio.run(svc.warm_up())
    finally:
        logger.remove(handle)
    assert not [l for l in lines if l.record["level"].name == "ERROR"]
    assert svc._primary_unusable is False


# --- the turn goes to the fallback ---------------------------------------------------------


def test_a_404_on_a_turn_is_served_by_the_fallback():
    svc = _service(fallback=FALLBACK, primary_raises=_404())
    result = asyncio.run(svc.get_chat_completions(_ctx()))
    assert result == "from-fallback"
    assert svc._fallback_client.chat.completions.calls == 1


def test_a_404_with_no_fallback_still_raises():
    """There is nothing to switch to. Raising lets the agent's error handler sign off,
    which is worse than a served turn and better than silence."""
    svc = _service(fallback=None, primary_raises=_404())
    with pytest.raises(NotFoundError):
        asyncio.run(svc.get_chat_completions(_ctx()))


def test_after_the_first_404_the_primary_is_not_asked_again():
    """A model that is gone is gone for the call. Asking again each turn pays a failed
    round trip before every fallback request, on the caller's clock."""
    svc = _service(fallback=FALLBACK, primary_raises=_404())
    asyncio.run(svc.get_chat_completions(_ctx()))
    asyncio.run(svc.get_chat_completions(_ctx()))
    asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._client.chat.completions.calls == 1
    assert svc._fallback_client.chat.completions.calls == 3


def test_a_404_seen_on_warm_up_sends_the_very_first_turn_to_the_fallback():
    """The warm-up already knows. The first turn should not have to find out again."""
    svc = _service(fallback=FALLBACK, primary_raises=_404())
    asyncio.run(svc.warm_up())
    svc._client.chat.completions.calls = 0
    asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._client.chat.completions.calls == 0
    assert svc._fallback_client.chat.completions.calls == 1


def test_the_switch_is_logged_as_an_error_naming_both_sides():
    from loguru import logger

    lines, handle = _capture(logger)
    try:
        asyncio.run(_service(fallback=FALLBACK, primary_raises=_404()).get_chat_completions(_ctx()))
    finally:
        logger.remove(handle)
    errors = [str(l) for l in lines if l.record["level"].name == "ERROR"]
    assert len(errors) == 1
    assert "cerebras/gemma-4-31b" in errors[0]
    assert "groq/llama-3.3-70b" in errors[0]


# --- and a rate limit is still this minute's problem, not the call's -------------------------


def test_a_rate_limit_does_not_mark_the_primary_as_gone():
    svc = _service(fallback=FALLBACK, primary_raises=_429())
    result = asyncio.run(svc.get_chat_completions(_ctx()))
    assert result == "from-fallback"
    assert svc._primary_unusable is False


def test_after_a_rate_limit_the_primary_is_asked_again_next_turn():
    svc = _service(fallback=FALLBACK, primary_raises=_429())
    asyncio.run(svc.get_chat_completions(_ctx()))
    asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._client.chat.completions.calls == 2


def test_with_no_fallback_every_turn_after_the_404_still_fails_cleanly():
    """The flag is set on the first 404 whether or not there is a fallback. With none, the
    next turn must go back to the primary and raise the same NotFoundError — not take the
    early exit into a fallback client that is None and die of an AttributeError, which the
    agent's error handler does not know how to read."""
    svc = _service(fallback=None, primary_raises=_404())
    with pytest.raises(NotFoundError):
        asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._primary_unusable is True
    with pytest.raises(NotFoundError):
        asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._client.chat.completions.calls == 2, "nowhere else to go, so the primary is asked"


# --- and the same for a parameter the model does not take --------------------------------------


def _400() -> BadRequestError:
    request = httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Error code: 400 - {'message': \"Unsupported value 'none' for 'reasoning_effort'\"}",
        response=response,
        body={"code": "invalid_request_error"},
    )


def _endpoint_with_effort() -> LLMEndpoint:
    from dataclasses import replace as _replace

    return _replace(PRIMARY, reasoning_effort="none")


def test_a_rejected_parameter_is_served_by_the_fallback():
    """reasoning_effort is the parameter that varies between models — qwen takes "none" and
    gpt-oss does not — and it goes into every request. Refused, every turn of the call is
    refused identically: the same shape of failure as the 404, and the same treatment. The
    fallback gets the request without the primary's extra params, so it can answer."""
    svc = _service(fallback=FALLBACK, primary_raises=_400())
    assert asyncio.run(svc.get_chat_completions(_ctx())) == "from-fallback"


def test_a_rejected_parameter_with_no_fallback_still_raises():
    svc = _service(fallback=None, primary_raises=_400())
    with pytest.raises(BadRequestError):
        asyncio.run(svc.get_chat_completions(_ctx()))


def test_after_a_400_the_primary_is_not_asked_again():
    svc = _service(fallback=FALLBACK, primary_raises=_400())
    asyncio.run(svc.get_chat_completions(_ctx()))
    asyncio.run(svc.get_chat_completions(_ctx()))
    assert svc._client.chat.completions.calls == 1
    assert svc._fallback_client.chat.completions.calls == 2


def test_the_warm_up_names_the_setting_to_look_at():
    """A 400 says nothing about which parameter. The one that varies by model is named, so
    whoever reads the log knows where to look rather than reading the provider's prose."""
    from loguru import logger

    lines, handle = _capture(logger)
    try:
        svc = ResilientLLMService(call_sid="t", endpoint=_endpoint_with_effort(), fallback=FALLBACK)
        svc._client.chat.completions = _Completions(raise_=_400())
        svc._fallback_client.chat.completions = _Completions()
        asyncio.run(svc.warm_up())
    finally:
        logger.remove(handle)
    errors = [str(l) for l in lines if l.record["level"].name == "ERROR"]
    assert len(errors) == 1, errors
    assert "LLM_REASONING_EFFORT='none'" in errors[0]
    assert "Every turn will fail" in errors[0]


def test_a_400_on_warm_up_sends_the_first_turn_straight_to_the_fallback():
    svc = ResilientLLMService(call_sid="t", endpoint=_endpoint_with_effort(), fallback=FALLBACK)
    svc._client.chat.completions = _Completions(raise_=_400())
    svc._fallback_client.chat.completions = _Completions(result="from-fallback")
    asyncio.run(svc.warm_up())
    svc._client.chat.completions.calls = 0
    assert asyncio.run(svc.get_chat_completions(_ctx())) == "from-fallback"
    assert svc._client.chat.completions.calls == 0
