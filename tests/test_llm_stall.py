"""A primary that accepts the request and sends nothing is not waited for.

Call 81bdc87a, 15 Sep 2026, turn 4. Cerebras took the request and produced no chunk for
nineteen seconds — no error, so nothing fell back; the prospect said "Hello?" twice and
hung up. The stream now has a deadline on its FIRST chunk only. A stream that has started
is left alone however long it takes; one that has not started by the deadline is closed
and the same turn is sent to the fallback.
"""

import asyncio
import inspect

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from app.services.llm_provider import LLMEndpoint, ResilientLLMService, build_llm_service

PRIMARY = LLMEndpoint(name="cerebras", api_key="k", base_url="https://api.cerebras.ai/v1", model="m")
FALLBACK = LLMEndpoint(name="openai", api_key="k2", base_url="https://api.openai.com/v1", model="f")


class _Stream:
    """An OpenAI AsyncStream stand-in: chunks after a delay, and a close() that is counted."""

    def __init__(self, chunks, first_after=0.0):
        self._chunks = list(chunks)
        self._first_after = first_after
        self.closed = 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        await asyncio.sleep(self._first_after)
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        self.closed += 1


class _Completions:
    def __init__(self, stream):
        self.stream = stream
        self.calls = 0

    async def create(self, **params):
        self.calls += 1
        return self.stream


def _service(primary_stream, fallback_stream=None, deadline=0.05):
    svc = ResilientLLMService(
        call_sid="t",
        endpoint=PRIMARY,
        fallback=FALLBACK if fallback_stream is not None else None,
        first_token_deadline=deadline,
    )
    svc._client.chat.completions = _Completions(primary_stream)
    if fallback_stream is not None:
        svc._fallback_client.chat.completions = _Completions(fallback_stream)
    return svc


def _ctx():
    return LLMContext(messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])


async def _collect(svc):
    stream = await svc.get_chat_completions(_ctx())
    out = []
    async for chunk in stream:
        out.append(chunk)
    return out


def test_a_stream_that_starts_in_time_passes_through_untouched():
    primary = _Stream(["a", "b", "c"], first_after=0.0)
    fallback = _Stream(["fb"])
    svc = _service(primary, fallback)
    assert asyncio.run(_collect(svc)) == ["a", "b", "c"]
    assert svc.stalls == 0
    assert svc._fallback_client.chat.completions.calls == 0


def test_a_stream_that_has_not_started_by_the_deadline_goes_to_the_fallback():
    primary = _Stream(["late"], first_after=0.5)
    fallback = _Stream(["fb1", "fb2"])
    svc = _service(primary, fallback, deadline=0.05)
    assert asyncio.run(_collect(svc)) == ["fb1", "fb2"]
    assert svc.stalls == 1
    assert primary.closed >= 1, "the stalled request is released, not left open"
    assert svc._fallback_client.chat.completions.calls == 1


def test_a_slow_middle_is_not_a_stall():
    """Only the first chunk has a deadline. A model that thinks between sentences is left
    to finish; cutting it there would turn every long answer into a fallback."""

    class _Gappy(_Stream):
        async def _gen(self):
            yield "first"
            await asyncio.sleep(0.2)
            yield "second"

    svc = _service(_Gappy([]), _Stream(["fb"]), deadline=0.05)
    assert asyncio.run(_collect(svc)) == ["first", "second"]
    assert svc.stalls == 0


def test_without_a_fallback_the_stall_is_raised_and_logged():
    from loguru import logger

    lines = []
    handle = logger.add(lambda m: lines.append(m), level="ERROR")
    try:
        svc = _service(_Stream(["late"], first_after=0.5), None, deadline=0.05)
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(_collect(svc))
    finally:
        logger.remove(handle)
    assert any("sent nothing for 0.1s" in str(l) or "sent nothing for 0.0s" in str(l) or "sent nothing" in str(l) for l in lines)


def test_the_stall_is_a_warning_that_names_both_endpoints():
    from loguru import logger

    lines = []
    handle = logger.add(lambda m: lines.append(m), level="WARNING")
    try:
        svc = _service(_Stream(["late"], first_after=0.5), _Stream(["fb"]), deadline=0.05)
        asyncio.run(_collect(svc))
    finally:
        logger.remove(handle)
    warning = next(str(l) for l in lines if "sent nothing" in str(l))
    assert "cerebras" in warning and "openai" in warning


def test_no_deadline_means_the_stream_is_returned_as_is():
    primary = _Stream(["a"])
    svc = ResilientLLMService(call_sid="t", endpoint=PRIMARY)
    svc._client.chat.completions = _Completions(primary)
    assert asyncio.run(svc.get_chat_completions(_ctx())) is primary


def test_the_agent_gets_the_deadline_from_the_setting():
    from app.core.config import Settings

    assert "first_token_deadline=settings.LLM_FIRST_TOKEN_DEADLINE_SECS" in inspect.getsource(build_llm_service)
    field = Settings.model_fields["LLM_FIRST_TOKEN_DEADLINE_SECS"]
    assert field.default == 2.5
    bounds = {type(m).__name__: m for m in field.metadata}
    assert bounds["Ge"].ge >= 0.5, "below this a slow but healthy primary would be abandoned every turn"


def test_the_wrapper_answers_everything_the_base_class_calls_on_a_stream():
    """base_llm iterates with __aiter__, then closes the iterator and the stream."""
    svc = _service(_Stream(["a"]), _Stream(["fb"]))
    wrapped = svc._within_deadline(_Stream(["a"]), _ctx())
    assert hasattr(wrapped, "__aiter__") and hasattr(wrapped, "aclose")
    asyncio.run(wrapped.aclose())
