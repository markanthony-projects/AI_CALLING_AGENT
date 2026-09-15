"""A reply with nothing in it is noticed and answered, not left as silence.

Call 29b2f355, 14 Sep 2026, turn 3: "I was looking something near Whitefield." — six seconds
of nothing, then "Did you get it?". No error, no LATENCY line, nothing that could say a reply
had been produced with no words in it. This guard sits after the cutter and before the
voice engine, so what it counts is what would actually have been spoken.
"""

import asyncio
import inspect

import pytest

from app.utils.empty_reply import EmptyReplyGuard


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _run(guard, frames):
    import pipecat.processors.frame_processor as fp
    from pipecat.processors.frame_processor import FrameDirection

    captured = _Captured()
    guard.push_frame = captured.push
    guard.process_frame = guard.__class__.process_frame.__get__(guard)

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await guard.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured


def _frames():
    from pipecat.frames.frames import (
        FunctionCallInProgressFrame,
        InterruptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )

    return LLMFullResponseStartFrame, LLMFullResponseEndFrame, LLMTextFrame, FunctionCallInProgressFrame, InterruptionFrame


def _guard():
    seen = []

    async def on_empty(count):
        seen.append(count)

    return EmptyReplyGuard("sid", on_empty=on_empty), seen


def test_a_reply_with_words_is_not_empty():
    Start, End, Text, *_ = _frames()
    guard, seen = _guard()
    asyncio.run(_run(guard, [Start(), Text("Whitefield, noted. "), Text("Own stay or investment?"), End()]))
    assert guard.empty == 0 and seen == []


def test_a_reply_with_nothing_in_it_is_counted_and_reported():
    Start, End, *_ = _frames()
    guard, seen = _guard()
    asyncio.run(_run(guard, [Start(), End()]))
    assert guard.empty == 1 and seen == [1]


def test_whitespace_is_nothing():
    Start, End, Text, *_ = _frames()
    guard, seen = _guard()
    asyncio.run(_run(guard, [Start(), Text("  \n"), End()]))
    assert guard.empty == 1 and seen == [1]


def test_a_tool_call_is_a_reply():
    """end_call with no spoken words is the model hanging up, not the model saying nothing."""
    Start, End, Text, ToolCall, _ = _frames()
    guard, seen = _guard()
    asyncio.run(
        _run(
            guard,
            [Start(), ToolCall(function_name="end_call", tool_call_id="t1", arguments={}), End()],
        )
    )
    assert guard.empty == 0 and seen == []


def test_an_interrupted_reply_is_unfinished_not_empty():
    Start, End, Text, _, Interruption = _frames()
    guard, seen = _guard()
    asyncio.run(_run(guard, [Start(), Interruption(), End()]))
    assert guard.empty == 0 and seen == []


def test_every_frame_still_passes_through():
    Start, End, Text, *_ = _frames()
    guard, _ = _guard()
    captured = asyncio.run(_run(guard, [Start(), Text("Hi."), End()]))
    assert [type(f).__name__ for f in captured.frames] == [
        "LLMFullResponseStartFrame",
        "LLMTextFrame",
        "LLMFullResponseEndFrame",
    ]


# --- wired, and answered the same way an empty prospect turn is --------------------------------


def test_the_guard_sits_after_the_cutter_and_before_the_voice():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    order = src[src.index("pipeline = Pipeline([") :]
    assert order.index("one_question,") < order.index("empty_reply,") < order.index("tts,")


def test_an_empty_reply_asks_the_last_question_again():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "async def on_empty_reply(count: int)" in src
    assert 'await ask_again(f"The model replied with nothing' in src
    # Both doors count against the same ceiling: the prospect's empty turn (inline in the
    # handler, its guard shape pinned by test_hold_request) and the model's empty reply.
    helper = src[src.index("async def ask_again(") :]
    assert "_dead_air_nudges >= MAX_DEAD_AIR_NUDGES" in helper
    assert "nudge == _last_nudged" in helper
    assert "_holding or _ending" in helper


def test_a_reply_that_never_starts_is_named_in_the_log():
    """No response at all is invisible to the guard — there is no response to be empty. The
    watchdog names it, with whether an inference was even in flight."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "async def watch_for_reply(serial: int)" in src
    assert "No reply" in src and "inference in flight=" in src
    assert agent.REPLY_WATCHDOG_SECS > 2.0, "shorter than a slow healthy turn is a false alarm"
    assert agent.REPLY_WATCHDOG_SECS < 6.0, "the prospect asked 'Did you get it?' at six"


def test_the_started_frame_counts_because_it_is_the_one_that_arrives_in_time():
    """Call b6d2bea9, 15 Sep 2026: the end_call turn was reported as empty. The LLM service
    broadcasts FunctionCallsStartedFrame synchronously and pushes LLMFullResponseEndFrame
    in its finally; FunctionCallInProgressFrame comes from the call's task, after both."""
    from pipecat.frames.frames import FunctionCallsStartedFrame

    Start, End, *_ = _frames()
    guard, seen = _guard()
    asyncio.run(_run(guard, [Start(), FunctionCallsStartedFrame(function_calls=[]), End()]))
    assert guard.empty == 0 and seen == []


def test_the_order_pipecat_actually_uses_is_the_one_tested():
    import inspect

    from pipecat.services import llm_service
    from pipecat.services.openai import base_llm

    run = inspect.getsource(llm_service.LLMService.run_function_calls)
    assert "broadcast_frame(FunctionCallsStartedFrame" in run
    process = inspect.getsource(base_llm.BaseOpenAILLMService.process_frame)
    assert process.index("_process_context(") < process.index("LLMFullResponseEndFrame()")
