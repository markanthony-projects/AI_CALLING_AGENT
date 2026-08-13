"""The caller must not be asked the same question twice.

From a live call, one prospect turn produced two spoken replies:

    USER  → "Yeah एक साल sounds interesting."   (turn duration 1626ms)
    LATENCY turn 4: 90ms voice-to-voice | sarvam=227ms      <- no LLM timing at all
    AGENT → "That works well, Bhupendra. Since you are planning for a year from now,
             are you looking for a home to live in, or for investment?"
    AGENT → "That sounds good, Bhupendra. Are you looking for a home to stay in, or is
             this for investment?"

Pipecat fires inference speculatively while a turn is still open and declines to finalize
the turn while the user is audible, so a prospect who pauses and carries on gets one
inference on the fragment and another on the whole utterance. Both were spoken.
"""

import asyncio
import inspect

import pytest
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app.utils.turn_gate import TurnFinalityGate


class _Sink:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.frames.append(frame)

    @property
    def spoken(self) -> str:
        return "".join(getattr(f, "text", "") for f in self.frames)


def _gate():
    """A gate with the base class's plumbing bypassed; the logic under test is ours."""
    gate = TurnFinalityGate("sid")
    sink = _Sink()
    gate.push_frame = sink.push
    gate.process_frame = gate.__class__.process_frame.__get__(gate)

    import pipecat.processors.frame_processor as fp

    async def noop(*a, **kw):
        pass

    gate._noop_patch = (fp.FrameProcessor.process_frame, noop)
    return gate, sink


async def _send(gate, *frames):
    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame

    async def noop(*a, **kw):
        pass

    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original


def _reply(*chunks):
    return [LLMFullResponseStartFrame(), *(LLMTextFrame(c) for c in chunks), LLMFullResponseEndFrame()]


# --- the call this came from ---------------------------------------------------------


def test_the_reply_to_half_a_sentence_is_never_spoken():
    """The whole point. Reproduces turn 4 exactly: inference fires while the prospect is
    still talking, they carry on, and a second inference answers the full utterance."""
    gate, sink = _gate()

    gate.user_turn_started()
    gate.inference_triggered()  # speculative, prospect still speaking
    asyncio.run(_send(gate, *_reply("That works well, Bhupendra. ", "Planning for a year from now?")))
    assert sink.spoken == "", "a reply generated mid-sentence must not reach the TTS"

    gate.inference_triggered()  # the prospect finished; the real inference
    asyncio.run(gate.user_turn_stopped())
    assert sink.spoken == "", "and the stale one must be dropped, not released late"

    asyncio.run(_send(gate, *_reply("That sounds good, Bhupendra. For investment?")))
    assert sink.spoken == "That sounds good, Bhupendra. For investment?"


def test_the_ordinary_turn_is_not_delayed_at_all():
    """The common case: inference and finalize fire together, so the turn is already closed
    by the time the model answers. A gate that held here would add latency to every turn."""
    gate, sink = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(gate.user_turn_stopped())

    asyncio.run(_send(gate, LLMFullResponseStartFrame(), LLMTextFrame("Wonderful, Rahul.")))
    assert sink.spoken == "Wonderful, Rahul.", "must stream, not wait for the response to end"


def test_a_speculative_reply_survives_when_nothing_supersedes_it():
    """The prospect paused, inference fired, and then they simply stopped. Nothing newer
    arrived, so the held reply is the right answer and must be released rather than lost."""
    gate, sink = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(_send(gate, *_reply("That works well.")))
    assert sink.spoken == ""

    asyncio.run(gate.user_turn_stopped())  # no new inference
    assert sink.spoken == "That works well."


def test_the_end_frame_is_held_with_its_text():
    """Releasing the terminator while holding the words would have the assistant
    aggregator record an empty turn and the TTS speak nothing."""
    gate, sink = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(_send(gate, *_reply("Held.")))
    assert not any(isinstance(f, LLMFullResponseEndFrame) for f in sink.frames)

    asyncio.run(gate.user_turn_stopped())
    assert isinstance(sink.frames[-1], LLMFullResponseEndFrame)


def test_a_third_inference_supersedes_the_second():
    """A prospect who stops and starts twice gets three inferences. Only the last is an
    answer to what they actually said."""
    gate, sink = _gate()
    gate.user_turn_started()
    for text in ("first. ", "second. "):
        gate.inference_triggered()
        asyncio.run(_send(gate, *_reply(text)))
    gate.inference_triggered()
    asyncio.run(gate.user_turn_stopped())
    asyncio.run(_send(gate, *_reply("third.")))
    assert sink.spoken == "third."


def test_an_interruption_discards_whatever_is_held():
    """Speaking it afterwards would answer a question the prospect has moved on from."""
    gate, sink = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(_send(gate, *_reply("stale")))
    asyncio.run(_send(gate, InterruptionFrame()))
    asyncio.run(gate.user_turn_stopped())
    assert "stale" not in sink.spoken


def test_frames_that_are_not_llm_text_pass_straight_through():
    """The greeting is a TTSSpeakFrame and has no turn to be final about."""
    gate, sink = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(_send(gate, TTSSpeakFrame("Hi, I am Priya.")))
    assert sink.frames and isinstance(sink.frames[0], TTSSpeakFrame)


def test_held_replies_are_counted_for_the_log():
    gate, _ = _gate()
    gate.user_turn_started()
    gate.inference_triggered()
    asyncio.run(_send(gate, *_reply("stale")))
    gate.inference_triggered()
    asyncio.run(gate.user_turn_stopped())
    assert gate.dropped == 1


# --- the assumption the whole design rests on ----------------------------------------


def test_the_inference_event_really_does_fire_before_the_finalize_event():
    """If Pipecat ever reversed these, a reply about to be superseded would still read as
    current when the turn closes, and it would be released and spoken — the original bug,
    reintroduced silently. This is the one external fact the gate cannot verify at runtime.
    """
    from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy

    src = inspect.getsource(BaseUserTurnStopStrategy.trigger_user_turn_stopped)
    assert src.index("trigger_user_turn_inference_triggered") < src.index("trigger_user_turn_finalized")


# --- wiring ---------------------------------------------------------------------------


def test_the_gate_sits_above_the_tool_syntax_filter():
    """A reply to half a sentence must not reach the leaked-end_call path either: a model
    answering a fragment has decided the conversation was over before now."""
    import ast

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    stages = next(
        ast.unparse(n.value.args[0])
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "pipeline" for t in n.targets)
    )
    names = [s.strip() for s in stages.strip("[]").split(",")]
    assert names.index("llm") < names.index("turn_gate") < names.index("tool_syntax_filter")


@pytest.mark.parametrize(
    "event,call",
    [
        ("on_user_turn_inference_triggered", "turn_gate.inference_triggered()"),
        ("on_user_turn_started", "turn_gate.user_turn_started()"),
        ("on_user_turn_stopped", "await turn_gate.user_turn_stopped()"),
    ],
)
def test_the_gate_is_fed_every_edge_it_needs(event, call):
    """Miss any one and it either never holds, never releases, or never supersedes."""
    import ast

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == event
    )
    assert call in ast.unparse(handler)


# --- the goodbye that got cut off ------------------------------------------------------


def test_a_spoken_farewell_flushes_before_the_pipeline_stops():
    """On a live call end_call fired at 17:37:46.471 and the pipeline was finished by
    17:37:46.896 — 425ms for a sentence that takes about three seconds. EndFrame stops the
    transport as soon as it is received in queue order; EndWorkerFrame flushes first."""
    import ast

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        rendered = ast.unparse(node)
        if "TTSSpeakFrame" in rendered and "EndFrame()" in rendered:
            pytest.fail(f"speech followed by EndFrame is cut off mid-sentence: {rendered}")
