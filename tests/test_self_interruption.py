"""The agent cutting off its own sentence to say goodbye.

Live call c085e397, 4 Sep 2026, final turn:

    USER  -> "Around in next 6 months."
    AGENT initiated call end via tool -> "Thank you for sharing these details, Rahul..."
    AGENT -> "That works well. Since you are looking in North Bangalore, I will have our
              property expert suggest some better options for you."  [interrupted]
    AGENT -> "Thank you for sharing these details, Rahul. Our team will call you soon..."

One inference produced both the reply and the tool call. end_call opens with an
InterruptionWorkerFrame, which exists to discard a STALE reply from a split turn — one
inference asking "What time on Sunday?" while another hangs up. This reply was not stale.
It was the same turn's lead-in, and the prospect heard half of it and then a farewell.

So the interruption now applies only when the words on the wire came from an earlier
inference. Telling the two apart has to survive frame ordering: end_call is dispatched
from inside the LLM service before LLMFullResponseEndFrame is pushed, but they are
separate tasks and either can land first.
"""

import ast
import asyncio
import inspect

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app.utils.farewell import FarewellGate
from app.utils.spoken_text import ToolSyntaxFilter

LEAD_IN = (
    "That works well. Since you are looking in North Bangalore, I will have our property "
    "expert suggest some better options for you."
)


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _feed(processor, frames):
    """Drive the real process_frame, bypassing only the started-processor check."""
    import pipecat.processors.frame_processor as fp

    captured = _Captured()
    processor.push_frame = captured.push
    processor.process_frame = processor.__class__.process_frame.__get__(processor)

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


# --- telling a lead-in from a leftover ---------------------------------------------------


def test_nothing_has_been_said_before_the_first_response():
    """The opening greeting is queued as a TTSSpeakFrame and never passes through here, so
    a hangup during it must still take the interrupting path."""
    assert ToolSyntaxFilter("sid").lead_in == ""


def test_what_this_response_has_spoken_is_the_lead_in():
    filt = ToolSyntaxFilter("sid")
    asyncio.run(_feed(filt, [LLMFullResponseStartFrame(), LLMTextFrame(LEAD_IN)]))
    assert filt.lead_in == LEAD_IN


def test_it_survives_the_end_of_the_response():
    """The load-bearing property. run_function_calls is invoked at the end of
    _process_context and LLMFullResponseEndFrame is pushed in the finally that follows, but
    the function call runs as its own task — so end_call can read this either before or
    after that frame arrives, and must get the same answer both times."""
    filt = ToolSyntaxFilter("sid")
    asyncio.run(
        _feed(
            filt,
            [LLMFullResponseStartFrame(), LLMTextFrame(LEAD_IN), LLMFullResponseEndFrame()],
        )
    )
    assert filt.lead_in == LEAD_IN


def test_a_reply_from_the_previous_response_is_not_a_lead_in():
    """This is the stale case the interruption was built for, and it must keep working:
    one inference speaks, a second one hangs up, and the first is left on the wire."""
    filt = ToolSyntaxFilter("sid")
    asyncio.run(
        _feed(
            filt,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("What time on Sunday works for you?"),
                LLMFullResponseEndFrame(),
                LLMFullResponseStartFrame(),
            ],
        )
    )
    assert filt.lead_in == ""


def test_a_response_that_spoke_nothing_has_no_lead_in():
    """The ordinary close: the model calls end_call and writes no text at all. There is
    nothing to protect, so the interruption still runs and clears anything left over."""
    filt = ToolSyntaxFilter("sid")
    asyncio.run(_feed(filt, [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()]))
    assert filt.lead_in == ""


def test_a_response_whose_every_word_was_tool_syntax_has_no_lead_in():
    """Suppressed markup never reaches the voice engine, so there is no audio to wait for.
    Waiting on speech that was never sent would stall the goodbye until the ceiling."""
    filt = ToolSyntaxFilter("sid")
    asyncio.run(
        _feed(
            filt,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame('<function=end_call>{"closing_line": "Bye"}</function>'),
            ],
        )
    )
    assert filt.lead_in == ""


def test_streamed_chunks_accumulate_into_one_lead_in():
    """The wait is sized to this sentence, so it has to be the whole sentence and not the
    last token of it."""
    filt = ToolSyntaxFilter("sid")
    asyncio.run(
        _feed(
            filt,
            [
                LLMFullResponseStartFrame(),
                LLMTextFrame("That works well. "),
                LLMTextFrame("I will have our expert call you."),
            ],
        )
    )
    assert filt.lead_in == "That works well. I will have our expert call you."


# --- knowing whether anything is on the wire ---------------------------------------------


def test_the_gate_starts_quiet():
    assert FarewellGate().is_speaking is False


def test_it_knows_while_the_bot_is_speaking():
    gate = FarewellGate()
    asyncio.run(_feed(gate, [BotStartedSpeakingFrame()]))
    assert gate.is_speaking is True


def test_it_knows_when_the_bot_has_stopped():
    gate = FarewellGate()
    asyncio.run(_feed(gate, [BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()]))
    assert gate.is_speaking is False


def test_waiting_for_quiet_returns_at_once_when_nothing_is_playing():
    """end_call can fire after the whole reply has already been played out. Blocking on an
    event in that case would add the entire ceiling to a goodbye that needed no wait."""
    gate = FarewellGate()

    async def scenario():
        return await gate.wait_for_quiet(30)

    assert asyncio.run(asyncio.wait_for(scenario(), 1)) is True


def test_waiting_for_quiet_holds_until_the_sentence_finishes():
    gate = FarewellGate()

    async def scenario():
        await _feed(gate, [BotStartedSpeakingFrame()])
        waiting = asyncio.create_task(gate.wait_for_quiet(5))
        await asyncio.sleep(0)
        assert not waiting.done(), "the goodbye went out over the lead-in"
        await _feed(gate, [BotStoppedSpeakingFrame()])
        return await waiting

    assert asyncio.run(scenario()) is True


def test_waiting_for_quiet_gives_up_rather_than_holding_the_leg_open():
    """A TTS that dies mid-sentence never raises BotStoppedSpeakingFrame. The goodbye goes
    out over it rather than the carrier leg staying up for ever."""
    gate = FarewellGate()

    async def scenario():
        await _feed(gate, [BotStartedSpeakingFrame()])
        return await gate.wait_for_quiet(0.05)

    assert asyncio.run(scenario()) is False


# --- and how end_call uses them -----------------------------------------------------------


def _goodbye_source():
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "say_goodbye_then_hang_up":
            return node
    raise AssertionError("say_goodbye_then_hang_up not found")


def _branch_on_lead_in(func):
    for node in func.body:
        if isinstance(node, ast.If) and "lead_in" in ast.unparse(node.test):
            return node
    raise AssertionError("no branch on the lead-in")


def test_a_lead_in_from_this_turn_is_never_interrupted():
    branch = _branch_on_lead_in(_goodbye_source())
    kept = "".join(ast.unparse(n) for n in branch.body)
    assert "InterruptionWorkerFrame" not in kept, "the fresh reply is still being cut off"
    assert "wait_for_quiet" in kept


def test_anything_else_still_is():
    """Removing the interruption altogether would bring back the split-turn bug it was
    written for: a question from one inference followed straight by another's goodbye."""
    branch = _branch_on_lead_in(_goodbye_source())
    otherwise = "".join(ast.unparse(n) for n in branch.orelse)
    assert "InterruptionWorkerFrame" in otherwise
    assert "flush_pipeline" in otherwise


def test_the_wait_is_sized_to_the_lead_in_and_not_fixed():
    """A two-word lead-in must not hold the line for the length of a paragraph, and a long
    one must not be cut off by a ceiling meant for a short one."""
    branch = _branch_on_lead_in(_goodbye_source())
    assert "farewell_timeout(lead_in)" in "".join(ast.unparse(n) for n in branch.body)


def test_the_goodbye_is_still_waited_for_either_way():
    """Whichever branch ran, the closing line itself is what the prospect must actually
    hear before the carrier leg drops."""
    src = ast.unparse(_goodbye_source())
    assert "wait_until_spoken(farewell_timeout(line))" in src
    assert "EndWorkerFrame" in src


# --- and the rule the model was given -------------------------------------------------------


def test_the_prompt_tells_the_model_not_to_write_both():
    """The code fix is a safety net. The turn is better still if the model never produces a
    reply and a hangup in the same breath, because then nothing has to be waited on."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "ENTIRE reply for that turn" in prompt
    assert "cut off in the middle" in prompt


@pytest.mark.parametrize("name", ["is_speaking", "wait_for_quiet"])
def test_the_gate_exposes_what_end_call_needs(name):
    assert hasattr(FarewellGate(), name)
