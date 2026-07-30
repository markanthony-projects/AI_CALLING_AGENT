"""The sign-off the agent speaks as it hangs up.

A live call ended like this:

    Prospect: Yeah, I'll be free on Sunday, so Sunday afternoon 3PM.
    Agent:    Thank you so much for your time. Have a wonderful day!

The visit was booked and stored, but the prospect was never told so — the farewell was a
fixed string that could not name a day or a time. The model now supplies the line, which
means the line also has to be vetted before it becomes the last thing the caller hears.
"""

import ast
import inspect

import pytest

from app.services import agent
from app.services.agent import FAREWELL_LINE, MAX_CLOSING_CHARS, closing_line

BOOKING_READBACK = "Perfect Kumar, that's Sunday at 3 PM at Lakeview Residency. I'll send you the details. Thank you!"


def _node(name):
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for n in ast.walk(tree):
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in run_voice_agent")


# ── the validator ──────────────────────────────────────────────────────────────

def test_model_line_is_spoken_verbatim():
    assert closing_line(BOOKING_READBACK) == BOOKING_READBACK


@pytest.mark.parametrize("junk", [None, "", "   ", "\n\t "])
def test_nothing_usable_falls_back_to_the_fixed_farewell(junk):
    assert closing_line(junk) == FAREWELL_LINE


def test_devanagari_falls_back():
    """Sarvam breaks up mid-word on mixed scripts; a garbled goodbye is worse than a generic one."""
    assert closing_line("Thank you कुमार, see you Sunday!") == FAREWELL_LINE


def test_monologue_falls_back():
    assert closing_line("word " * 200) == FAREWELL_LINE


@pytest.mark.parametrize(
    "hedged",
    [
        "Perfect Santosh, that's Sunday at a time to be decided at Lakeview Residency. Thank you!",
        "Great, Sunday at a time to be confirmed. I'll send the details!",
        "Lovely, that's Saturday, exact time TBD. Thank you!",
        "So that's Sunday, hour yet to be agreed. Thanks!",
    ],
)
def test_a_hedged_slot_is_not_a_confirmation(hedged):
    """Spoken verbatim on a live call: "that's Sunday at a time to be decided", then hangup.
    A placeholder read out loud is worse than a plain goodbye."""
    assert closing_line(hedged) == FAREWELL_LINE


def test_a_real_clock_time_is_not_mistaken_for_a_hedge():
    assert closing_line(BOOKING_READBACK) == BOOKING_READBACK


def test_a_full_readback_fits_within_the_cap():
    """The cap exists to stop rambling, not to reject the confirmation we are asking for."""
    assert len(BOOKING_READBACK) <= MAX_CLOSING_CHARS


def test_curly_apostrophes_survive():
    """A smart quote is not a script problem, and 'I’ll send you the details' is the line we want."""
    line = "Perfect, that’s Sunday at 3 PM. I’ll send you the details!"
    assert closing_line(line) == line


def test_newlines_are_flattened():
    assert closing_line("Perfect.\n  See you Sunday.") == "Perfect. See you Sunday."


# ── the tool the model actually sees ───────────────────────────────────────────

def test_tool_schema_requires_a_closing_line():
    """Built from the real end_call in run_voice_agent, so the signature cannot drift."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    ns = {}
    exec(compile(ast.Module([_node("end_call")], []), "<end_call>", "exec"), ns)

    ctx = LLMContext(messages=[{"role": "system", "content": "x"}], tools=[ns["end_call"]])
    schema = ctx.tools.direct_functions[0].to_function_schema().to_default_dict()

    assert schema["name"] == "end_call"
    assert "closing_line" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["closing_line"]["type"] == "string"
    assert "closing_line" in schema["parameters"]["required"], (
        "an optional closing_line lets the model hang up without confirming the booking"
    )
    described = schema["parameters"]["properties"]["closing_line"]["description"].lower()
    assert "read it back" in described or "read back" in described


# ── the handler ────────────────────────────────────────────────────────────────

class _Frame:
    def __init__(self, *args, **kwargs):
        pass


class _Speak(_Frame):
    def __init__(self, text=None):
        self.text = text


def _spoken(batches):
    """The line the caller actually hears, wherever it sits in the batch."""
    return next(f.text for batch in batches for f in batch if isinstance(f, _Speak))


class _Task:
    def __init__(self):
        self.batches = []

    async def queue_frames(self, frames):
        self.batches.append(frames)


def _run_handler(arguments):
    """Execute the real end_call_handler body against fakes."""
    task = _Task()
    ns = {
        "closing_line": closing_line,
        "logger": agent.logger,
        "call_sid": "sid",
        "task_ref": [task],
        "TTSSpeakFrame": _Speak,
        "EndFrame": _Frame,
        "InterruptionWorkerFrame": _Frame,
    }
    exec(compile(ast.Module([_node("end_call_handler")], []), "<handler>", "exec"), ns)

    import asyncio

    params = type("P", (), {"arguments": arguments})()
    asyncio.run(ns["end_call_handler"](params))
    return task.batches


def test_handler_speaks_the_models_confirmation():
    batches = _run_handler({"closing_line": BOOKING_READBACK})
    assert batches, "handler queued nothing"
    assert _spoken(batches) == BOOKING_READBACK, (
        "the booking read-back was dropped in favour of the generic farewell"
    )


def test_handler_falls_back_when_the_model_sends_nothing():
    assert _spoken(_run_handler({})) == FAREWELL_LINE


def test_handler_discards_speech_still_queued_behind_it():
    """A split turn ran two inferences: one asked "What time on Sunday?" while the other
    hung up. Both played, so the agent answered its own question and rang off."""
    node = _node("end_call_handler")
    frames = [
        el.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "queue_frames"
        for el in n.args[0].elts
        if isinstance(el, ast.Call)
    ]
    assert "InterruptionWorkerFrame" in frames, "stale speech is never flushed"
    assert frames.index("InterruptionWorkerFrame") < frames.index("TTSSpeakFrame"), (
        "the interruption must precede the farewell or it cancels the farewell instead"
    )


def test_handler_survives_a_params_object_without_arguments():
    """Pipecat hands us FunctionCallParams; a provider that omits args must not crash the hangup."""
    task = _Task()
    ns = {
        "closing_line": closing_line,
        "logger": agent.logger,
        "call_sid": "sid",
        "task_ref": [task],
        "TTSSpeakFrame": _Speak,
        "EndFrame": _Frame,
        "InterruptionWorkerFrame": _Frame,
    }
    exec(compile(ast.Module([_node("end_call_handler")], []), "<handler>", "exec"), ns)

    import asyncio

    asyncio.run(ns["end_call_handler"](None))
    assert _spoken(task.batches) == FAREWELL_LINE
