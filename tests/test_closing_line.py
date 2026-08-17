"""The sign-off the agent speaks as it hangs up.

A live call ended like this:

    Prospect: Yeah, I'll be free on Sunday, so Sunday afternoon 3PM.
    Agent:    Thank you so much for your time. Have a wonderful day!

The visit was booked and stored, but the prospect was never told so — the farewell was a
fixed string that could not name a day or a time. The model now supplies the line, which
means the line also has to be vetted before it becomes the last thing the caller hears.
"""

import ast
import asyncio
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

    async def flush_pipeline(self, timeout=None):
        return True


class _Farewell:
    """Stands in for FarewellGate. Reports the goodbye as heard, so the handler proceeds."""

    def __init__(self, heard=True):
        self.heard = heard
        self.armed = False

    def arm(self):
        self.armed = True

    async def wait_until_spoken(self, timeout):
        assert self.armed, "the gate must be armed before the farewell is queued"
        return self.heard


def _build(task, farewell):
    """Compile the real end_call_handler and its helper together.

    They have to be built inside one enclosing function because end_call_handler declares
    `nonlocal _ending` and closes over say_goodbye_then_hang_up. Exec'ing either alone is a
    SyntaxError, and rewriting them for the test would be testing the rewrite.
    """
    spawned = []

    class _AsyncioShim:
        @staticmethod
        def create_task(coro):
            # asyncio.run() would cancel a real pending task the moment the handler returns.
            spawned.append(coro)
            return coro

    outer = ast.parse(
        "def _outer():\n    _ending = False\n    return end_call_handler\n"
    ).body[0]
    outer.body[1:1] = [_node("say_goodbye_then_hang_up"), _node("end_call_handler")]

    ns = {
        "closing_line": closing_line,
        "logger": agent.logger,
        "call_sid": "sid",
        "task_ref": [task],
        "farewell": farewell,
        "farewell_timeout": lambda line: 1.0,
        "asyncio": _AsyncioShim,
        "TTSSpeakFrame": _Speak,
        "EndFrame": _Frame,
        # The farewell is followed by EndWorkerFrame, not EndFrame: EndFrame stops the
        # transport in queue order and cut a live goodbye off after 425ms.
        "EndWorkerFrame": _Frame,
        "InterruptionWorkerFrame": _Frame,
    }
    exec(compile(ast.Module([outer], []), "<handler>", "exec"), ns)
    return ns["_outer"](), spawned


def _run_handler(arguments):
    """Execute the real end_call path against fakes, including the detached hangup."""
    task = _Task()
    handler, spawned = _build(task, _Farewell())
    params = type("P", (), {"arguments": arguments})()

    async def drive():
        await handler(params)
        for coro in spawned:
            await coro

    asyncio.run(drive())
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
    batches = _run_handler({"closing_line": BOOKING_READBACK})
    kinds = [[type(f).__name__ for f in batch] for batch in batches]
    flat = [k for batch in kinds for k in batch]
    # _Frame stands in for every non-speech frame, so position is what carries the meaning:
    # something is queued before the farewell, and something after it.
    assert flat.index("_Speak") > 0, "nothing precedes the farewell — stale speech is never flushed"
    assert flat.index("_Speak") < len(flat) - 1, "nothing follows the farewell — the call never ends"


def test_a_second_hangup_is_refused():
    """A model can emit a structured tool call and leaked syntax for the same turn, so both
    end paths fire. Each one opens with an interruption, so the second would cancel the
    first one's goodbye mid-sentence — exactly the cutoff this path exists to prevent."""
    import asyncio

    task = _Task()
    handler, spawned = _build(task, _Farewell())
    params = type("P", (), {"arguments": {"closing_line": BOOKING_READBACK}})()

    async def drive():
        await handler(params)
        await handler(params)
        for coro in spawned:
            await coro

    asyncio.run(drive())
    spoken = [f for batch in task.batches for f in batch if isinstance(f, _Speak)]
    assert len(spoken) == 1, f"the farewell was queued {len(spoken)} times"


def test_the_goodbye_is_hung_up_on_even_if_it_never_plays():
    """A dead TTS raises no BotStoppedSpeakingFrame. Waiting for one that is not coming
    would hold the carrier leg open and go on billing."""
    import asyncio

    task = _Task()
    handler, spawned = _build(task, _Farewell(heard=False))

    async def drive():
        await handler(type("P", (), {"arguments": {"closing_line": BOOKING_READBACK}})())
        for coro in spawned:
            await coro

    asyncio.run(drive())
    assert task.batches, "the call was never ended"
    assert len(task.batches) >= 3


def test_handler_survives_a_params_object_without_arguments():
    """Pipecat hands us FunctionCallParams; a provider that omits args must not crash the hangup."""
    import asyncio

    task = _Task()
    handler, spawned = _build(task, _Farewell())

    async def drive():
        await handler(None)
        for coro in spawned:
            await coro

    asyncio.run(drive())
    assert _spoken(task.batches) == FAREWELL_LINE
