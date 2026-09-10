""""Say it again" is not a goodbye.

Live call 6a58a7f4, 10 Sep 2026, forty-one seconds long:

    USER  -> "Yeah, I was interested."
    AGENT -> "It sits on 45 acres with 14 towers. Do you know Varthur?"
    USER  -> "Sorry, I did not catch that. Can you say it again?"
    AGENT -> end_call: "Sorry about that Rahul. I will call you back later."

The prompt forbids this twice over — the OBJECTIONS section in capitals ("NEVER offer a
callback for this") and the tool's own description ("Do NOT call it for a 'hello' or an
interruption"). The model did it anyway and hung up on a lead that had just said it was
interested. So it is code now, like the budget, the locality and the invented booking
before it.
"""

import ast
import asyncio
import inspect

import pytest

from app.utils.repeat_request import (
    MAX_REPEAT_REFUSALS,
    REFUSAL_REASON,
    say_again,
    wants_repeat,
)


# --- what counts as asking to hear it again ---------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Sorry, I did not catch that. Can you say it again?",  # the live one
        "I did not understand ma'am.",                          # also live, 10 Sep
        "Sorry, I didn't catch that.",
        "Come again?",
        "Can you repeat that?",
        "Could you say that again please",
        "Once more?",
        "One more time please",
        "Pardon?",
        "What did you say?",
        "I can't hear you",
        "Your voice is not clear",
        "You are breaking up",
        "Sorry samajh nahi aaya",
        "Phir se boliye",
        "Dobara bataiye",
    ],
)
def test_a_request_to_hear_it_again_is_recognised(line):
    assert wants_repeat(line) is True


@pytest.mark.parametrize(
    "line",
    ["Hello?", "hello", "Hello.", "Are you there?", "Can you hear me?", "Is anyone there"],
)
def test_checking_the_line_counts_too(line):
    """They heard silence and are checking the call is still on — the OBJECTIONS section
    says in capitals that this is not a brush-off."""
    assert wants_repeat(line) is True


# --- and what does not -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Not interested, please do not call again.",  # contains "again"
        "Don't call me again",
        "Stop calling me",
        "Please remove my number",
        "Not interested",
        # The cases the refusal guard actually exists for: a bad line and a refusal in the
        # same breath. Every one of these matches a repeat pattern too, and without the
        # guard checked first they would hold the call open on someone leaving. Mutation
        # testing found the guard removable until these were here.
        "Sorry, I didn't catch that, but not interested.",
        "I can't hear you, please don't call again.",
        "Pardon? No, stop calling me.",
        "Say that again — actually no, remove my number.",
    ],
)
def test_a_refusal_is_never_read_as_a_request_to_repeat(line):
    """The one direction this must never get wrong: a person trying to leave has to be
    allowed to leave, and half of these say "again" while doing it."""
    assert wants_repeat(line) is False


@pytest.mark.parametrize(
    "line",
    [
        "Yeah, I was interested.",
        "Anything in 3 BHK",
        "My budget is around 1.5 Crores",
        "Hello, yes, who is this?",   # "hello" with a conversation after it
        "I already told you that",
        "Saturday 3 PM works",
        "",
        None,
    ],
)
def test_ordinary_speech_is_not_a_request_to_repeat(line):
    assert wants_repeat(line) is False


# --- the line spoken when there is no way back to the model ------------------------------------


def test_it_repeats_the_agents_own_last_words():
    assert say_again("It sits on 45 acres. Do you know Varthur?") == (
        "Sorry about that. It sits on 45 acres. Do you know Varthur?"
    )


def test_with_nothing_to_repeat_it_still_keeps_the_call_open():
    assert say_again(None) == "Sorry about that. Can you hear me now?"
    assert say_again("   ") == "Sorry about that. Can you hear me now?"


# --- the handler refuses to hang up ------------------------------------------------------------


def _handler_source() -> str:
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    return src[src.index("async def end_call_handler") : src.index("llm.register_function")]


def test_the_handler_checks_before_it_does_anything_else():
    """Before the closing line is built, before the booking check, before the log. Nothing
    about a hangup that is not going to happen belongs in the record."""
    handler = _handler_source()
    assert "wants_repeat(" in handler
    assert handler.index("wants_repeat(") < handler.index("closing_line(")
    assert handler.index("wants_repeat(") < handler.index("AGENT initiated call end")


def test_it_reads_only_the_prospects_last_line():
    """Their last line, not the whole call: they asked to repeat NOW. An earlier "pardon"
    would otherwise hold every later hangup open."""
    handler = _handler_source()
    assert "prospect_lines[-1] if prospect_lines else None" in handler
    assert 'm.get("role") == "user"' in handler


def test_the_turn_goes_back_to_the_model_rather_than_a_canned_line():
    """The model can rephrase; we cannot. say_again is the path taken only when the tool
    call arrives with no callback to answer through."""
    handler = _handler_source()
    assert "result_callback" in handler
    assert "REFUSAL_REASON" in handler
    assert handler.index("callback is not None") < handler.index("say_again(")


def test_the_reason_tells_the_model_what_to_do_instead():
    """A refusal with no instruction is a model that calls the tool again immediately."""
    assert "Do NOT end the call" in REFUSAL_REASON
    assert "say your last point again" in REFUSAL_REASON


def test_the_refusals_are_bounded():
    """A guard meant to save a lead must not become a call nobody can get off. After this
    the model's judgement wins."""
    handler = _handler_source()
    assert "_repeat_refusals < MAX_REPEAT_REFUSALS" in handler
    assert "_repeat_refusals += 1" in handler
    assert MAX_REPEAT_REFUSALS == 2


def test_the_counter_is_a_closure_variable_of_the_call():
    """Per call, not per process: a module-level counter would exhaust itself across calls
    and stop guarding the tenth prospect of the day."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "_repeat_refusals: int = 0" in src
    tree = ast.parse(src.lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "end_call_handler"
    )
    nonlocals = [n for n in ast.walk(handler) if isinstance(n, ast.Nonlocal)]
    assert any("_repeat_refusals" in n.names for n in nonlocals)


# --- driven end to end through the real handler ------------------------------------------------


def _drive(prospect_last_line, callback=None, refusals=0):
    """Compile the real handler with a context whose last user line is the one given."""
    from tests.test_closing_line import _Farewell, _Task, _build

    task = _Task()
    handler, spawned = _build(task, _Farewell())
    # _build's context stub carries a booking read-back; replace its last user line.
    handler.__globals__["context"] = type(
        "Ctx", (), {"messages": [{"role": "user", "content": prospect_last_line}]}
    )()
    params = type("P", (), {"arguments": {"closing_line": "Thank you. Have a good day."}})()
    if callback is not None:
        params.result_callback = callback

    async def run():
        await handler(params)
        for coro in spawned:
            await coro

    asyncio.run(run())
    return task


def test_a_repeat_request_does_not_end_the_call():
    task = _drive("Sorry, I did not catch that. Can you say it again?")
    queued = [type(f).__name__ for batch in task.batches for f in batch]
    assert "_Frame" not in queued, f"something was queued to end the call: {queued}"


def test_a_repeat_request_answers_through_the_callback_when_there_is_one():
    seen = {}

    async def callback(result):
        seen.update(result)

    _drive("Can you say it again?", callback=callback)
    assert seen == {"refused": REFUSAL_REASON}


def test_with_no_callback_the_last_line_is_spoken_again():
    task = _drive("Can you say it again?")
    said = " ".join(
        f.text for batch in task.batches for f in batch if getattr(f, "text", None)
    )
    assert said.startswith("Sorry about that.")


def test_an_ordinary_goodbye_still_ends_the_call():
    task = _drive("Yeah, that sounds good. Thank you.")
    assert task.batches, "the handler queued nothing at all"
    said = " ".join(
        f.text for batch in task.batches for f in batch if getattr(f, "text", None)
    )
    assert "Have a good day" in said
