"""One word is an answer. It is never a decision to end the call.

Call 578195d1, 14 Sep 2026. Forty-five seconds, and the prospect was warm:

    USER  → "Yeah, I was looking to buy a property."
    AGENT → "It is called Abhee Codename New Dimension. … Is Varthur convenient for you?"
    USER  → "Yeah."
    AGENT → "It will have 14 towers spread over 45 acres. Have you been to that side of town?"
    USER  → "No."
    AGENT → "Thank you for your time, Rahul." → end_call

They answered a question about geography. They had already said they were buying and
already said the area suited them. The model read a one-word answer as the end of the
conversation and hung up on a qualified lead.

Both halves of that were mine: "If no -> step 5" sits in the opening gate, and every easy
question in step 3 is a yes/no whose "no" means nothing. So the guard is code, and the
prompt says which "no" routes where.
"""

import ast
import inspect
from pathlib import Path

import pytest

from app.utils.bare_answer import (
    ENOUGH_TO_BE_A_DECISION,
    MAX_BARE_REFUSALS,
    REFUSAL_REASON,
    is_a_bare_answer,
)

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "said", ["No.", "No", "Yeah.", "Yes", "yeah", "Haan", "nahi", "Ok", "Yeah yeah", "हाँ"]
)
def test_a_bare_answer_does_not_end_a_call(said):
    assert is_a_bare_answer(said) is True


@pytest.mark.parametrize(
    "said",
    [
        # Short, and still unmistakably somebody leaving. A refusal is a refusal at any
        # length, which is why the words are checked before the word count.
        "No, thank you.",
        "no thanks",
        "Not interested",
        "I am not interested.",
        "nahi chahiye",
        "Call me later",
        "I am busy",
        # And anything long enough to be a decision.
        "I said I'm not interested in this project.",
        "No, actually I was not interested in buying property. Thank you.",
        "Please do not call me again",
    ],
)
def test_a_real_goodbye_still_ends_the_call(said):
    assert is_a_bare_answer(said) is False


@pytest.mark.parametrize("nothing", [None, "", "   ", "...", "?"])
def test_nothing_said_is_not_an_answer_either(nothing):
    """An empty turn is a transcript of silence. The dead-air path owns that, not this."""
    assert is_a_bare_answer(nothing) is False


def test_the_threshold_is_small_enough_to_be_about_bare_answers():
    """Three words is "no thank you". Any higher and this starts holding calls open on
    people who have plainly said they are done."""
    assert ENOUGH_TO_BE_A_DECISION == 3


def test_the_refusal_is_bounded():
    """A guard written to save a lead must not become a call nobody can get off."""
    assert 1 <= MAX_BARE_REFUSALS <= 3


def test_the_reason_tells_the_model_what_to_do_instead():
    """A refusal with no instruction leaves the model with nothing to say next, and a turn
    with nothing to say is where it reaches for the goodbye again."""
    assert "Do NOT end the call" in REFUSAL_REASON
    assert "ask your next question" in REFUSAL_REASON


# --- wired into the one place that can hang up ---------------------------------------------


def _handler() -> str:
    tree = ast.parse(AGENT_SRC)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "end_call_handler"
    )
    return ast.unparse(node)


def test_end_call_consults_it():
    src = _handler()
    assert "is_a_bare_answer" in src
    assert "_bare_refusals < MAX_BARE_REFUSALS" in src
    assert "_bare_refusals += 1" in src


def test_it_is_checked_before_the_call_can_end():
    """After the guard, not before it: a check that runs once the goodbye is queued is not a
    guard, it is a log line."""
    src = _handler()
    assert src.index("is_a_bare_answer") < src.index("say_goodbye_then_hang_up")


def test_the_turn_goes_back_to_the_model_rather_than_a_canned_line():
    """The same shape as the repeat refusal: the model carries on in its own words. A fixed
    sentence from us would be the second thing on the call that sounds like a recording."""
    src = _handler()
    assert "BARE_ANSWER_REASON" in src
    assert "result_callback" in src


def test_there_is_a_fallback_when_there_is_no_callback():
    """Some tool-call paths arrive with no way to hand the turn back. Asking the question
    again is worse than the model carrying on, and far better than hanging up on somebody
    who is still buying."""
    src = _handler()
    after = src[src.index("BARE_ANSWER_REASON") :]
    assert "dead_air_nudge" in after


def test_the_counter_is_a_closure_of_the_call():
    """Per call, not per process — a module-level counter would carry one call's refusals
    into the next."""
    assert "_bare_refusals: int = 0" in AGENT_SRC
    tree = ast.parse(AGENT_SRC)
    nonlocals = [n for n in ast.walk(tree) if isinstance(n, ast.Nonlocal)]
    assert any("_bare_refusals" in n.names for n in nonlocals)


def test_the_prompt_says_which_no_routes_to_step_five():
    """The code guard is the reliable half. This is the half that stops the model wanting to
    hang up in the first place."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert 'ONLY THIS "no" ROUTES TO STEP 5' in prompt
    assert "A one-word answer is never a decision about the call." in prompt
