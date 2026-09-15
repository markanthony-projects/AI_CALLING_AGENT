"""A yes is not a goodbye; end_call on one is refused once and the turn goes back.

Call 96080daa, 15 Sep 2026: "Yeah. I was looking for property purchase." → end_call.
"""

import inspect

import pytest

from app.utils.engaged_answer import (
    MAX_EARLY_REFUSALS,
    REFUSAL_REASON,
    said_yes_and_nothing_was_closed,
)

OPENING = "We are launching a new project in Varthur. Are you looking to buy a property?"


@pytest.mark.parametrize(
    "said",
    [
        "Yeah. I was looking for property purchase.",
        "Yeah, I was looking to buy a property.",
        "Yes, tell me more.",
        "Haan ji, batao.",
        "Sure, go ahead.",
        "I am interested, which area?",
        "Ok.",
    ],
)
def test_a_yes_to_a_question_that_closed_nothing_holds_the_call(said):
    assert said_yes_and_nothing_was_closed(said, OPENING) is True


@pytest.mark.parametrize(
    "said",
    [
        "No, not interested.",
        "Yeah, thank you, bye.",
        "I am busy right now.",
        "Yes but call me later.",
        "Don't call me again.",
        "No.",
        "",
        None,
    ],
)
def test_a_refusal_or_a_no_is_not_held(said):
    """A no is bare_answer's business; a refusal in any number of words ends the call."""
    assert said_yes_and_nothing_was_closed(said, OPENING) is False


@pytest.mark.parametrize(
    "offer",
    [
        "Shall I send you the floor plans and prices on WhatsApp?",
        "Should our property expert call you with the details?",
        "Would you like to visit the site on Saturday at 11 AM?",
        "Is there anything else I can help you with?",
    ],
)
def test_a_yes_to_a_close_is_the_close(offer):
    assert said_yes_and_nothing_was_closed("Yes, sure.", offer) is False


def test_with_no_agent_line_on_record_a_yes_still_holds():
    assert said_yes_and_nothing_was_closed("Yes.", None) is True
    assert said_yes_and_nothing_was_closed("Yes.", "") is True


def test_the_refusal_is_bounded_to_one():
    assert MAX_EARLY_REFUSALS == 1


def test_the_reason_tells_the_model_what_to_do_instead():
    assert "Do NOT end the call" in REFUSAL_REASON
    assert "carry on with the next step" in REFUSAL_REASON


def test_the_handler_consults_it_after_the_other_refusals_and_before_the_goodbye():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def end_call_handler") :]
    handler = handler[: handler.index('llm.register_function("end_call"')]
    yes = handler.index("said_yes_and_nothing_was_closed(")
    assert handler.index("is_a_bare_answer(") < yes
    assert handler.index("wants_repeat(") < yes
    assert yes < handler.index("line = closing_line(")
    assert "callback({\"refused\": EARLY_REFUSAL_REASON})" in handler
    assert "_early_refusals < MAX_EARLY_REFUSALS" in handler


def test_the_refusal_cue_is_shared_with_the_bare_answer_guard():
    from app.utils import bare_answer, engaged_answer

    assert engaged_answer.is_a_refusal is bare_answer.is_a_refusal
    assert bare_answer.is_a_refusal("not interested") and not bare_answer.is_a_refusal("yes please")


# --- a yes to the close is the close ----------------------------------------------------


def test_a_one_word_yes_to_the_whatsapp_offer_ends_the_call():
    """Call be096321: "Yes." to the brochure offer was refused as a bare answer."""
    from app.utils.engaged_answer import is_a_yes_to_a_close

    offer = "Shall I send you the 3 BHK Regular floor plan and pricing on WhatsApp?"
    assert is_a_yes_to_a_close("Yes.", offer) is True
    assert is_a_yes_to_a_close("Haan.", offer) is True
    assert is_a_yes_to_a_close("No.", offer) is False
    assert is_a_yes_to_a_close("Yes.", "Is Varthur convenient for you?") is False
    assert is_a_yes_to_a_close("", offer) is False


def test_the_bare_answer_refusal_steps_aside_for_it():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def end_call_handler") :]
    handler = handler[: handler.index('llm.register_function("end_call"')]
    bare = handler.index("is_a_bare_answer(")
    assert "and not is_a_yes_to_a_close(" in handler[bare : bare + 300]
