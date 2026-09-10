"""The goodbye may not announce an appointment the prospect never made.

Live call 2eeb48a0, 10 Sep 2026: "Your visit is confirmed for Saturday at 11 AM" to a
prospect whose last words were "I said it is for investment". The extractor's side is in
test_booking_attribution.py; this is the sentence the prospect actually hears.
"""

import inspect

import pytest

from app.utils.booking_claim import unagreed_booking

LIVE_LINE = "Thank you, Rahul. Your visit is confirmed for Saturday at 11 AM at Abhee Codename New Dimension. Have a great day."
LIVE_PROSPECT = [
    "Regarding what ma'am?",
    "Yeah, I was looking for a property.",
    "Yeah, I had a budget around 1.4 CR",
    "This was for this month.",
    "I said it is for investment",
]


def test_the_live_calls_claim_is_caught():
    assert unagreed_booking(LIVE_LINE, LIVE_PROSPECT) == "Saturday"


def test_a_read_back_of_what_they_said_passes():
    assert unagreed_booking("Perfect, Sunday at 3 PM. Thank you!", ["Sunday at 3 PM works"]) is None


def test_the_hour_is_checked_even_when_the_day_was_agreed():
    """"Sunday" they said. "3 PM" the agent picked. The hour is the claim that gets caught."""
    assert unagreed_booking("Perfect, Sunday at 3 PM.", ["Sunday is fine"]) == "3 PM"


@pytest.mark.parametrize("said", ["3 PM", "3pm", "three in the afternoon", "15:00"])
def test_the_hour_in_any_form_they_used(said):
    assert unagreed_booking("Sunday at 3 PM then.", [f"Sunday, {said}"]) is None


def test_tomorrow_is_a_claim_too():
    assert unagreed_booking("See you tomorrow at 5 PM.", ["okay call me"]) == "tomorrow"
    assert unagreed_booking("See you tomorrow at 5 PM.", ["tomorrow at 5 is fine"]) is None


def test_a_bare_number_is_not_a_time_claim():
    """"2 BHK" and "3 acres" are not appointments."""
    assert unagreed_booking("Thank you for your interest in the 2 BHK. Have a great day.", ["ok"]) is None


def test_a_plain_thank_you_has_nothing_to_catch():
    assert unagreed_booking("Thank you for your time, Rahul. Have a great day.", LIVE_PROSPECT) is None
    assert unagreed_booking("", LIVE_PROSPECT) is None
    assert unagreed_booking(None, LIVE_PROSPECT) is None


def test_only_the_prospects_lines_count():
    """The agent's own "Would Saturday work?" is not the prospect agreeing to Saturday. The
    caller passes user messages only; this pins that the helper reads nothing else."""
    assert unagreed_booking("Saturday at 11 AM.", []) == "Saturday"


# --- the handler uses it ----------------------------------------------------------------------


def test_end_call_replaces_an_unagreed_claim_with_the_plain_farewell():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def end_call_handler") : src.index("llm.register_function")]
    assert "unagreed_booking(" in handler
    assert "line = FAREWELL_LINE" in handler
    assert 'm.get("role") == "user"' in handler, "the check must read the prospect's lines only"


def test_the_check_runs_before_the_goodbye_is_logged_or_queued():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def end_call_handler") : src.index("llm.register_function")]
    assert handler.index("unagreed_booking(") < handler.index("AGENT initiated call end")
    assert handler.index("unagreed_booking(") < handler.index("say_goodbye_then_hang_up(line)")


def test_the_context_the_handler_reads_exists_before_anything_can_call_it():
    """end_call_handler closes over `context`, which is assigned seventy lines below its own
    definition. That is legal — Python resolves a closure when the function runs — and it
    holds only because nothing can call the handler until the pipeline is up, which is after
    the assignment. A reorder that moved the pipeline above it would raise NameError on a
    live call, in the handler that hangs up, with the prospect on the line.

    Pinned as source order rather than trusted: the failure has no other warning."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert src.index("context = LLMContext(") < src.index("PipelineWorker(")
    assert src.index("context = LLMContext(") < src.index("Pipeline([")
