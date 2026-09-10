"""A booking has to be something the prospect said, not something the agent announced.

Live call 2eeb48a0, 10 Sep 2026. No site visit was offered at any point. The prospect's last
words were "I said it is for investment", and the model ended the call with:

    AGENT -> "Your visit is confirmed for Saturday at 11 AM at Abhee Codename New Dimension."

Then the extraction worker read that sentence — the agent's own — as the booking:

    Site visit resolved to Saturday 2026-09-12 11:00

The budget and the location have had this check since 2 Sep: "is there a Prospect line this
value could have come from". The appointment fields did not, and an invented appointment is
the dearest of the three: a colleague waits at the site for someone who was never asked.
"""

import inspect

import pytest

from app.models.schemas import Weekday
from app.models.schemas import LeadExtraction
from app.utils.attribution import day_is_grounded, time_is_grounded

# The live transcript, abridged to the lines that matter. The prospect never names a day.
LIVE = (
    "Agent: Is this purchase for your own stay or for investment?\n"
    "Prospect: This was for this month.\n"
    "Agent: Would a 2 BHK be suitable for you?\n"
    "Prospect: Does it does it allow with my budget?\n"
    "Prospect: I said it is for investment\n"
    "Agent: Thank you, Rahul. Your visit is confirmed for Saturday at 11 AM at Abhee Codename New Dimension. Have a great day.\n"
)

AGREED = (
    "Agent: Would Saturday work for a visit?\n"
    "Prospect: Yeah Saturday is fine, around 11 in the morning.\n"
    "Agent: Perfect, Saturday at 11 AM.\n"
)


# --- the day ------------------------------------------------------------------------------


def test_the_live_calls_saturday_is_not_grounded():
    assert day_is_grounded(Weekday.SATURDAY, None, LIVE) is False


def test_a_day_the_prospect_said_is():
    assert day_is_grounded(Weekday.SATURDAY, None, AGREED) is True


@pytest.mark.parametrize("word", ["sat", "Saturday", "SATURDAY"])
def test_short_and_long_forms_and_case(word):
    assert day_is_grounded("SATURDAY", None, f"Prospect: {word} works")


def test_a_day_only_the_agent_said_is_not():
    """"Would Saturday work?" is the agent's line. Only Prospect: lines count, as everywhere
    in this module."""
    assert day_is_grounded(Weekday.SATURDAY, None, "Agent: Would Saturday work?\nProspect: hmm") is False


@pytest.mark.parametrize(
    "in_days,said",
    [(0, "today"), (0, "this evening"), (1, "tomorrow"), (1, "kal"), (2, "day after")],
)
def test_relative_days_are_grounded_by_the_words_that_produce_them(in_days, said):
    assert day_is_grounded(None, in_days, f"Prospect: {said} is fine") is True


def test_a_relative_day_the_prospect_never_said_is_not():
    assert day_is_grounded(None, 1, "Prospect: I will think about it") is False


def test_no_day_at_all_is_left_alone():
    """Nothing to check. The resolver returns None for it anyway."""
    assert day_is_grounded(None, None, LIVE) is True


# --- the hour -----------------------------------------------------------------------------


@pytest.mark.parametrize("said", ["11 AM", "11", "eleven", "11:30", "around eleven o'clock"])
def test_the_hour_in_figures_or_words(said):
    assert time_is_grounded("11:00", f"Prospect: {said}") is True


def test_the_hour_the_agent_proposed_is_not_grounded():
    assert time_is_grounded("11:00", LIVE) is False


def test_twenty_four_hour_times_match_the_twelve_hour_words():
    assert time_is_grounded("15:00", "Prospect: three is fine") is True
    assert time_is_grounded("15:00", "Prospect: 3 PM") is True


def test_a_digit_inside_a_price_does_not_ground_an_hour():
    """"1.17 Cr" contains a 1. It is not one o'clock."""
    assert time_is_grounded("13:00", "Prospect: my budget is 1.17 Cr") is False


def test_no_time_is_left_alone():
    assert time_is_grounded(None, LIVE) is True
    assert time_is_grounded("", LIVE) is True


# --- the worker applies it ----------------------------------------------------------------


def _lead(**fields):
    return LeadExtraction(is_prospect=True, **fields)


def _drop(lead, transcript):
    from app.worker import _drop_unagreed_appointments

    return _drop_unagreed_appointments(lead, transcript, "sid")


def test_the_live_calls_booking_is_removed_entirely():
    out = _drop(_lead(site_visit_weekday=Weekday.SATURDAY, site_visit_at="11:00"), LIVE)
    assert out.site_visit_weekday is None
    assert out.site_visit_at is None


def test_an_agreed_booking_survives():
    out = _drop(_lead(site_visit_weekday=Weekday.SATURDAY, site_visit_at="11:00"), AGREED)
    assert out.site_visit_weekday == Weekday.SATURDAY
    assert out.site_visit_at == "11:00"


def test_a_day_agreed_but_an_hour_only_the_agent_named_loses_the_hour():
    """The day stands, the hour goes, and resolve_appointment then makes no booking of it —
    which is right: a day without an agreed time was never an appointment."""
    transcript = "Agent: Sunday?\nProspect: Sunday is fine.\nAgent: Shall we say 3 PM then? I will book 3 PM.\n"
    out = _drop(_lead(site_visit_weekday=Weekday.SUNDAY, site_visit_at="15:00"), transcript)
    assert out.site_visit_weekday == Weekday.SUNDAY
    assert out.site_visit_at is None


def test_callbacks_are_held_to_the_same_rule():
    out = _drop(_lead(callback_weekday=Weekday.MONDAY, callback_at="18:00"), LIVE)
    assert out.callback_weekday is None
    assert out.callback_at is None


def test_the_other_fields_are_untouched():
    out = _drop(_lead(site_visit_weekday=Weekday.SATURDAY, site_visit_at="11:00", budget=14000000.0), LIVE)
    assert out.budget == 14000000.0


def test_a_lead_with_nothing_to_drop_is_returned_unchanged():
    lead = _lead(budget=14000000.0)
    assert _drop(lead, LIVE) is lead


def test_it_runs_after_the_other_grounding_and_before_the_lead_is_built():
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert src.index("_drop_ungrounded(lead_data") < src.index("_drop_unagreed_appointments(lead_data")
    assert src.index("_drop_unagreed_appointments(lead_data") < src.index("lead = Lead(")


def test_it_reads_the_same_text_as_the_other_checks():
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert "_drop_unagreed_appointments(lead_data, grounding_text, call_sid)" in src


# --- and the model is told, beside the call that showed why ------------------------------


def test_the_prompt_forbids_announcing_a_booking_nobody_agreed_to():
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "NEVER announce a visit or a callback the prospect did not agree to" in prompt
    assert "Your visit is confirmed for Saturday at 11 AM" in prompt
    assert "a promise the prospect made, never one you are making for them" in prompt
