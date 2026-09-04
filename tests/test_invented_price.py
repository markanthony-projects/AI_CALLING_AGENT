"""A figure the agent spoke that nobody gave it.

Live call c085e397, 4 Sep 2026. The prospect wanted North Bangalore; the project is in
Varthur and the campaign context carries pricing for nothing else:

    USER  -> "I am not sure of the budget. Can you tell me what is the possible budget
              I should have?"
    AGENT -> "That depends on the area and the size of the home you want. For a good
              2 or 3 BHK in North Bangalore, budgets usually start from 80 Lakhs and
              go up. When are you planning to buy?"

The prompt already said "Never invent facts, prices, sizes or locations that are not in
the campaign context" and the model did it anyway. So the prompt rule was made specific
about other areas, and this check was added underneath it — not to block the sentence,
which would leave "budgets usually start from and go up", but so the next one is a line
in the log instead of something only the prospect heard.
"""

import asyncio
import inspect

import pytest

from app.utils.money import amounts_in, as_spoken, ungrounded_amounts
from app.utils.spoken_text import ToolSyntaxFilter

# The shape build_campaign_context actually emits, for the project on that call.
CONTEXT = (
    "Project Name: Abhee Codename New Dimension\n"
    "Location: Varthur\n"
    "Overall Price Range: 1.17 Crores to 3.5 Crores\n"
    "Key Selling Points (USPs): Scotland-themed township, "
    "20 to 30 Lakhs below the launch price\n"
)


# --- reading figures out of speech ----------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("budgets usually start from 80 Lakhs and go up", [0.8]),
        ("Prices start at 1.17 Crores.", [1.17]),
        ("2 BHK homes from 1.2 Crores to 3.5 Crores", [1.2, 3.5]),
        # "20 to" carries no unit of its own; only the figure the unit is attached to
        # is money. Both halves of a range are still covered, because the context is
        # searched the same way and carries the same phrase.
        ("about 20 to 30 Lakhs below the launch price", [0.3]),
    ],
)
def test_money_is_read_in_crores_whatever_unit_it_was_said_in(text, expected):
    assert amounts_in(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "We have 2, 3 and 4 BHK homes.",
        "Around in the next 6 months.",
        "It is 15 minutes from the ORR.",
        "I am calling from Abhee Codename New Dimension.",
    ],
)
def test_a_number_without_a_money_unit_is_not_a_price(text):
    """Configurations, timelines and distances are numbers too. Flagging those would make
    the warning noise, and a warning that fires on every call is one nobody reads."""
    assert amounts_in(text) == []


def test_the_units_a_model_was_told_not_to_write_are_still_read():
    """The prompt forbids "1.2 Cr". This exists for the turns where it was not followed."""
    assert amounts_in("from 1.2 Cr") == pytest.approx([1.2])
    assert amounts_in("around 80 lacs") == pytest.approx([0.8])


# --- grounding ------------------------------------------------------------------------


def test_the_figure_from_the_live_call_is_caught():
    spoken = (
        "For a good 2 or 3 BHK in North Bangalore, budgets usually start from 80 Lakhs "
        "and go up."
    )
    assert ungrounded_amounts(spoken, CONTEXT) == pytest.approx([0.8])


@pytest.mark.parametrize("line", ["Prices start at 1.17 Crores.", "up to 3.5 Crores"])
def test_a_figure_the_context_carries_passes(line):
    assert ungrounded_amounts(line, CONTEXT) == []


def test_rounding_the_price_is_not_inventing_it():
    """The agent is told to say 1.17 Crores. A model that says 1.2 has quoted the same
    home, and flagging that would bury the real case in false positives."""
    assert ungrounded_amounts("Prices start at 1.2 Crores.", CONTEXT) == []


def test_the_price_benefit_in_the_usps_grounds_itself():
    """The USP line carries "20 to 30 Lakhs below the launch price", so saying it is not
    an invention — even though 0.2 Crores is nowhere near the price of a home here."""
    line = "That is about 20 to 30 Lakhs below the launch price."
    assert ungrounded_amounts(line, CONTEXT) == []


def test_grounding_is_membership_and_not_a_range():
    """The context spans 0.2 to 3.5 Crores once the USP line is counted, so a range test
    would have let the 80 Lakhs through. This is why the check compares against each
    figure rather than the span of them."""
    assert 0.2 < 0.8 < 3.5
    assert ungrounded_amounts("start from 80 Lakhs", CONTEXT) == pytest.approx([0.8])


def test_a_campaign_with_no_prices_grounds_nothing():
    """An agent quoting prices for a project it was given none for is inventing every one
    of them, and a campaign configured that way is the case worth hearing about."""
    bare = "Project Name: X\nLocation: Y"
    assert ungrounded_amounts("Prices start at 1.17 Crores.", bare) == pytest.approx([1.17])


def test_a_sentence_with_no_figures_never_looks_at_the_context():
    assert ungrounded_amounts("Are you looking for any property purchase?", "") == []


@pytest.mark.parametrize(
    "value,written",
    [(0.8, "80 Lakhs"), (1.0, "1 Crores"), (1.17, "1.17 Crores"), (3.5, "3.5 Crores")],
)
def test_the_log_says_the_figure_the_way_the_agent_said_it(value, written):
    assert as_spoken(value) == written


# --- the processor ---------------------------------------------------------------------


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _run(filter_, chunks):
    import pipecat.processors.frame_processor as fp
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection

    captured = _Captured()
    filter_.push_frame = captured.push
    filter_.process_frame = filter_.__class__.process_frame.__get__(filter_)

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        await filter_.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for chunk in chunks:
            await filter_.process_frame(LLMTextFrame(chunk), FrameDirection.DOWNSTREAM)
        await filter_.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


def _spoken(frames):
    return "".join(getattr(f, "text", "") for f in frames)


def _errors_while(filt, *responses):
    """What the agent said, and the price warnings loguru emitted while it said it.

    loguru does not write through the stdlib logging module, so caplog sees nothing. A
    sink is the only thing that observes these lines the way production does.
    """
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="ERROR")
    try:
        spoken = "".join(_spoken(asyncio.run(_run(filt, chunks))) for chunks in responses)
    finally:
        logger.remove(sink)
    return spoken, [line for line in seen if "not in the campaign context" in line]


def test_the_sentence_still_reaches_the_caller():
    """Reports rather than strips. "budgets usually start from and go up" is worse than
    the invention, and the fix belongs in the prompt."""
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    frames = asyncio.run(_run(filt, ["For a 3 BHK, budgets start from 80 Lakhs and go up."]))
    assert "80 Lakhs" in _spoken(frames)


def test_an_invented_price_is_logged():
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    _, errors = _errors_while(filt, ["Budgets start from 80 Lakhs in North Bangalore."])
    assert len(errors) == 1
    assert "80 Lakhs" in errors[0]


def test_a_grounded_price_is_not_logged():
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    _, errors = _errors_while(filt, ["We have 3 BHK homes. Prices start at 1.17 Crores."])
    assert errors == []


def test_a_figure_split_across_streaming_chunks_is_still_seen():
    """The model streams tokens, so "80 Lakhs" can arrive as "from 80" then " Lakhs ".
    This processor forwards each chunk as it comes rather than reassembling sentences, so a
    check that looked at the chunk would have missed the very line that prompted it."""
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    _, errors = _errors_while(filt, ["Budgets start ", "from 80", " Lakhs ", "and go up."])
    assert len(errors) == 1
    assert "80 Lakhs" in errors[0]


def test_it_reports_once_per_response_however_many_figures():
    """A model that has started inventing prices tends to keep going in the same breath.
    One line per response is enough to count them by."""
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    _, errors = _errors_while(filt, ["Budgets start from 80 Lakhs. ", "Some go to 90 Lakhs. "])
    assert len(errors) == 1


def test_the_next_response_can_report_again():
    """The flag is per response, not per call. Otherwise the first invention would mask
    every later one and the count would always be one."""
    filt = ToolSyntaxFilter("sid", campaign_context=CONTEXT)
    _, errors = _errors_while(
        filt, ["Budgets start from 80 Lakhs."], ["Others are around 95 Lakhs."]
    )
    assert len(errors) == 2


def test_a_filter_built_without_a_context_does_not_crash_the_call():
    """Every other test in the suite constructs ToolSyntaxFilter without one. A default
    that raised would take out the pipeline rather than the price check."""
    filt = ToolSyntaxFilter("sid")
    frames = asyncio.run(_run(filt, ["Prices start at 1.17 Crores."]))
    assert "1.17 Crores" in _spoken(frames)


def test_the_agent_hands_the_filter_the_same_context_the_model_was_given():
    """A check run against a different context than the prompt would flag the model for
    following its instructions."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "campaign_context=campaign_context" in src
    assert "get_system_prompt(campaign_context" in src


# --- and the rule the model was given ---------------------------------------------------


def test_the_prompt_names_the_other_area_case():
    """The general "never invent" rule was already there and was not enough. What was
    missing was the specific situation: a prospect asking about a market we have nothing
    on, where the honest answer is that we do not know it."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt(CONTEXT, "Rahul")
    assert "OTHER areas" in prompt
    assert "property expert will share exact options" in prompt
