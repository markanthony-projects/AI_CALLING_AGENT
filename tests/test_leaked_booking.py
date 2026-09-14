"""A prospect in the middle of booking a visit is asked for the time, not hung up on.

Call 6102ee87, 14 Sep 2026, turn 13. The prospect said "Are you free on next weekend, like
Saturday" — a day, no hour — and the model wrote its tool call as text:

    {"name": "end_call", "arguments": {"closing_line":
        "Your site visit is confirmed for Saturday at 11 AM. ..."}}

Nobody had said 11 AM. The filter caught the markup, then read the words in front of it —
"...Thank you for your time." — as a goodbye already on the wire. But OneQuestionPerTurn,
downstream, had cut everything after the first question; those words were never spoken.
The handler then waited twelve seconds for a farewell nobody had queued, the prospect said
"Hello? Hello?" into the silence, and the line went dead. Three faults, one ending:

  1. the leak path judged words the cutter dropped              -> lead_in is cut the same way
  2. the leaked closing line skipped the booking check           -> it gets the tool path's
  3. a leaked end_call with no agreed hour still ended the call  -> it asks for the hour

and the wait was for a goodbye this branch never queues        -> wait_for_quiet instead.
"""

import ast
import asyncio
import inspect

import pytest

from app.utils.booking_claim import ASK_FOR_TIME, MAX_BOOKING_REFUSALS, REFUSAL_REASON
from app.utils.one_question import spoken_part
from app.utils.spoken_text import ToolSyntaxFilter, sounds_like_goodbye

# --- the rule, as a function --------------------------------------------------------------


@pytest.mark.parametrize(
    "written, heard",
    [
        ("Which day works for you?", "Which day works for you?"),
        ("Great, Saturday at 11 AM works?Perfect, so Saturday. Thank you for your time.",
         "Great, Saturday at 11 AM works?"),
        ("Got it. Tuesday at 8 PM. I will book that for you.",
         "Got it. Tuesday at 8 PM. I will book that for you."),
        ("", ""),
    ],
)
def test_what_the_cutter_lets_through_is_computable_upstream(written, heard):
    assert spoken_part(written) == heard


# --- the filter reports what will be heard ---------------------------------------------------


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
    return captured


# The live reply, as it streamed: a question, a run-on the cutter drops, then the JSON.
LIVE = [
    "Great, Saturday at 11 AM works?",
    "Perfect, so Saturday at 11 AM for the site visit. ",
    "I will send you the details on WhatsApp. Thank you for your time.",
    '{"name": "end_call", "arguments": {"closing_line": '
    '"Your site visit is confirmed for Saturday at 11 AM. We will send the details on WhatsApp. '
    'Thank you for your time."}}',
]


def test_the_handler_is_told_only_the_words_that_will_be_heard():
    seen = {}

    async def on_leak(line, spoken_already):
        seen["line"], seen["spoken"] = line, spoken_already

    asyncio.run(_run(ToolSyntaxFilter("sid", on_leaked_end_call=on_leak), LIVE))
    assert seen["spoken"] == "Great, Saturday at 11 AM works?"
    assert not sounds_like_goodbye(seen["spoken"]), "the goodbye after the cut was never spoken"
    assert seen["line"].startswith("Your site visit is confirmed for Saturday at 11 AM")


def test_lead_in_stops_at_the_first_question():
    filt = ToolSyntaxFilter("sid")
    asyncio.run(_run(filt, LIVE[:3]))
    assert filt.lead_in == "Great, Saturday at 11 AM works?"


def test_a_price_after_the_cut_is_not_reported_as_spoken(capsys):
    """Turn 5 of the same call logged "Agent spoke a price that is not in the campaign
    context (1.64 Crores)" for a figure inside a runaway the cutter dropped. The guard
    reads the words on the wire now, so an invented price the prospect never heard is not
    a walk-back for the team."""
    from loguru import logger

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="ERROR", format="{message}")
    filt = ToolSyntaxFilter("sid", campaign_context="- 2 BHK: 1200 sqft, Price: 1.17 Cr")
    asyncio.run(_run(filt, ["Which budget are you thinking of?", "We also have 3 BHK at 1.64 Crores."]))
    assert "1.64" not in capsys.readouterr().out


def test_a_price_before_the_cut_is_still_reported(capsys):
    from loguru import logger

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="ERROR", format="{message}")
    filt = ToolSyntaxFilter("sid", campaign_context="- 2 BHK: 1200 sqft, Price: 1.17 Cr")
    asyncio.run(_run(filt, ["We have 3 BHK at 1.64 Crores. ", "Does that work?"]))
    assert "1.64" in capsys.readouterr().out


# --- what the handlers do with it ---------------------------------------------------------------


def _handler(name):
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name
    )
    return ast.unparse(node)


def test_the_leak_path_vets_the_booking_the_way_the_tool_path_does():
    src = _handler("on_leaked_end_call")
    assert "unagreed_booking(" in src
    assert "closing_line(line)" in src
    # And it decides before the call is marked as ending, so a refusal leaves it open.
    assert src.index("unagreed_booking(") < src.index("_ending = True")


def test_a_leak_with_no_agreed_hour_asks_for_it_instead_of_hanging_up():
    src = _handler("on_leaked_end_call")
    assert "_booking_refusals < MAX_BOOKING_REFUSALS" in src
    assert "spoken(ASK_FOR_TIME)" in src
    # Only when the words on the wire did not already ask a question.
    assert "endswith('?')" in src


def test_the_tool_path_refuses_once_then_falls_back_to_the_plain_farewell():
    src = _handler("end_call_handler")
    refusal = src.index("_booking_refusals < MAX_BOOKING_REFUSALS")
    assert refusal < src.index("line = FAREWELL_LINE")
    assert "callback({'refused': BOOKING_REFUSAL_REASON})" in src
    assert MAX_BOOKING_REFUSALS == 1


def test_the_leak_path_waits_only_for_what_is_actually_playing():
    """wait_until_spoken waits for a farewell; this branch queues none. The old code sat its
    full ceiling out — twelve seconds on the live call — for audio that was never coming."""
    src = _handler("on_leaked_end_call")
    assert "wait_for_quiet(" in src
    assert "wait_until_spoken(" not in src
    assert "farewell.arm()" not in src


def test_the_reason_and_the_question_say_what_is_missing():
    assert "day and what time" in REFUSAL_REASON
    assert "Do NOT end the call" in REFUSAL_REASON
    assert ASK_FOR_TIME.endswith("?")
    assert "which day and what time" in ASK_FOR_TIME.lower()


def test_the_prompt_says_a_day_alone_is_not_a_booking():
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "A DAY WITHOUT A CLOCK TIME IS NOT A BOOKING" in prompt
    assert "next weekend, like Saturday" in prompt
