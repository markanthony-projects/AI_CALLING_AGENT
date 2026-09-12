"""The lines the system speaks itself reach the voice engine one sentence at a time.

Live call, 10 Sep 2026. Every model-generated reply sounded fine; the opening line — "Hi,
Good afternoon Rahul. I am Priya calling you from Abhee Ventures. Can I speak to you for a
minute?" — was reported as sounding like a machine, flat and slow. Same voice, same pace,
same engine. The difference is the route.

Pipecat cuts the model's text at sentence boundaries and synthesises each sentence as its
own request; the full stops are real gaps and the next sentence is already being made
while the last one plays. A TTSSpeakFrame skips that cut. The greeting went to Sarvam as
one 110-character request, three sentences in one breath. So did the goodbye, the nudges
and every recovery line.

spoken() gives those lines the model's route: one frame per sentence.
"""

import ast
import inspect

import pytest
from pipecat.frames.frames import TTSSpeakFrame

from app.services.agent import build_opening_line, spoken
from app.utils.sentences import sentences

from datetime import datetime

from app.utils.timeutils import time_of_day_greeting

NOW = datetime(2026, 9, 10, 8, 30)  # whatever part of the day this is, it is the same below
GREETING = build_opening_line(
    "Abhee Codename New Dimension", "Rahul Sharma", now=NOW, developer_name="Abhee Ventures"
)


# --- the cut ------------------------------------------------------------------------------


def test_the_opening_line_is_three_sentences():
    assert sentences(GREETING) == [
        f"Hello, Good {time_of_day_greeting(NOW)}.",
        "My name is Priya, and I am calling you from Abhee Ventures.",
        "Am I speaking with Rahul?",
    ]


def test_a_salutation_does_not_end_a_sentence():
    """spoken_name puts "Mr." and "Dr." in front of a name. "Good afternoon Mr." followed by
    "Rahul." is two frames where there was one sentence — the exact fault this fixes, in
    reverse."""
    # The name lives in the closing question now, so that is the sentence that must not be
    # split by the full stop inside "Mr." — the same fault, in the same place it can happen.
    line = build_opening_line("X", "Mr. Rahul Sharma", developer_name="Abhee Ventures")
    assert sentences(line)[-1] == "Am I speaking with Mr. Rahul?"
    line = build_opening_line("X", "Dr Sunita", developer_name="Abhee Ventures")
    assert sentences(line)[-1] == "Am I speaking with Dr. Sunita?"


def test_a_company_suffix_does_not_end_a_sentence():
    line = build_opening_line("X", "Rahul", developer_name="Prestige Pvt. Ltd.")
    assert "My name is Priya, and I am calling you from Prestige Pvt. Ltd." in sentences(line)


def test_a_decimal_does_not_end_a_sentence():
    assert sentences("It starts at 2.5 Crores. Shall I go on?") == [
        "It starts at 2.5 Crores.",
        "Shall I go on?",
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sorry, I missed that. Could you say it once more?", ["Sorry, I missed that.", "Could you say it once more?"]),
        ("Thank you for your time, Rahul!", ["Thank you for your time, Rahul!"]),
        ("Are you there? I am Priya. Hello! Can you hear me?", ["Are you there?", "I am Priya.", "Hello!", "Can you hear me?"]),
        ("One.  Two.   Three.", ["One.", "Two.", "Three."]),
        ("  Trailing space. ", ["Trailing space."]),
        ("", []),
        (None, []),
    ],
)
def test_ordinary_lines(text, expected):
    assert sentences(text) == expected


# --- the frames --------------------------------------------------------------------------


def test_spoken_queues_one_frame_per_sentence():
    frames = spoken(GREETING)
    assert [type(f) for f in frames] == [TTSSpeakFrame] * 3
    assert [f.text for f in frames] == sentences(GREETING)


def test_the_greeting_is_not_appended_to_the_context_by_the_engine():
    """run_voice_agent adds the whole line to the context by hand. Letting the engine append
    each sentence too would put the greeting there twice, in three pieces."""
    for frame in spoken(GREETING, append_to_context=False):
        assert frame.append_to_context is False


def test_other_lines_keep_the_default():
    assert all(f.append_to_context for f in spoken("Sorry, I missed that. Once more?"))


# --- every line the system speaks takes this route -----------------------------------------


def _speak_calls():
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("queue_frames")
    ]


def test_no_system_line_is_queued_as_a_single_frame():
    """A bare TTSSpeakFrame(...) inside run_voice_agent is a line going to the engine in one
    breath. The greeting was one; so were the nudge, the goodbye and four recovery lines."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "TTSSpeakFrame(" not in src, "a system line is bypassing spoken()"


def test_the_greeting_goes_through_spoken_without_context_append():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "queue_frames(spoken(opening_line, append_to_context=False))" in src


def test_sign_offs_still_end_the_call_after_the_last_sentence():
    """The sentences are unpacked ahead of EndWorkerFrame, so the hangup follows the last
    of them rather than the first."""
    spoken_signoffs = [
        ast.unparse(c) for c in _speak_calls()
        if "EndWorkerFrame" in ast.unparse(c) and "spoken(" in ast.unparse(c)
    ]
    assert spoken_signoffs, "no sign-off speaks before it hangs up?"
    for text in spoken_signoffs:
        assert "*spoken(" in text, text
        assert text.index("*spoken(") < text.index("EndWorkerFrame"), text


def test_the_reason_is_written_beside_the_helper():
    doc = inspect.getdoc(spoken)
    assert "one breath" in doc
    assert "app/utils/sentences.py" in doc
