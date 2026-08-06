"""The prospect answered, nothing came back, and the agent went quiet.

Every case here is taken from call 5664ace6, where the pipeline twice finalized a five
second user turn with an empty transcript, said nothing at all, and waited to be rescued:

    AGENT → "Got it, Shivam. Is this for your own stay, or for investment?"
    VAD fired with no transcribable speech after 5003ms
    USER  → "Hello."                      <- the prospect breaking eleven seconds of silence
    USER  → "Yeah, I SAID for investment."
"""

import ast
import asyncio
import inspect
import textwrap
from pathlib import Path

import pytest
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from app.utils.reprompt import (
    DEAD_AIR_APOLOGY,
    MAX_DEAD_AIR_NUDGES,
    dead_air_nudge,
    last_question,
)
from app.utils.stt_witness import SttWitness


# ─── Which question gets asked again ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "Got it, Shivam. Is this for your own stay, or for investment?",
            "Is this for your own stay, or for investment?",
        ),
        (
            "Hi Shivam. We are launching a new project in Varthur, near Sarjapur Road. "
            "Are you looking for any property purchase?",
            "Are you looking for any property purchase?",
        ),
        (
            "Sure, that helps, Shivam. Which city or specific area in North Bengaluru are "
            "you focusing on?",
            "Which city or specific area in North Bengaluru are you focusing on?",
        ),
    ],
)
def test_it_repeats_the_question_without_the_acknowledgement(line, expected):
    """The acknowledgement was for what they said last time; only the question is unanswered."""
    assert last_question(line) == expected


def test_a_sign_off_is_never_repeated():
    """The closing line ends the call. Saying it twice is the one thing worse than silence."""
    goodbye = (
        "That works well, Shivam. Thank you for sharing these details. Our property expert "
        "will call you with better options for North Bengaluru."
    )
    assert last_question(goodbye) is None
    assert dead_air_nudge(goodbye) is None


def test_nothing_to_repeat_when_there_is_no_last_line():
    assert dead_air_nudge("") is None
    assert dead_air_nudge(None) is None


def test_the_last_question_wins_when_a_turn_asks_two():
    line = "Are you free this week? Or would Saturday suit you better?"
    assert last_question(line) == "Or would Saturday suit you better?"


def test_the_nudge_blames_the_line_and_not_the_prospect():
    """They answered. "Sorry, I missed that" tells them they mumbled, which is both wrong
    and the sort of thing that loses a sale."""
    nudge = dead_air_nudge("Got it, Shivam. What budget are you thinking of?")
    assert nudge == f"{DEAD_AIR_APOLOGY} What budget are you thinking of?"
    lowered = DEAD_AIR_APOLOGY.lower()
    assert "line" in lowered
    for blame in ("you said", "you were saying", "didn't catch", "did not catch"):
        assert blame not in lowered


def test_repeating_forever_is_not_the_answer():
    """A line that swallows three answers is not going to carry the call; the idle timeout
    should take it from there rather than the agent talking to itself."""
    assert 1 <= MAX_DEAD_AIR_NUDGES <= 2


# ─── What the STT actually sent ───────────────────────────────────────────────────────


def _interim(text):
    return InterimTranscriptionFrame(text=text, user_id="caller", timestamp="t")


def _final(text):
    return TranscriptionFrame(text=text, user_id="caller", timestamp="t")


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _feed(witness, frames):
    """Drive the real process_frame, bypassing only the base class's started-processor check."""
    captured = _Captured()
    witness.push_frame = captured.push

    async def noop(*a, **kw):
        pass

    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await witness.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


def _witness(*frames):
    w = SttWitness()
    asyncio.run(_feed(w, frames))
    return w


def test_silence_from_the_stt_is_distinguishable_from_silence_on_the_line():
    """The whole point. These two need different fixes, and the old log line could not tell
    them apart: one is audio never reaching Deepgram, the other is Deepgram hearing noise."""
    nothing = SttWitness().report()
    noise = _witness(_interim(""), _interim("")).report()
    assert nothing != noise
    assert "nothing" in nothing.lower()
    assert "no words" in noise.lower()


def test_interims_without_a_final_are_reported_with_what_was_heard():
    """Words arrived and the turn still finalized empty — an endpointing problem, and the
    text is the evidence that the prospect really did answer."""
    report = _witness(_interim("for"), _interim("for invest")).report()
    assert "2 interim" in report
    assert "0 final" in report
    assert "for invest" in report


def test_every_frame_reaches_the_aggregator_unchanged():
    """The witness sits in the audio path. Swallowing or rewriting a transcript here would
    trade a diagnosis for a broken call."""
    sent = [_interim("for"), _final("for investment"), _interim("")]
    got = asyncio.run(_feed(SttWitness(), sent))
    assert got == sent


def test_the_witness_never_touches_the_audio_path():
    """The structural half of the same guarantee: no branch may return early or push
    something other than what it was handed."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(SttWitness.process_frame)))
    pushes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "push_frame"
    ]
    assert len(pushes) == 1, "exactly one push, so no branch can swallow a frame"
    # ...and it pushes the frame it was handed, unmodified.
    assert isinstance(pushes[0].args[0], ast.Name) and pushes[0].args[0].id == "frame"
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is None]
    assert not returns, "an early return would drop a transcript"


def test_a_turn_starts_with_a_clean_tally():
    w = _witness(_interim("hello"), _final("hello"))
    w.reset()
    assert "nothing" in w.report().lower()


def test_last_turn_words_do_not_leak_into_this_turn_s_report():
    """The exact misdiagnosis this class exists to prevent: a turn that heard nothing but
    noise would report the PREVIOUS turn's words and read as an endpointing bug."""
    w = _witness(_final("for investment"))
    w.reset()
    asyncio.run(_feed(w, [_interim(""), _interim("")]))
    report = w.report()
    assert "for investment" not in report
    assert "no words" in report


# ─── How the agent uses them ──────────────────────────────────────────────────────────


def _agent_source() -> str:
    return Path("app/services/agent.py").read_text(encoding="utf-8")


def test_the_nudge_is_held_back_while_the_agent_is_still_speaking():
    """An empty turn during the agent's own speech is the false barge-in this counter was
    built for. The question is already being asked; repeating it over the top is worse than
    the line noise that triggered it."""
    src = _agent_source()
    guard = "if _agent_speaking or _llm_in_flight:"
    assert guard in src
    # The guard must sit between the warning and the nudge, not anywhere in the file.
    warn = src.index("VAD fired with no transcribable speech")
    speak = src.index("Nothing heard back; asking again")
    assert warn < src.index(guard) < speak


def test_speaking_state_is_tracked_from_both_ends():
    """Set on one event and cleared on the other. Missing either leaves it stuck, and stuck
    True silences the rescue permanently while stuck False fires it over live speech."""
    src = _agent_source()
    assert "_agent_speaking = True" in src
    assert "_agent_speaking = False" in src


def test_the_empty_turn_warning_carries_the_witness_report():
    """Otherwise the next occurrence is as undiagnosable as the last two were."""
    src = _agent_source()
    warn = src.index("VAD fired with no transcribable speech")
    assert "stt_witness.report()" in src[warn : warn + 400]
