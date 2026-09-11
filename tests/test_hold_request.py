"""When somebody asks for a moment, they get the moment.

Call 8d86156e, 11 Sep 2026:

    AGENT → "It sits in Varthur near Sarjapur Road. There is a 3-acre on-site golf course
             and a 1.5-acre private lake. Does that sound like your kind of place?"
    USER  → "One second, one second, wait. Wait wait wait wait.
             एक minute रुको. एक minute रुको ma'am."
    AGENT → "Sure, I'll wait. As I mentioned, there is a 3-acre golf course and a
             1.5-acre private lake. Does that sound like your kind of place?"
    USER  → "No, thank you."
    USER  → "I'm not interested."

Asked four times, in two languages. The agent said it would wait and then said its pitch
again, word for word, and the prospect left in the next turn.

The reply is the evidence that a prompt cannot fix this. "Sure, I'll wait" is the correct
sentence — the model understood exactly — and then it kept talking, because a reply is the
only thing a model can emit. Silence has to be the code's decision.
"""

import ast
import inspect
from pathlib import Path

import pytest

from app.utils.hold_request import HOLD_ACK, wants_to_hold

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")

THE_LIVE_ONE = (
    "One second, one second, wait. Wait wait wait wait. "
    "एक minute रुको. एक minute रुको ma'am."
)


# --- the utterance this exists for ------------------------------------------------------


def test_the_line_that_lost_the_call():
    assert wants_to_hold(THE_LIVE_ONE) is True


def test_the_number_and_the_unit_may_be_in_different_scripts():
    """"एक minute" — Devanagari number, Latin unit, one breath. A first draft of the pattern
    demanded one script per phrase and missed the only line it was written for. This is not
    an exotic case on a Hindi-English call; it is the normal case."""
    assert wants_to_hold("एक minute") is True
    assert wants_to_hold("एक मिनट") is True
    assert wants_to_hold("ek minute") is True


@pytest.mark.parametrize(
    "said",
    [
        "wait",
        "Wait wait wait",
        "one second",
        "one sec",
        "just a second",
        "hold on",
        "hold on please sir",
        "hang on",
        "give me a minute",
        "two minutes",
        "2 min",
        "let me check",
        "let me see",
        "ek minute",
        "ek minute ruko",
        "ruko",
        "rukiye",
        "zara ruko",
        "thoda ruko ji",
        "ok wait",
        "haan ek minute",
        "wait na",
        "रुको",
        "रुकिए",
        "एक मिनट",
        "ठहरो",
    ],
)
def test_asking_for_a_moment(said):
    assert wants_to_hold(said) is True


# --- and everything that only sounds like it --------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        # A prospect being keen. Silencing the agent here would be the opposite of the fix.
        "I can't wait to see it",
        "I can't wait to move in",
        "we are waiting for possession",
        "I am waiting for the launch",
        # A hold with a question attached is a question.
        "wait, so what is the price?",
        "hold on, is that the carpet area?",
        "one second, what did you say the price was",
        # Listening, not asking. app/utils/barge_in.py owns these.
        "ok ok",
        "haan haan",
        "hmm",
        "ji",
        # Ordinary answers.
        "3 BHK",
        "Sunday",
        "no thank you",
        "not interested",
        "give me the price",
        "let me know the price",
        "एक crore",
        "एक point four crore",
    ],
)
def test_not_a_request_for_silence(said):
    assert wants_to_hold(said) is False


@pytest.mark.parametrize(
    "said",
    ["wait what did you say", "hold on can you repeat that", "one second, say that again"],
)
def test_asking_to_hear_it_again_wins(said):
    """Both are "stop talking", and they want opposite things next. Going quiet on somebody
    who asked to hear something again is the dead air this codebase has already paid for
    twice — so repeat_request is checked first and takes it."""
    assert wants_to_hold(said) is False


def test_a_repeat_request_wins_even_when_the_words_look_like_a_hold(monkeypatch):
    """No real sentence reaches this today — the whole-string anchor rejects every repeat
    request before the check matters, and mutation testing proved it by deleting the check
    with nothing failing.

    It is kept, and this is what keeps it honest. Both detectors are grown from live calls
    and both keep growing; the day one of them widens far enough to overlap, the answer has
    to be the same as it is today. Being wrong toward speaking costs a repeated sentence.
    Being wrong toward silence costs the lead, and this codebase has paid that twice."""
    import app.utils.hold_request as hold_request

    monkeypatch.setattr(hold_request, "wants_repeat", lambda line: True)
    assert hold_request.wants_to_hold("one second") is False
    assert hold_request.wants_to_hold(THE_LIVE_ONE) is False


@pytest.mark.parametrize("said", [None, "", "   "])
def test_nothing_said_is_not_a_hold(said):
    assert wants_to_hold(said) is False


# --- what the agent does with it --------------------------------------------------------


def test_the_acknowledgement_is_short():
    """Anything longer is the agent taking the turn it was just asked to give up. The
    prospect wanted quiet, not a sentence about how quiet it is about to be."""
    assert len(HOLD_ACK.split()) <= 6


def _handler() -> str:
    tree = ast.parse(AGENT_SRC)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_user_turn_stopped"
    )
    return ast.unparse(node)


def test_the_reply_is_discarded_before_the_gate_is_released():
    """Order, not presence. TurnFinalityGate.user_turn_stopped() is what puts a held reply
    on the line, so a discard after it would be a discard of something already spoken."""
    src = _handler()
    assert "wants_to_hold" in src
    assert src.index("discard_reply") < src.index("user_turn_stopped()")


def test_the_agent_says_something_rather_than_going_silent_mid_sentence():
    """It was cut off mid-pitch by the barge-in. Saying nothing at all after that reads as
    a dropped line, and a prospect who thinks the line dropped says "hello?" — which this
    codebase has watched turn into a hang-up."""
    assert "HOLD_ACK" in _handler()


def test_the_interruption_is_flushed_before_the_acknowledgement():
    """The same three steps end_call uses. An InterruptionWorkerFrame only takes effect
    after its own lap through the pipeline, so without the flush the acknowledgement can
    enter first and be cancelled by the interruption meant to clear the way for it."""
    src = _handler()
    assert src.index("InterruptionWorkerFrame") < src.index("flush_pipeline")
    assert src.index("flush_pipeline") < src.index("HOLD_ACK")


def test_the_nudge_cannot_break_the_silence():
    """The other door into the same bug. VAD fires on a rustle, the transcript is empty, and
    the dead-air nudge asks the question again — into the quiet the prospect just asked for.

    Asserted on the parsed guard rather than on the text: mutation testing turned the guard
    into `if False: return` and a substring search for "_holding" stayed green, because the
    name still appears three lines earlier in the branch that sets it."""
    tree = ast.parse(AGENT_SRC)
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_user_turn_stopped"
    )
    guards = [
        n for n in ast.walk(handler)
        if isinstance(n, ast.If)
        and "_holding" in ast.unparse(n.test)
        and len(n.body) == 1
        and isinstance(n.body[0], ast.Return)
    ]
    assert guards, "nothing stops the dead-air nudge while the prospect is thinking"
    nudge = next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "dead_air_nudge"
    )
    assert min(g.lineno for g in guards) < nudge.lineno, "the guard is after the nudge"


def test_speaking_again_ends_the_hold():
    """Otherwise the agent is mute for the rest of the call."""
    src = _handler()
    assert "_holding = False" in src


def test_the_hold_is_counted_as_a_turn_heard():
    """It is the prospect speaking. A turn that does not count reads as an unresponsive
    call to every downstream decision that looks at _turns_heard."""
    src = _handler()
    hold_branch = src[src.index("wants_to_hold") : src.index("discard_reply") + 400]
    assert "_turns_heard += 1" in hold_branch
