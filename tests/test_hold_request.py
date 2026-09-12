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

# Call da2f64fd, 11 Sep 2026 — the first live test of the guard above, which did not fire.
# "One" on its own is not a hold and was not filler, so one stuttered word threw out the
# whole utterance. The agent got lucky: the model happened to answer "No problem at all.
# Take your time." of its own accord. The next prospect would not have been.
THE_ONE_THAT_STILL_GOT_THROUGH = "Hold on, hold on. One one minute."


# --- the utterance this exists for ------------------------------------------------------


def test_the_line_that_lost_the_call():
    assert wants_to_hold(THE_LIVE_ONE) is True


def test_the_line_that_slipped_past_the_first_version():
    assert wants_to_hold(THE_ONE_THAT_STILL_GOT_THROUGH) is True


@pytest.mark.parametrize(
    "said",
    [
        "One one minute",
        "one one minute please",
        "ek ek minute",
        "एक एक minute",
        "hold on hold on",
        "two two minutes",
        "wait wait wait wait",
    ],
)
def test_people_stutter_when_they_want_you_to_stop(said):
    """The repetition is not noise around the request. On a phone call it IS the request,
    said twice because the agent did not stop the first time — so a pattern that treats the
    repeated word as foreign content fails exactly when it is needed most."""
    assert wants_to_hold(said) is True


@pytest.mark.parametrize("said", ["sorry one second", "just one second", "hold", "a minute"])
def test_the_ordinary_ways_of_asking(said):
    assert wants_to_hold(said) is True


@pytest.mark.parametrize(
    "said",
    ["one crore", "एक crore", "a 3 BHK", "two BHK", "one two three", "a villa or a plot"],
)
def test_a_loose_number_is_not_a_request_for_silence(said):
    """What widening the filler risks, and the reason _HAS_A_HOLD still has to find a real
    hold phrase: a budget, a configuration and a count are numbers too."""
    assert wants_to_hold(said) is False


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
    # From wants_to_hold forward: there is a second discard_reply earlier in the handler now,
    # for the line check, and index() from the start of the string finds that one instead.
    start = src.index("wants_to_hold")
    hold_branch = src[start : src.index("discard_reply", start) + 400]
    assert "_turns_heard += 1" in hold_branch


# --- "Hello?" is not "tell me about the project" ----------------------------------------
#
# Call a7f92175, 11 Sep 2026. Twenty-seven seconds, start to finish:
#
#     AGENT → "Hi, Good afternoon Rahul. I am Priya... Can I speak to you for a minute?"
#     USER  → "Hello."
#     AGENT → "We are launching a new project in Varthur, Sarjapur Road. It is called
#              Abhee Codename New Dimension. It is Bengaluru's first Scotland-themed
#              residential township. Are you looking for any property purchase?"
#     USER  → "Regarding what No no, thank you."
#
# They said one word, got thirty back, and still did not know why they were being called.


def test_a_line_check_is_not_consent_to_pitch():
    from app.utils.repeat_request import checking_the_line

    for said in ["Hello.", "Hello?", "hello", "Are you there?", "can you hear me"]:
        assert checking_the_line(said) is True, said


@pytest.mark.parametrize(
    "said",
    ["Yeah tell me", "3 BHK", "not interested", "Hello, yes, who is this?", "say it again"],
)
def test_anything_with_a_conversation_in_it_is_not_a_line_check(said):
    """"Hello, yes, who is this?" is somebody talking. Re-introducing over that would be the
    agent answering a question nobody asked."""
    from app.utils.repeat_request import checking_the_line

    assert checking_the_line(said) is False


def test_it_only_applies_before_the_agent_has_said_anything_of_its_own():
    """Later in the call a "Hello?" means the line went quiet mid-sentence, and
    GreetingOnlyMinWords already holds those until the agent finishes. Re-introducing at turn
    six would be worse than the bug."""
    src = _handler()
    guard = src[src.index("checking_the_line") - 200 : src.index("checking_the_line") + 60]
    assert "_turns_heard == 0" in guard


def test_the_reintroduction_drops_the_time_of_day():
    """"Good afternoon" is true once. Said twice inside ten seconds it is the single most
    obviously automated thing on the call."""
    from app.services.agent import build_opening_line, build_reintroduction

    again = build_reintroduction("Some Project", "RAHUL", developer_name="Some Developer")
    assert "Good" not in again
    assert "afternoon" not in again and "morning" not in again and "evening" not in again
    # but it still says who is calling and still hands the turn over
    assert "Rahul" in again
    assert "Some Developer" in again
    assert again.rstrip().endswith("?")
    assert build_opening_line("Some Project", "RAHUL", developer_name="Some Developer") != again


def test_the_reintroduction_survives_a_missing_name():
    """The dial list has blanks, and a greeting that says "Hi ." is worse than one that just
    starts."""
    again = None
    from app.services.agent import build_reintroduction

    for missing in (None, "", "   "):
        again = build_reintroduction("Some Project", missing, developer_name="Some Developer")
        # No name to confirm, so it asks for one rather than leaving the sentence hanging.
        assert "May I know your good name?" in again
        assert "Am I speaking with" not in again
        assert "  " not in again


@pytest.mark.parametrize(
    "said",
    [
        "can you hear me, I said do not call again",
        "are you there? not interested",
        "stop calling, can you hear me",
    ],
)
def test_somebody_refusing_is_not_checking_the_line(said):
    """Both at once is a real thing to say: they cannot hear well AND they want off the
    call. Re-introducing there would restart a call the person is trying to end, and this
    codebase does not hold anybody on a line.

    Mutation testing found it: the refusal check was removable with nothing failing."""
    from app.utils.repeat_request import checking_the_line

    assert checking_the_line(said) is False


def test_a_greeting_still_playing_is_not_cut_to_be_said_again():
    """Call f1d9804b, 11 Sep 2026. The prospect said "Hello." two seconds into the greeting:

        AGENT → "Hi, Good evening Rahul. I am Priya calling you from Abhee Ventures..."
        USER  → "Hello."
        TTS reconnected (1)
        AGENT → "Hi Rahul. I am Priya from Abhee Ventures. Can I speak to you for a minute?"

    They were hearing the introduction when they said it, and the guard cut it off to say
    the same thing again — so they heard it twice, both halves, and Sarvam reopened its
    websocket in between. Dropping the reply is the whole job while the greeting is playing;
    the greeting already ends on the question that hands them the turn."""
    src = _handler()
    branch = src[src.index("checking_the_line") : src.index("wants_to_hold")]
    assert "farewell.is_speaking" in branch
    # and it must return before anything is queued, not after
    assert branch.index("farewell.is_speaking") < branch.index("InterruptionWorkerFrame")


def test_the_reintroduction_goes_into_the_context_as_one_message():
    """spoken() appends per sentence by default, which put three assistant messages in for
    one thing said — three AGENT lines in the log, and a changed prefix that cost the first
    inference its cache: turn 2 of that call read cached=0 where every other call reads
    thousands. The greeting has always done this correctly; this now matches it."""
    src = _handler()
    branch = src[src.index("build_reintroduction") : src.index("wants_to_hold")]
    assert "context.add_message" in branch
    assert "append_to_context=False" in branch
