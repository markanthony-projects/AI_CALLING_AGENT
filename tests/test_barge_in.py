"""Which words are allowed to cut the agent off mid-sentence.

Call 5023ff25, 10 Sep 2026. The agent went mute for the last thirty seconds:

    AGENT -> "...Which size are you thinking of?"
    USER  -> "Hello."      TTS reconnected (3)
    AGENT -> "Hi, could you tell me..."     [interrupted]
    USER  -> "Hello."      TTS reconnected (4)
    AGENT -> "Which BHK size..."            [interrupted]
    (prospect hangs up)

One word takes the floor after the greeting, "Hello" is one word, and Sarvam's TTS reopens
its websocket on every interruption. They were saying "Hello?" because they could not hear,
and each "Hello?" was what stopped them hearing.
"""

import pytest

from app.utils.barge_in import takes_the_floor


@pytest.mark.parametrize(
    "said",
    [
        "Hello.", "Hello?", "hello", "HELLO", "  Hello  ",
        "sir", "Sir?", "madam", "Ma'am", "maam",
        "Are you there?", "are you still there", "can you hear me", "Is anyone there",
    ],
)
def test_checking_the_line_does_not_take_the_floor(said):
    assert takes_the_floor(said) is False


@pytest.mark.parametrize(
    "said",
    [
        # Every short answer a person actually gives. These must interrupt exactly as before
        # — the gate was relaxed to one word because losing them cost a caller 37 seconds.
        #
        # "Yeah", "Yes", "Okay", "haan" and "Yeah sure." moved out on 11 Sep 2026, into
        # test_agreeing_along_does_not_cut_the_sentence_off below. Not a loosening: while
        # the bot is speaking — the only time this is asked — those were being DISCARDED by
        # the base class for being under min_words, which is the 37-second bug itself. Held
        # instead of dropped, they are answered the moment the sentence ends. The wait is
        # the tail of a sentence that was finishing anyway.
        "No", "3 BHK", "Sunday", "1.5 CR",
        "Yes please", "Not interested",
        # "Hello" with a conversation attached is a conversation.
        "Hello, yes, who is this?",
        "Hello can you tell me the price",
        "sir I want a 3 BHK",
    ],
)
def test_everything_else_still_takes_the_floor(said):
    assert takes_the_floor(said) is True


@pytest.mark.parametrize("said", ["", "   ", None])
def test_nothing_said_takes_nothing(said):
    assert takes_the_floor(said) is False


def test_the_call_that_produced_the_rule_is_written_beside_it():
    import app.utils.barge_in as module

    assert "5023ff25" in module.__doc__
    assert "InterruptibleTTSService" in module.__doc__


# --- agreeing along is not asking to speak ---------------------------------------------------
#
# Live gap found 11 Sep 2026 while working out what Flux would cost. The filter knew "hello"
# and "sir" and not one Hindi word, so on a Hindi call every "haan haan" and "achha theek hai"
# cut the agent off mid-description. Nothing to do with Flux; it was in production.


@pytest.mark.parametrize(
    "said",
    [
        "haan",
        "haan haan",
        "haan haan haan",
        "hmm",
        "hmmm",
        "ji",
        "ji haan",
        "haan ji",
        "achha",
        "acha acha",
        "theek hai",
        "thik hai thik hai",
        "sahi hai",
        "bilkul",
        "ok",
        "ok ok",
        "okay",
        "right",
        "yes",
        "yeah yeah",
        "mhm",
        "हाँ",
        "जी हाँ",
        "अच्छा अच्छा",
        "ठीक है",
        "बिल्कुल",
    ],
)
def test_agreeing_along_does_not_cut_the_sentence_off(said):
    assert takes_the_floor(said) is False


@pytest.mark.parametrize("said", ["haan.", "haan, haan!", "achha?", "हाँ।", "  ok  "])
def test_punctuation_and_spacing_do_not_smuggle_an_acknowledgement_past(said):
    assert takes_the_floor(said) is False


@pytest.mark.parametrize(
    "said",
    [
        "nahi",
        "nahi nahi",
        "nahi nahi ye nahi chahiye",
        "no",
        "no thanks",
    ],
)
def test_a_refusal_is_never_an_acknowledgement(said):
    """The one thing a prospect most needs to be able to say over the top of a sales pitch.
    It is also why "nahi" is not in the list, and this is the test that keeps it out."""
    assert takes_the_floor(said) is True


@pytest.mark.parametrize(
    "said",
    [
        "haan main Sunday free hoon",
        "haan bilkul batao",
        "ok to kitne ka hai",
        "yes tell me more",
        "theek hai lekin price kya hai",
        "achha ye kaunsi location hai",
    ],
)
def test_a_sentence_that_opens_with_agreement_is_still_a_sentence(said):
    """The whole-string anchor is what makes the list safe to widen. Somebody who says
    "haan" and then keeps talking is taking the floor, and the anchor is the only thing
    standing between that and a filter that swallows half the call."""
    assert takes_the_floor(said) is True


@pytest.mark.parametrize("said", ["3 BHK", "Sunday", "ek minute ruko", "kitne ka hai"])
def test_the_answers_people_actually_give_still_interrupt(said):
    assert takes_the_floor(said) is True
