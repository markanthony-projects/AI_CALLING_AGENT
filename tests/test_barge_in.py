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
        "Yeah", "Yes", "No", "Okay", "haan", "3 BHK", "Sunday", "1.5 CR",
        "Yeah sure.", "Yes please", "Not interested",
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
