"""Recognise an answering machine from what it says.

A dial landed on voicemail and the whole pipeline treated it as a conversation:

    AGENT → "Hi, I am Ananya calling you on behalf of ..." [interrupted]
    USER  → "The person you are trying to reach is not available. At the please record
             your message. When you have finished recording, you may hang up."
    Call finalised | status=COMPLETED

Three separate costs. The dashboard counted a connect that never happened, so the campaign's
answer rate is wrong in the direction that flatters it. An extraction job ran against a
recording of somebody's outgoing message. And the agent spoke its opening line to a machine,
which is the one thing on the call that cost real money.

Detection is on the words rather than on carrier signalling because it works on every
carrier and can be tested. Vobiz does offer machine_detection at dial time and that is the
better long-term signal — it would stop us paying for the audio leg at all — but it has to
be verified against their webhook payload before anything can depend on it.
"""

import re
from typing import Optional

# Deliberately phrases, not keywords. "not available" alone is something a person says
# ("sorry I'm not available on Sunday"), and hanging up on a live prospect who used those
# words is far worse than transcribing one voicemail.
_MACHINE_PHRASES = (
    "record your message",
    "leave your message",
    "leave a message",
    "after the tone",
    "after the beep",
    "at the tone",
    "at the beep",
    "finished recording",
    "when you have finished",
    "is not available",
    "is unavailable",
    "not answering the call",
    "unable to take your call",
    "cannot take your call",
    "can't take your call",
    "please try again later",
    "the person you are trying to reach",
    "the number you have dialled",
    "the number you have dialed",
    "switched off",
    "out of coverage area",
    "voice mail",
    "voicemail",
    "kripya apna sandesh",
    "sandesh record",
    "uplabdh nahi",
    "sampark kshetra se bahar",
)

# One phrase can be a coincidence in a long sentence. Machines say several.
_MIN_PHRASES = 2

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", text.lower().replace("'", "'"))


def machine_phrases(text: Optional[str]) -> list[str]:
    """Which voicemail phrases appear in `text`."""
    if not text:
        return []
    low = _normalise(text)
    return [p for p in _MACHINE_PHRASES if p in low]


def is_answering_machine(text: Optional[str]) -> bool:
    """True when this turn reads like a recorded greeting rather than a person.

    Two phrases are required, or one that no human utterance contains. A prospect saying
    "I'm not available right now" trips one generic phrase and must stay on the call.
    """
    found = machine_phrases(text)
    if len(found) >= _MIN_PHRASES:
        return True
    return any(p in found for p in ("record your message", "after the beep", "after the tone"))
