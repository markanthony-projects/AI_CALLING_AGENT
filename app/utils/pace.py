"""Speaking slower when the prospect asks for it.

Live call 3c43b6bc, 4 Sep 2026:

    AGENT → "...Are you looking for your own stay, or for investment?"
    USER  → "Sorry I didn't catch that. Can you say it again with"
    AGENT → "No problem at all. It is a huge 45-acre township with a 3-acre golf
             course and a private lake. Are you looking for your own stay..."
    USER  → "little bit slow hai?"
    AGENT → "No problem at all. The project is a big township..."

They asked twice. The agent repeated itself twice and said it at exactly the same speed
both times, because the pace was a constant. There was no way for it to do the one thing
being asked — which is worse than not understanding, because the prospect can hear that
they were heard and ignored.

Pace can be changed mid-call. Pipecat's TTSUpdateSettingsFrame reaches Sarvam's
_update_settings, which resends the config on the connection already open, so this costs no
reconnect and nothing is interrupted.

Deliberately never faster than the configured default. A prospect asking to slow down is a
real request; nobody on a cold sales call wants the pitch delivered faster, so "faster"
only ever walks back an earlier slow-down rather than running past where the call started.

The detection is loose on purpose. "The market is slow" would move the pace one step, and
that is the trade being made: the cost of a false positive is a slightly slower sentence
that the next request undoes, and the cost of a miss is the call above.
"""

import re
from typing import Optional

# One step per request, because a prospect who is still not comfortable will ask again and
# two steps down from one sentence would leave the agent crawling.
PACE_STEP = 0.15

# Sarvam accepts 0.5 to 2.0 on bulbul:v3. This floor is well inside it: below about 0.7 the
# voice stops sounding like a person speaking carefully and starts sounding broken, which is
# not what was asked for either.
MIN_PACE = 0.7

_SLOWER = re.compile(
    r"\bslow(?:er|ly)?\b"
    r"|\bslow\s*down\b"
    r"|\b(?:too|so|very|bahut|bohot)\s+fast\b"
    r"|\bdheere\b|\bdhire\b|\baaram\s*se\b"
    r"|धीरे|धीमे",
    re.I,
)

_FASTER = re.compile(
    r"\bfaster\b|\bspeed\s*(?:it\s*)?up\b|\bhurry\b"
    r"|\b(?:too|so|very|bahut|bohot)\s+slow\b"
    r"|\bjaldi\b|जल्दी",
    re.I,
)


def pace_request(transcript: Optional[str]) -> Optional[str]:
    """"slower", "faster", or None when the prospect said nothing about it.

    "Too slow" is checked first and wins outright. It contains the word "slow" and would
    otherwise be read as a request to slow down further — the opposite of what was said.
    """
    text = (transcript or "").strip()
    if not text:
        return None
    if _FASTER.search(text):
        return "faster"
    if _SLOWER.search(text):
        return "slower"
    return None


def adjusted_pace(current: float, request: Optional[str], default: float) -> float:
    """The new pace, or the current one when nothing should change.

    Bounded below by MIN_PACE and above by the call's own default: this exists to honour
    "please slow down", not to let a prospect drive the delivery anywhere they like.
    """
    if request == "slower":
        return max(MIN_PACE, round(current - PACE_STEP, 2))
    if request == "faster":
        return min(default, round(current + PACE_STEP, 2))
    return current
