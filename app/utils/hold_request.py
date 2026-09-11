"""When the prospect asks for a moment, the agent stops. Not says it stops — stops.

Live call 8d86156e, 11 Sep 2026:

    AGENT -> "It sits in Varthur near Sarjapur Road. There is a 3-acre on-site golf
              course and a 1.5-acre private lake. Does that sound like your kind of place?"
    USER  -> "One second, one second, wait. Wait wait wait wait.
              एक minute रुको. एक minute रुको ma'am."
    AGENT -> "Sure, I'll wait. As I mentioned, there is a 3-acre golf course and a
              1.5-acre private lake. Does that sound like your kind of place?"
    USER  -> "No, thank you."
    AGENT -> "Thank you for your time, Rahul."
    USER  -> "I'm not interested."

The prospect asked four times, in two languages. The agent said "Sure, I'll wait" and did
not wait — it repeated the pitch it had just given, word for word, and lost the call in the
next turn.

Read the agent's reply again: the words are right. The model understood perfectly and
answered correctly, and then kept talking anyway, because a reply is the only thing a model
can produce. Asking the prompt for this more firmly cannot work — a prompt can make a model
SAY it will wait, which is exactly what went wrong. Silence is not something a language
model can emit. It has to be the code's decision not to speak.

So this joins repeat_request.py and booking_claim.py: rules that stopped being prose the
day the model followed them in words and broke them in behaviour.

Bounded the same way as those. A repeat request is checked first and wins, because "wait,
what did you say?" is somebody asking for MORE speech and going quiet on them is the dead
air this codebase has paid for twice. And the match is anchored whole-string, so "I can't
wait to see it" and "we are waiting for possession" — a prospect being keen — stay ordinary
sentences that get an ordinary answer.
"""

import re
from typing import Optional

from app.utils.repeat_request import wants_repeat

# Asking for a pause. English, Hinglish and Devanagari together, because on a live call one
# person used all three inside four seconds.
_HOLD = (
    r"(?:"
    r"\b(?:just\s+)?(?:one|a|two|2|1)\s*(?:second|seconds|sec|secs|minute|minutes|min|mins|moment)\b"
    r"|\bgive\s+me\s+(?:a|one|two)\s*(?:second|sec|minute|min|moment)\b"
    r"|\bhold\s+on\b|\bhang\s+on\b|\bhold\s+please\b"
    r"|\bwait\b"
    r"|\blet\s+me\s+(?:check|see|think|look)\b"
    r"|\bek\s*(?:minute|min|minat|second|sec)\b"
    r"|\b(?:zara|zara|jara|thoda|thodi)\s*(?:ruko|rukiye|ruk|der|sa)\b"
    r"|\bruko\b|\brukiye\b|\brukna\b|\bruk\s+ja(?:o|iye)\b|\bthehro\b"
    # "एक minute" — Devanagari number, Latin unit, inside one breath. This is not an edge
    # case to be tidy about: it is the exact utterance from call 8d86156e, and a pattern
    # that demanded one script per phrase missed the only line it was written for.
    r"|एक\s*(?:मिनट|मिनिट|सेकंड|minute|min|minat|second|sec)"
    r"|रुको|रुकिए|रुकिये|ठहरो|ज़रा|जरा"
    r")"
)

# Words that can sit around a hold without changing what it is: politeness, hesitation, and
# the agreement people put in front of a request. On their own they are NOT a hold — "ok ok"
# is somebody listening, and app/utils/barge_in.py already knows what to do with that.
_AROUND_IT = (
    r"(?:ma'?am|madam|sir|please|plz|ji|bhai|yaar|na|to|toh|ok|okay|okey|haan|haa|han"
    r"|and|um|uh|er|hmm|hm|मैडम|सर|जी|हाँ|हां|और)"
)

_ONLY_A_HOLD = re.compile(rf"^\s*(?:(?:{_HOLD}|{_AROUND_IT})[\s,.!?।-]*)+$", re.I)
_HAS_A_HOLD = re.compile(_HOLD, re.I)

# What the agent says instead of its reply. Short on purpose: anything longer is the agent
# taking the turn it was just asked to give up, and a prospect who wanted ten seconds of
# quiet does not want a sentence about how quiet it is going to be.
HOLD_ACK = "Sure, take your time."


def wants_to_hold(line: Optional[str]) -> bool:
    """True when the prospect asked the agent to stop talking for a moment.

    False for anything that is also a request to hear something again: those want more of
    the call, not less, and the cost of confusing them is dead air on somebody who was
    leaning in. False, too, for any sentence with content of its own — "wait, so what is the
    price?" is a question and gets answered.
    """
    if not line or not line.strip():
        return False
    text = line.strip()
    if wants_repeat(text):
        return False
    if not _HAS_A_HOLD.search(text):
        return False
    # Anchored: the utterance has to be the request and nothing else. Without this, every
    # "I can't wait to move in" would silence the agent on its most interested prospect.
    #
    # The ^ and $ in the pattern are what do the work, so match() and search() are the same
    # here and no test can tell them apart — match() says the intent. The anchor is also why
    # the wants_repeat() check above cannot currently be reached by any real sentence: a
    # repeat request always carries a word that is neither a hold nor filler. It is kept
    # because the two detectors are both grown from live calls and will keep growing, and
    # the day they overlap the wrong answer is silence on somebody leaning in.
    return bool(_ONLY_A_HOLD.match(text))
