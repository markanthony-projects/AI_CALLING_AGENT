""""Say it again" is not a goodbye, and the call must not end on one.

Live call 6a58a7f4, 10 Sep 2026. Forty-one seconds, and the prospect had already said they
were interested:

    USER  -> "Yeah, I was interested."
    AGENT -> "It sits on 45 acres with 14 towers. Do you know Varthur?"
    USER  -> "Sorry, I did not catch that. Can you say it again?"
    AGENT -> end_call: "Sorry about that Rahul. I will call you back later."

The prompt already forbids exactly this, in the OBJECTIONS section, in capitals: a line
check "is NOT a brush-off — they heard silence and are checking the call is still on. NEVER
offer a callback for this; it sounds like you want to get off the phone." The tool's own
description says "Do NOT call it for a 'hello' or an interruption." The model did it anyway,
and a warm lead was hung up on.

So this is the rule as code, the same way the budget, the locality and the invented booking
became code after being ignored as prose. A prospect asking to hear something again is
asking for MORE of the call, not less. It is the one sentence where hanging up is furthest
from what they meant.

Deliberately conservative in one direction only. A refusal that also happens to mention
hearing — "not interested, do not call again" — is a refusal, and this must never hold a
call open on someone trying to leave. Everything else is caught: being wrong here costs one
repeated question, and being wrong the other way costs the lead.
"""

import re
from typing import Optional

# Checked first, and it wins. "Don't call again" contains "again"; a person leaving must be
# allowed to leave.
_REFUSAL = re.compile(
    r"\bnot interested\b"
    r"|\bdo ?n'?t call\b|\bdo not call\b|\bstop calling\b|\bnever call\b"
    r"|\bremove my number\b|\bcut the call\b|\bhang up\b",
    re.I,
)

# Asking to hear the last thing again.
_REPEAT = re.compile(
    r"\bcome again\b"
    r"|\bsay (?:it|that|again)\b"
    r"|\brepeat (?:it|that|again)\b"
    r"|\b(?:can|could|would|will) you repeat\b"
    r"|\bonce (?:more|again)\b|\bone more time\b"
    # `\s*n[o']?t` covers "didn't", "did not" and "didnt" in one, so a separate "did not"
    # branch would be dead: mutation testing found it unremovable-by-any-test, which is what
    # a redundant pattern looks like from the outside.
    r"|\b(?:did|could|would)\s*n[o']?t (?:catch|hear|understand|get|follow)\b"
    r"|\b(?:can'?t|cannot) hear (?:you|that)\b"
    r"|\bnot audible\b|\bbreaking up\b|\bnot clear\b"
    r"|\bpardon\b|\bwhat did you say\b|\bwhat was that\b"
    r"|\bsamajh nahi\b|\bphir se\b|\bdobara\b",
    re.I,
)

# Checking the line is still open. "Hello" on its own only — the greeting reply "Hello, yes?"
# is a conversation, not a line check, and the difference is whether anything follows it.
_LINE_CHECK = re.compile(
    r"^\s*(?:hello|hallo)\s*[?.!]*\s*$"
    r"|\bare you there\b|\bis anyone there\b|\bcan you hear me\b|\bare you still there\b",
    re.I,
)

# After this many, the model has asked to end the call twice against this rule and the
# prospect may genuinely be trying to leave in words this does not recognise. Bounded so a
# guard meant to save a lead cannot become a call nobody can get off.
MAX_REPEAT_REFUSALS = 2

REFUSAL_REASON = (
    "The prospect asked you to repeat yourself — they are still on the call and want to "
    "hear you. Do NOT end the call. Apologise in a few words and say your last point again, "
    "more simply, then ask your question again."
)

_APOLOGY = "Sorry about that."


def wants_repeat(line: Optional[str]) -> bool:
    """True when the prospect asked to hear something again, or checked the line is open."""
    if not line or not line.strip():
        return False
    text = line.strip()
    if _REFUSAL.search(text):
        return False
    return bool(_REPEAT.search(text) or _LINE_CHECK.search(text))


def checking_the_line(line: Optional[str]) -> bool:
    """True when the words are only "is this call still on?" and nothing else.

    A narrower question than wants_repeat: that one also covers "say it again", which asks
    for the last thing back. This one is somebody who has heard NOTHING yet and is checking
    the line is alive — and answering it with a sales pitch is how call a7f92175 was over in
    twenty-seven seconds.
    """
    if not line or not line.strip():
        return False
    text = line.strip()
    if _REFUSAL.search(text):
        return False
    return bool(_LINE_CHECK.search(text))


def say_again(agent_line: Optional[str]) -> str:
    """What to say when there is no way to hand the turn back to the model.

    Used only when the tool call arrives with no result_callback to answer through. Repeats
    the agent's own last words rather than paraphrasing them, because paraphrasing needs the
    model and the model is what is unavailable on this path.
    """
    spoken = (agent_line or "").strip()
    return f"{_APOLOGY} {spoken}" if spoken else f"{_APOLOGY} Can you hear me now?"
