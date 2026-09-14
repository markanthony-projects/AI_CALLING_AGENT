"""One word is an answer to a question. It is never a decision to end the call.

Call 578195d1, 14 Sep 2026. Forty-five seconds, and the prospect was warm:

    USER  -> "Yeah, I was looking to buy a property."
    AGENT -> "It is called Abhee Codename New Dimension. … Is Varthur convenient for you?"
    USER  -> "Yeah."
    AGENT -> "It will have 14 towers spread over 45 acres. Have you been to that side of town?"
    USER  -> "No."
    AGENT -> "Thank you for your time, Rahul." → end_call

They answered a question about geography. Nothing about that "No" was a refusal — they had
already said they were buying, and already said the area suited them. The model read a
one-word answer as the end of the conversation and hung up on a qualified lead.

The prompt has "If no -> step 5" in the opening gate, and step 3's easy questions are all
yes/no — "Have you been to that side of town?" invites exactly this "No" and it means
nothing. So the routing rule and the question style pull against each other, and the model
picked the wrong one. Both of those are mine.

This is the rule the prompt cannot be trusted with, in the shape the rest of them take: a
bare yes or no is an answer, and the call does not end on it. To end a call somebody has to
have said something — "not interested", "no thank you", "call me later", anything with
enough in it to be a decision.

Deliberately one-sided. Being wrong here costs one more question on a call that was ending
anyway. Being wrong the other way is what this was written for.
"""

import re
from typing import Optional

# Enough words to be a decision rather than a reply. "No." and "Yeah." are answers; "No,
# thank you." and "Not right now" are somebody leaving, and they clear this easily.
ENOUGH_TO_BE_A_DECISION = 3

# Said in any number of words, these end a call and this must never hold them. Kept separate
# from the length test on purpose: a short refusal is still a refusal.
_A_REFUSAL = re.compile(
    r"\bnot interested\b|\bno thanks?\b|\bthank you\b|\bnahi chahiye\b"
    r"|\bdo ?n'?t call\b|\bdo not call\b|\bstop calling\b|\bnever call\b"
    r"|\bremove my number\b|\bcut the call\b|\bhang up\b|\bbusy\b|\blater\b",
    re.I,
)

REFUSAL_REASON = (
    "They answered your question. A one-word yes or no is a reply, not a decision to end "
    "the call — and this prospect has already told you they are buying. Do NOT end the "
    "call. Carry on from where you were: react to what they just said, then ask your next "
    "question."
)

# After this many, the model has asked to end the call twice against this rule and may be
# reading something in the conversation that this does not. Bounded so a guard meant to save
# a lead cannot become a call nobody can get off.
MAX_BARE_REFUSALS = 2


def is_a_bare_answer(line: Optional[str]) -> bool:
    """True when the last thing they said is too small to be a decision about the call."""
    text = (line or "").strip()
    if not text:
        return False
    if _A_REFUSAL.search(text):
        return False
    # Punctuation only would be a transcript of silence, not an answer.
    words = [w for w in re.findall(r"[^\W_]+", text, re.UNICODE)]
    return 0 < len(words) < ENOUGH_TO_BE_A_DECISION
