"""When they have said no to THIS project, stop pitching it — and say one true thing.

Call f1d9804b, 11 Sep 2026, and again on f556caf9 the next morning after step 5 had been
told not to:

    USER  -> "Yeah but not in this project."
    ... four qualifying questions, all answered ...
    USER  -> "ऐसे two CR"
    AGENT -> "Okay, a budget of about two Crores works well. In this project we have a
              3.5 BHK Presidential at 2.06 Crores and a 4.5 BHK Presidential at 2.64
              Crores. Does the 3.5 BHK option sound like a good fit for you?"
    USER  -> "I said I'm not interested in this project."

"I said" is the prospect repeating themselves, and the call ended on it.

The static rule had been added to step 5 the day before — "NEVER pitch this project again
after that" — and this is the call that followed it. That is not evidence the model is
careless; it is evidence about where a rule lives. Step 5 is one paragraph inside a
two-hundred-line prompt resent on every turn, competing with every other paragraph. A block
that appears ONLY once the thing has happened competes with nothing. So this follows
app/utils/asked.py, which solved exactly that for a repeated question.

The first version of this block said "never mention this project again", and that was wrong
in the other direction. The campaign IS for this project, and a salesperson who hears a
requirement it genuinely fits and says nothing is not being tactful, they are being useless.

What made the live turn bad was not that it mentioned the project. It was that the prospect
had named an area AND a budget, only the budget fitted, and the agent read the price list
back as though the area had never been said — matching on the one dimension that suited it
and ignoring the one that did not. So the rule is not silence. It is: earn it by asking,
say it once, name their own words in the sentence, and take the answer.

Deliberately narrow in what it fires on. "Not this area" from someone who has not been told
where the project is stays a GUESS, and step 5's re-check on that saved a 1.5 Crore lead.
This fires only on a rejection that names the project itself.
"""

import re
from typing import Optional

# Rejecting the project, not the area and not the call. The word "project" (or "yeh/is
# project" in Hinglish) has to be in it, with a negation attached: a prospect who says only
# "not interested" is leaving, and app/utils/repeat_request.py and end_call own that.
_REJECTS_THE_PROJECT = re.compile(
    r"\bnot\s+(?:interested\s+)?(?:in\s+)?(?:this|that|the|your|ye|is)\s+project\b"
    r"|\b(?:do|does|did)\s*n[o']?t\s+want\s+(?:in\s+)?(?:this|that|the)\s+(?:\w+\s+and\s+that\s+)?project\b"
    r"|\bnot\s+(?:for|about)\s+(?:this|that)\s+project\b"
    r"|\b(?:this|that)\s+project\s+(?:is\s+)?not\s+(?:for|suitable|good)\b"
    r"|\b(?:is|ye|yeh)\s+project\s+(?:me|mein)?\s*nahi\b"
    r"|\b(?:is|ye|yeh)\s+project\s+nahi\s+chahiye\b"
    r"|इस\s*प्रोजेक्ट\s*में?\s*नहीं"
    r"|ये\s*प्रोजेक्ट\s*नहीं",
    re.I,
)

BRIEF = "\n".join(
    [
        "THEY HAVE SAID NO TO THIS PROJECT. Stop pitching it. Do not answer their next "
        "words with its prices, its configurations or its amenities, and do not ask again "
        "whether some size of it suits them — that is the same pitch in different words, "
        "and on a live call it got back \"I said I'm not interested in this project\" and "
        "the call ended there.",
        "FIND OUT WHAT THEY DO WANT. One question at a time, reacting to what they just "
        "said before you ask the next, and stop at two or three if that is all they will "
        "give you.",
        "THEN ONCE, AND ONLY IF IT GENUINELY MATCHES. This campaign is for this project, "
        "and a salesperson who hears a requirement it actually fits and says nothing is no "
        "use to anyone. So say it — once, in ONE sentence, naming the match in their own "
        "words: \"You said Varthur and around two Crores; that is exactly where this one "
        "is, and the 3.5 BHK is 2.06.\" A match means the things THEY named line up. On "
        "that live call they had named an area and a budget, only the budget fitted, and "
        "the price list went back to them as though the area had never been said. If you "
        "cannot put their own words in the sentence, it is not a match and you say nothing.",
        "THEIR ANSWER TO THAT IS FINAL, either way. If it is no again, never raise it a "
        "third time. Call end_call, read back what you noted — what they are looking for, "
        "where, and their budget — and say our property expert will call them with options "
        "that fit.",
    ]
)


def rejects_the_project(line: Optional[str]) -> bool:
    """True when the prospect ruled out THIS project, as opposed to the area or the call."""
    if not line or not line.strip():
        return False
    return bool(_REJECTS_THE_PROJECT.search(line.strip()))
