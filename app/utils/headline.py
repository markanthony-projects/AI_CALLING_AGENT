"""The one line that says why this project is worth a minute — as a sentence, not a caption.

The Headline column is written to be read, not spoken. "Bengaluru's first Scotland-themed
residential township" is a noun phrase: it sits under a photograph perfectly and has no verb
in it. Said out loud after "It is called Abhee Codename New Dimension.", the ear hangs it
onto the sentence before and waits for an ending that never arrives. A live prospect said it
felt like a sentence still going.

Three prompt attempts went at this and the third made it worse:

    fe1e1c7  "THAT LINE MUST HAVE A VERB IN IT"          -> worked once
    44f3951  + "two sentences must not start the same"   -> verb dropped to satisfy it
    781077a  + "vary by rewriting, never by deleting"    -> still a caption

By then one paragraph of the prompt carried seven rules about one sentence — have a verb,
do not be a caption, no dashes, say the project name, never open with the name, the headline
is the only reason they keep listening, and now vary the opening. The model was not being
careless. It was being asked to satisfy seven constraints at once and dropping whichever it
dropped.

So the sentence is built here and handed over finished. Nothing to compose means nothing to
get wrong, the prompt loses a paragraph rather than gaining one, and the same input always
produces the same spoken line — which is the part no amount of prompting can promise.
"""

import re
from typing import Optional

# Already a sentence about the project: it has a subject and something to do. Anything
# matching is left exactly as written, because the copy was deliberate and a second "It is"
# bolted on the front would be worse than the label was.
_ALREADY_SAYS_SOMETHING = re.compile(
    r"^\s*(?:it|this|the\s+project|we|our)\b.*?"
    r"\b(?:is|are|was|were|has|have|offers?|brings?|sits?|comes?|gives?|includes?|features?)\b"
    r"|^\s*(?:offering|featuring)\b",
    re.I,
)

# "A Scotland-themed township" wants "It is a…", not "It is A…". ONLY the leading article is
# lowered. "Bengaluru's", "India's" and any brand are names that keep their capital, and
# there is no way to tell those from an ordinary word that happens to start the line — so
# "It is Luxury homes with a private lake." keeps its capital too. It reads as emphasis at
# worst, and it is spoken identically; guessing wrong on a name would not be.
_LEADS_WITH_AN_ARTICLE = re.compile(r"^(A|An|The)\b")


def as_a_sentence(headline: Optional[str]) -> str:
    """The headline as something a person could say out loud.

    Returns "" for nothing usable, so the caller can leave the line out altogether rather
    than speak an empty sentence.
    """
    text = (headline or "").strip()
    if not text:
        return ""
    # The engine reads this directly and a trailing full stop is supplied below; a headline
    # that already ends in one must not end in two.
    text = text.rstrip(" .;,")
    if not text:
        return ""
    if _ALREADY_SAYS_SOMETHING.match(text):
        return f"{text}."
    text = _LEADS_WITH_AN_ARTICLE.sub(lambda m: m.group(1).lower(), text, count=1)
    return f"It is {text}."
