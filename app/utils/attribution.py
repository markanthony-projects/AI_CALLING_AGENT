"""Check that what we recorded about a prospect is something the prospect actually said.

Twice now the extractor has filled a lead with the agent's own pitch:

    Agent:    ...starting at 1.17 Crores.
    Prospect: Yeah, that lie in my budget.
    -> budget = 11700000              (they named no figure at all)

    Agent:    We are launching a new project in Varthur - Sarjapur Road
    -> preferred_location = 'Varthur - Sarjapur Road'   (they named no locality)

Both rules were already in the extraction prompt, in capitals, with worked examples:
"never copy one out of the Agent's pitch". gpt-4o-mini ignored them anyway. Sales then
calls a stranger back believing they can spend money they never mentioned, and the lead
sheet says they asked for an area they were only told about.

Prompting harder is not available — it has been tried. This is the same rule expressed as
code, so it can be tested, and so it cannot quietly stop working. The check runs after the
model, is deliberately conservative, and answers exactly one question: is there a Prospect
line this value could have come from?
"""

import re
from typing import Optional

_LAKH = 100_000.0
_CRORE = 10_000_000.0

# "60 lakhs", "1.17 Crores", "75L", "2cr"
_MONEY = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>crores?|cr\b|lakhs?|lacs?|lakh|l\b)",
    re.I,
)

# "70 to 80 lakhs" — the lower bound carries no unit of its own, so the single-value
# pattern above sees only "80 lakhs". Without this a prospect who gives a range is
# recorded at its top, or a model reporting the midpoint is thrown away as ungrounded.
_MONEY_RANGE = re.compile(
    r"(?P<low>\d+(?:[.,]\d+)?)\s*(?:to|-|–|—|and)\s*"
    r"(?P<high>\d+(?:[.,]\d+)?)\s*(?P<unit>crores?|cr\b|lakhs?|lacs?|lakh|l\b)",
    re.I,
)

# Long enough to be evidence. "the", "in", "for" match everything and prove nothing.
_MIN_TOKEN = 4
_TOKEN = re.compile(r"[a-z0-9]+")

# Long enough to clear the bar and still worthless as proof. "Varthur - Sarjapur Road"
# would otherwise be grounded by a prospect who said "the road was busy", which is the
# whole failure this module exists to catch, only subtler.
# Only words that are purely structural in an address. Compass points are deliberately NOT
# here: "South Bangalore" and "East Delhi" are how people name where they want to live, and
# a prospect who says "South Bangalore" against a model that writes "South Bengaluru" must
# still be believed — dropping a preference they really stated is its own bug.
_GENERIC = frozenset(
    {
        "road", "lane", "main", "cross", "near", "area", "side",
        "sector", "phase", "block", "layout", "extension",
        "circle", "junction", "stop", "station",
    }
)


def prospect_text(transcript: str) -> str:
    """Everything the prospect said, and nothing the agent did.

    The transcript is built as alternating "Agent: ..." / "Prospect: ..." lines. A line
    with no speaker prefix is treated as the agent's, because attributing unknown text to
    the prospect is the failure this module exists to prevent.
    """
    said = []
    for line in transcript.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("prospect:"):
            said.append(stripped.split(":", 1)[1].strip())
    return "\n".join(said)


def _multiplier(unit: str) -> float:
    return _CRORE if unit.lower().startswith("cr") else _LAKH


def money_in_rupees(text: str) -> list[float]:
    """Every sum of money named in `text`, in rupees.

    Both ends of a range are returned. The two patterns overlap on the range's upper
    figure, which is harmless — only the smallest and largest are ever used.
    """
    found = []
    for match in _MONEY_RANGE.finditer(text):
        scale = _multiplier(match.group("unit"))
        found.append(float(match.group("low").replace(",", ".")) * scale)
        found.append(float(match.group("high").replace(",", ".")) * scale)
    for match in _MONEY.finditer(text):
        found.append(float(match.group("value").replace(",", ".")) * _multiplier(match.group("unit")))
    return found


def budget_is_grounded(budget: Optional[float], transcript: str) -> bool:
    """True when the prospect named a figure this budget could have come from.

    A range counts as everything between its ends: "70 to 80 lakhs" is two numbers, and a
    model reporting 75,00,000 for it has read the prospect correctly rather than invented
    anything. A single figure has to match it.
    """
    if budget is None:
        return True
    amounts = money_in_rupees(prospect_text(transcript))
    if not amounts:
        return False
    # 1% for rounding between "1.17 Crores" and 11700000.
    return min(amounts) * 0.99 <= budget <= max(amounts) * 1.01


def phrase_is_grounded(value: Optional[str], transcript: str) -> bool:
    """True when some distinctive word of `value` appears in what the prospect said.

    Word-level rather than exact-match on purpose: the prospect says "South Bangalore" and
    the model writes "South Bengaluru", which is the same answer and must survive. What it
    will not survive is a locality that appears nowhere in their speech.
    """
    if not value:
        return True
    said = set(_TOKEN.findall(prospect_text(transcript).lower()))
    wanted = [
        t
        for t in _TOKEN.findall(value.lower())
        if len(t) >= _MIN_TOKEN and t not in _GENERIC
    ]
    if not wanted:
        # Nothing long enough to check — "2 BHK" is all short tokens. Unverifiable is not
        # the same as wrong, so it is left alone rather than discarded.
        return True
    return any(t in said for t in wanted)
