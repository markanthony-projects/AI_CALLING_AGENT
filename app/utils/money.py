"""Whether a figure the agent just spoke is one somebody actually gave it.

From a live call on 4 Sep 2026 (c085e397), after the prospect said they were looking in
North Bangalore rather than Varthur:

    USER  -> "I'm not sure of the budget. Can you tell me what is the possible budget
              I should have?"
    AGENT -> "That depends on the area and the size of the home you want. For a good
              2 or 3 BHK in North Bangalore, budgets usually start from 80 Lakhs and
              go up. When are you planning to buy?"

Eighty Lakhs came from nowhere. The campaign context describes one project, in Varthur,
and carries no pricing for any other market. The prompt already forbids this in as many
words -- "Never invent facts, prices, sizes or locations that are not in the campaign
context" -- and the model did it anyway, which is the whole reason this module exists:
a rule the model can ignore is not a control, it is a hope.

This reports rather than blocks, for the same reason the non-Latin script check does.
Removing "80 Lakhs" mid-sentence leaves "budgets usually start from and go up", which is
worse than the invention. What it buys is that the next one is visible in the logs
instead of being something only the prospect heard.

Grounding is membership, not range. The context for a 1.17-to-3.5-Crore project also
carries "20 to 30 Lakhs below the launch price", so anything between 0.2 and 3.5 Crores
would pass a range test -- including the 0.8 that started this. A spoken figure is
grounded when it matches a figure the context actually contains.
"""

import re
from typing import Iterable, List, Sequence

_LAKHS_PER_CRORE = 100.0

# Digits followed by a unit. "3 BHK", "6 months" and "2 or 3" carry no unit and are not
# money. Bare "cr" and "l" are matched even though the prompt forbids writing them,
# because this is here to catch the turns where the prompt was not followed.
_AMOUNT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>crores|crore|cr|lakhs|lakh|lacs|lac|l)\b",
    re.I,
)

# How far a spoken figure may sit from a context figure and still count as the same one.
# The agent is told to say "1.17 Crores", but a model rounding that to "1.2 Crores" has
# not invented anything. Five percent covers the rounding and nothing else: the 80 Lakhs
# that prompted this module is 32 percent away from the nearest figure it could have meant.
_TOLERANCE = 0.05


def amounts_in(text: str) -> List[float]:
    """Every money figure in the text, in Crores, in the order they appear."""
    found = []
    for match in _AMOUNT.finditer(text or ""):
        value = float(match.group("value"))
        unit = match.group("unit").lower()
        found.append(value if unit.startswith("cr") else value / _LAKHS_PER_CRORE)
    return found


def _matches_any(value: float, grounded: Iterable[float]) -> bool:
    """Whether a spoken figure is near enough to one the context actually carries.

    Measured against the context figure rather than the spoken one, because that is the
    number the tolerance is a tolerance ON: the question is how far the agent strayed from
    a real price, not how far the real price sits from the agent.

    That choice is for explainability, and no test pins it, because there is no honest test
    to write. The two divide differently only when the smaller figure is between 95 and
    95.2 percent of the larger -- a band a fifth of a percent wide, whose edges land on
    float rounding. A test sitting there would be measuring IEEE754 rather than this rule.
    """
    for known in grounded:
        if abs(value - known) <= _TOLERANCE * abs(known):
            return True
    return False


def ungrounded(spoken: str, grounded: Sequence[float]) -> List[float]:
    """The figures in `spoken` that `grounded` cannot account for, in Crores.

    Takes the grounded figures already parsed, because the caller is a frame processor: it
    runs on every streamed chunk of every reply, and re-reading the campaign context each
    time would put a regex over a few hundred characters of amenities and USPs on the path
    the caller is waiting on. Parsed once at construction instead.

    An empty `grounded` accounts for nothing, so every figure spoken against it is returned.
    That is the honest answer: an agent quoting prices for a project it was given no prices
    for is inventing them, and a call configured that way is the case worth hearing about.
    """
    spoke = amounts_in(spoken)
    if not spoke:
        return []
    return [value for value in spoke if not _matches_any(value, grounded)]


def ungrounded_amounts(spoken: str, campaign_context: str) -> List[float]:
    """The same question asked with the context still in string form."""
    return ungrounded(spoken, amounts_in(campaign_context))


def as_spoken(value: float) -> str:
    """A Crore figure written the way the logs and the agent both say it.

    Below a Crore it reads in Lakhs, which is how the figure was almost certainly said:
    "80 Lakhs" is what the agent invented, and a log line reporting "0.8 Crores" would not
    match the recording anyone goes back to check.
    """
    if value < 1:
        lakhs = value * _LAKHS_PER_CRORE
        return f"{f'{lakhs:.2f}'.rstrip('0').rstrip('.')} Lakhs"
    return f"{f'{value:.2f}'.rstrip('0').rstrip('.')} Crores"
