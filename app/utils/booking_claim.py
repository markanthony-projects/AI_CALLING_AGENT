"""A closing line that announces a booking the prospect never agreed to, caught before it is spoken.

Live call 2eeb48a0, 10 Sep 2026. The prospect's last words were "I said it is for
investment". The model called end_call with:

    "Thank you, Rahul. Your visit is confirmed for Saturday at 11 AM at Abhee Codename New
     Dimension. Have a great day."

No visit had been offered. The prospect heard a confirmation of an appointment they never
made, and the extraction worker then recorded it. The worker's side is fixed in
app/utils/attribution.py (day_is_grounded, time_is_grounded); this is the same check at
the other end of the call, on the sentence the prospect is about to hear.

Same question, same rule: is there a line the PROSPECT said that this day, or this hour,
could have come from? If not, the goodbye is said without the claim. Conservative in the
same way as the rest of attribution: a day or hour the prospect can be heard saying passes,
so a real read-back — "Perfect, Sunday at 3 PM" after they said "Sunday, 3 PM works" — is
untouched.
"""

import re
from typing import Iterable, Optional

from app.utils.attribution import day_is_grounded, time_is_grounded

_WEEKDAY = re.compile(
    r"(?<![a-z])(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?![a-z])", re.I
)
_RELATIVE = re.compile(r"(?<![a-z])(day after tomorrow|tomorrow|today|tonight)(?![a-z])", re.I)
# "11 AM", "11:30 am", "3 PM", "3pm". A bare number is not a time claim: "2 BHK", "3 acres".
_CLOCK = re.compile(r"(?<![0-9.])(\d{1,2})(?::(\d{2}))?\s*(am|pm)(?![a-z])", re.I)
_RELATIVE_DAYS = {"today": 0, "tonight": 0, "tomorrow": 1, "day after tomorrow": 2}

# How many times end_call is refused for announcing a booking the prospect never agreed.
# Once. Call 6102ee87, 14 Sep 2026: the prospect said "next weekend, like Saturday" and the
# model closed with "confirmed for Saturday at 11 AM" — no hour had been said. Saying goodbye
# without the claim, which is what happened before, still hangs up on somebody in the middle
# of booking a visit; the right move is to ask the one thing that is missing. Bounded like
# the repeat and bare-answer refusals: a guard meant to save a booking must not become a
# call nobody can leave, so the second time the plain farewell stands.
MAX_BOOKING_REFUSALS = 1

REFUSAL_REASON = (
    "Your closing line announces a day or time the prospect never said in their own words, "
    "so there is no booking yet. Do NOT end the call. Ask which day and what time would suit "
    "them, wait for the answer, read it back, and only then close."
)

# Spoken by the system when the leak path has no tool result to hand back to the model.
ASK_FOR_TIME = "Sorry, one small thing. Which day and what time would suit you for the visit?"


def _as_transcript(prospect_lines: Iterable[str]) -> str:
    """The prospect's words in the shape attribution reads — Prospect: lines and nothing else."""
    return "\n".join(f"Prospect: {line}" for line in prospect_lines if line)


def unagreed_booking(line: Optional[str], prospect_lines: Iterable[str]) -> Optional[str]:
    """The day or hour in `line` the prospect never said, or None if every claim is theirs.

    Returns the offending words so the log can say what was cut, not only that something
    was.
    """
    if not line:
        return None
    transcript = _as_transcript(prospect_lines)

    for match in _WEEKDAY.finditer(line):
        if not day_is_grounded(match.group(1).upper(), None, transcript):
            return match.group(0)
    for match in _RELATIVE.finditer(line):
        if not day_is_grounded(None, _RELATIVE_DAYS[match.group(1).lower()], transcript):
            return match.group(0)
    for match in _CLOCK.finditer(line):
        hour = int(match.group(1)) % 12
        if match.group(3).lower() == "pm":
            hour += 12
        if not time_is_grounded(f"{hour:02d}:{match.group(2) or '00'}", transcript):
            return match.group(0)
    return None
