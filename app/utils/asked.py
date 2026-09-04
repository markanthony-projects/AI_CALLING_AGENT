"""What the agent has already asked, and what to do instead of asking it again.

From a live call on 3 Sep 2026, the agent asked "Would you like to visit the site and see
it once?" nineteen times. The prospect eventually said:

    "Apart from visit, you are not telling any details."
    "No, I'm not interested."

A three-crore lead, lost to a question. The model has no memory of the conversation beyond
the transcript it re-reads every turn, and underneath 2,686 words of rules it stops being
able to tell what it has already done from what it is about to do.

This is the cheap half of a fix. It tracks only what WE said, which needs no model and
cannot be wrong: the agent's own last question is already parsed for the dead-air nudge,
and the same parse answers "which of a handful of things did we just ask about". What the
prospect ANSWERED is a harder question that needs an extractor and can be stale, so it is
deliberately not here.

The second half matters as much as the first. Telling a model "do not ask that again" and
nothing else leaves it with no move — and on that call the repetition WAS the absence of a
move, because the script's only close is a site visit. So every topic carries what to do
instead. A rule with no alternative is a rule that gets broken or goes quiet.

Nothing is reported until a topic has been asked twice. Asking once is the conversation
working; asking twice is the first one not having landed, and that is the moment worth
spending tokens on. Until then this costs nothing at all.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from app.utils.reprompt import last_question

# Asked once, the prospect answers. Asked twice, the first one did not land and a third is
# the pattern that lost the call above.
REPEAT_LIMIT = 2


@dataclass(frozen=True)
class Topic:
    """One thing the agent asks about, and the way out when asking stops working."""

    key: str
    label: str
    pattern: re.Pattern
    instead: str


def _t(key: str, label: str, pattern: str, instead: str) -> Topic:
    return Topic(key, label, re.compile(pattern, re.I), instead)


# Matched against the QUESTION at the end of the agent's turn, never the whole reply — the
# script allows one question per reply, and an acknowledgement that happens to contain the
# word "budget" is not a request for one.
TOPICS: Sequence[Topic] = (
    _t(
        "site_visit",
        "a site visit",
        r"\bvisit\b|\bsee it\b|\bcome (?:and )?(?:see|have a look)\b",
        "offer to send the brochure, floor plans and price details on WhatsApp, then "
        "thank them and call end_call",
    ),
    _t(
        "budget",
        "their budget",
        r"\bbudget\b|\bprice range\b|\bhow much are you (?:looking|planning) to spend\b",
        "drop it and move on. Our team can come back to them with options at any price, "
        "and a number is not worth the call",
    ),
    _t(
        "timeline",
        "when they plan to buy",
        r"\bwhen are you\b|\bwhen do you\b|\bhow soon\b|\btimeline\b",
        "drop it and move on to what they would want, not when",
    ),
    _t(
        "purpose",
        "whether it is for their own stay or an investment",
        r"\bown stay\b|\bfor investment\b|\bstay in\b|\byourself or\b",
        "drop it and move on — either answer leads to the same next question",
    ),
    _t(
        "location",
        "which area they are looking in",
        r"\bwhich area\b|\bwhich part\b|\bwhereabouts\b|\blooking in\b",
        "drop it and offer to have our property expert send options on WhatsApp",
    ),
    _t(
        "unit_type",
        "which kind of home they want",
        r"\bapartment\b.*\bvilla\b|\bvilla\b.*\bplot\b|\b\d\s*BHK\b.*\bor\b",
        "drop it and describe what the project has instead of asking them to choose",
    ),
    _t(
        "callback",
        "a callback time",
        r"\bcall (?:you )?back\b|\bcall at\b|\bbetter time to call\b|\bcall you (?:later|tomorrow)\b",
        "stop proposing times. Say our team will reach out, and call end_call",
    ),
    _t(
        "name",
        "their name",
        r"\bgood name\b|\byour name\b|\bmay I know who\b",
        "carry on without it rather than asking a third time",
    ),
)


def topic_of(agent_line: Optional[str]) -> Optional[Topic]:
    """Which topic the agent's last question was about, or None if it asked nothing."""
    question = last_question(agent_line)
    if not question:
        return None
    return next((t for t in TOPICS if t.pattern.search(question)), None)


class AskedSoFar:
    """A running count of what the agent has asked about on this call.

    Held in the call's own closure rather than Redis or the database. The call is one
    process, one websocket and about three minutes; a network hop on the turn path to store
    something nothing else will ever read would cost latency for no reader.
    """

    def __init__(self, limit: int = REPEAT_LIMIT):
        self._limit = limit
        self._counts: Dict[str, int] = {}

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def record(self, agent_line: Optional[str]) -> Optional[Topic]:
        """Note what that turn asked about. Returns the topic, for the log line."""
        topic = topic_of(agent_line)
        if topic:
            self._counts[topic.key] = self._counts.get(topic.key, 0) + 1
        return topic

    def overasked(self) -> Sequence[Topic]:
        return [t for t in TOPICS if self._counts.get(t.key, 0) >= self._limit]

    def brief(self) -> str:
        """The block the model is shown, or "" while the conversation is working.

        Empty until something has been asked twice, so an ordinary call carries none of
        this — the prompt is already resent in full on every turn and there is no room to
        spend tokens describing a problem that has not happened.
        """
        over = self.overasked()
        if not over:
            return ""
        lines = [
            "ALREADY ASKED — you have asked each of these more than once on this call. "
            "Asking again is what makes a prospect hang up. Do NOT ask it again; do the "
            "thing written beside it instead."
        ]
        for topic in over:
            lines.append(
                f"- {topic.label} (asked {self._counts[topic.key]} times) -> {topic.instead}."
            )
        return "\n".join(lines)
