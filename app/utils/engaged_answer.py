"""A yes is not a goodbye. The call does not end on it.

Call 96080daa, 15 Sep 2026, twenty-four seconds:

    AGENT -> "We are launching a new project in Varthur. Are you looking to buy a property?"
    USER  -> "Yeah. I was looking for property purchase."
    AGENT -> "Thank you for your time. Goodbye."   -> end_call, no closing_line, 6 reasoning tokens

The prompt forbids this in three places — "NEVER call end_call in the same turn that the
prospect agrees to something", the tool's own description, the opening gate — and the
model did it anyway, on the very sentence app/utils/bare_answer.py quotes from the call
that produced that module. bare_answer holds a one-word answer; seven words cleared it.

So this is the other half, in the same shape. A prospect who has just said yes, with no
refusal in the sentence, to a question that was not a closing offer, is still on the call.
The turn goes back to the model with the reason, once; the second attempt stands, as the
other bounded refusals do, so a guard meant to save a lead cannot become a call nobody can
get off.

One-sided on purpose. "Yes" to "Shall I send it on WhatsApp?" IS the close, and that is
the one case let through: the agent's last line names the close. Being wrong here costs
one more turn on a call that was ending anyway.
"""

import re
from typing import Optional

from app.utils.bare_answer import is_a_refusal

# What an engaged answer sounds like, in either language the calls are in.
_ENGAGED = re.compile(
    r"\b(?:yes|yeah|yep|yup|ya|haan|han|ji|sure|ok|okay|of course|definitely|certainly"
    r"|interested|looking|tell me|go ahead|please|batao|bataiye|bolo|boliye|bilkul)\b",
    re.I,
)
# The closes the script has. A yes to one of these is the end of the call, and rightly so.
_CLOSING_OFFER = re.compile(
    r"whatsapp|call you|property expert|expert (?:will|to) call|visit|come (?:and|to) see"
    r"|anything else", re.I,
)

REFUSAL_REASON = (
    "Do NOT end the call. The prospect just said YES to your question, and nothing has been "
    "closed — no WhatsApp, no callback, no visit was offered or agreed. A yes means the "
    "conversation continues: react to what they said and carry on with the next step of the "
    "script. Call end_call only when they decline, say goodbye, or a close is agreed."
)
# Once. On the second attempt the model has been told and is presumably reading something
# in the conversation this cannot.
MAX_EARLY_REFUSALS = 1


def said_yes_and_nothing_was_closed(
    last_prospect_line: Optional[str], last_agent_line: Optional[str]
) -> bool:
    """True when hanging up now would be hanging up on a yes."""
    text = (last_prospect_line or "").strip()
    if not text or is_a_refusal(text):
        return False
    if not _ENGAGED.search(text):
        return False
    if last_agent_line and _CLOSING_OFFER.search(last_agent_line):
        return False
    return True


def is_a_yes_to_a_close(last_prospect_line: Optional[str], last_agent_line: Optional[str]) -> bool:
    """A yes to "Shall I send it on WhatsApp?" is the close, however short it is.

    Call be096321, 15 Sep 2026: the agent offered the WhatsApp brochure, the prospect said
    "Yes.", and the one-word guard refused the hangup as "an answer, not a goodbye" — so
    the model invented a step ("Could you share the best WhatsApp number?") and the close
    slipped. The one-word rule is right everywhere except here.
    """
    text = (last_prospect_line or "").strip()
    if not text or not _ENGAGED.search(text):
        return False
    return bool(last_agent_line and _CLOSING_OFFER.search(last_agent_line))


def offered_a_close(line: Optional[str]) -> bool:
    """Whether the agent's line was one of the script's closing offers."""
    return bool(line and _CLOSING_OFFER.search(line))


CUT_OFF_REASON = (
    "Do NOT end the call. Your closing offer was cut off before the prospect heard it — "
    "their answer was to your PREVIOUS question, not to the offer. Ask the offer again, in "
    "full, and wait for their answer to it."
)
MAX_CUT_OFF_REFUSALS = 1
