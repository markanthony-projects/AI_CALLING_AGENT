"""How warm a lead is allowed to be, given what the prospect actually told us.

A live call on 2 Sep 2026 produced this lead:

    Prospect: "Yes. Ok."
    Prospect: "Tell me tell me."
    -> status = WARM, budget=None purpose=None timeline=None unit=None area=None

Six words, every field empty, and the only two things on the row — the name and the number
— came off the dial list rather than out of the call. It then sat in the dashboard beside a
lead who had given a budget, a purpose, a timeline, a configuration and a booked callback,
wearing the same colour. That is what makes WARM useless: it stops meaning anything.

The extraction prompt does say a call where the prospect stated nothing is COLD. It said so
on this call too. Expressing it here as well, rather than only there, is the same move
_drop_ungrounded made for budget and locality after the prompt was ignored twice in
capitals: a rule that decides what sales sees has to be one that can be tested.

This is a CEILING, not a verdict. It only ever lowers. A model that says COLD about a
prospect who gave four facts has usually heard a refusal, and that judgement is left alone —
the failure being corrected runs one way, towards flattering the pipeline.
"""

from typing import Optional

from app.models.db import LeadStatus

# What counts as the prospect having told us something about themselves. Every one of these
# is a thing a rep can act on: what they can spend, why they are buying, when, what size,
# and where. Deliberately not customer_name or phone_number — those come from the dial list
# and are on the row whether the call happened or not.
QUALIFYING_FIELDS = (
    "budget",
    "purpose",
    "timeline",
    "timeline_months",
    "preferred_unit_type",
    "preferred_location",
)

# Coldest first, so a status can be compared against the ceiling by position.
_RANK = (LeadStatus.COLD, LeadStatus.WARM, LeadStatus.HOT)


def qualifying_facts(lead) -> list[str]:
    """Which of the prospect's own requirements this lead actually carries."""
    return [f for f in QUALIFYING_FIELDS if getattr(lead, f, None) not in (None, "")]


def status_ceiling(lead) -> LeadStatus:
    """The warmest status this lead's own contents can justify.

    A booking is a commitment and outranks everything: someone who agreed to a time has told
    us more than any answer could. Otherwise one stated requirement is the difference between
    a lead and a phone number that happened to pick up.
    """
    if lead.site_visit_time is not None or lead.callback_time is not None:
        return LeadStatus.HOT
    if qualifying_facts(lead):
        return LeadStatus.WARM
    return LeadStatus.COLD


def capped_status(claimed: Optional[LeadStatus], lead) -> LeadStatus:
    """`claimed`, lowered to whatever the lead's contents support.

    A missing claim becomes the ceiling itself. That replaces a flat default of WARM, which
    was reached on two of the first three production calls and was wrong on both: it was
    chosen when the alternative was a null status hiding the lead from every filter sales
    uses, and the ceiling answers that objection better — it is never null, and it is never
    warmer than the call earned.
    """
    ceiling = status_ceiling(lead)
    if claimed is None:
        return ceiling
    return claimed if _RANK.index(claimed) <= _RANK.index(ceiling) else ceiling
