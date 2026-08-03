import enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.db import LeadStatus, Purpose


class Weekday(str, enum.Enum):
    MONDAY = "MONDAY"
    TUESDAY = "TUESDAY"
    WEDNESDAY = "WEDNESDAY"
    THURSDAY = "THURSDAY"
    FRIDAY = "FRIDAY"
    SATURDAY = "SATURDAY"
    SUNDAY = "SUNDAY"


class LeadExtraction(BaseModel):
    """What the model reports about the call.

    Appointments are captured as intent — the weekday or day offset the prospect named,
    plus the time — never as a resolved date. Asking the model for calendar arithmetic
    turned "Sunday" into a Friday and invented an 11:00 slot nobody agreed to; the exact
    datetime is computed in code from the call's reference time instead.
    """

    is_prospect: bool = Field(default=False, description="True if the caller is a genuine prospect interested in real estate. False if spam, wrong number, or completely uninterested.")
    customer_name: Optional[str] = Field(None, description="Customer's name ALWAYS written in English/Latin script. Transliterate Indic scripts: कुंदन becomes 'Kundan', राहुल becomes 'Rahul'. Never output Devanagari here.")
    preferred_location: Optional[str] = Field(None, description="Locality or city the PROSPECT said they want. Null if only the Agent mentioned a location.")
    preferred_unit_type: Optional[str] = Field(None, description="Unit configuration the PROSPECT showed interest in, exactly as the project names it, e.g. '2 BHK', '3 BHK', 'Villament'. Set this when they accept or ask about a specific unit. Null if no particular unit was discussed or only the Agent named one without the prospect responding to it.")
    budget: Optional[float] = Field(None, description="Customer budget as a number in rupees. 75 Lakhs is 7500000, 1.5 Crores is 15000000.")
    purpose: Optional[Purpose] = Field(None, description="Why the PROSPECT is buying: SELF_USE if for themselves or their family to live in, INVESTMENT if to rent out or resell. Null unless they said which.")
    timeline: Optional[str] = Field(None, description="Purchase timeline in the prospect's own words")
    timeline_months: Optional[int] = Field(None, description="The prospect's purchase timeline as a whole number of months: 'in 2 months' is 2, 'next month' is 1, 'immediately' or 'this month' is 0, 'within a year' is 12, 'in 2 years' is 24. Null if they gave no timeframe.")

    site_visit_weekday: Optional[Weekday] = Field(None, description="Weekday the prospect named for a site visit, e.g. SUNDAY when they said 'Sunday works'. Null if they named no weekday.")
    site_visit_in_days: Optional[int] = Field(None, description="Days from today for a site visit when spoken relatively: 0 for 'today', 1 for 'tomorrow', 2 for 'day after tomorrow'. Null otherwise.")
    site_visit_at: Optional[str] = Field(None, description="Time agreed for the site visit in 24-hour HH:MM, e.g. '15:00' for 3 PM. Null if no specific time was agreed.")

    callback_weekday: Optional[Weekday] = Field(None, description="Weekday the prospect named for a callback. Null if they named no weekday.")
    callback_in_days: Optional[int] = Field(None, description="Days from today for a callback when spoken relatively: 0 for 'today', 1 for 'tomorrow'. Null otherwise.")
    callback_at: Optional[str] = Field(None, description="Time agreed for the callback in 24-hour HH:MM, e.g. '18:00' for 6 PM. Null if no specific time was agreed.")

    status: Optional[LeadStatus] = Field(None, description="Lead status: HOT, WARM, or COLD")
    transliterated_transcript: Optional[str] = Field(default="", description="The entire conversation transcript completely transliterated into English/Latin script (e.g. Hindi written in English letters).")


class ProjectFilterParams(BaseModel):
    city: Optional[str] = None
    locality: Optional[str] = None
    min_budget: Optional[float] = None
    max_budget: Optional[float] = None
