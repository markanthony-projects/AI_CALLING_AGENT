"""The worker path from transcript to Lead row.

Covers the rules that are enforced in code rather than asked of the model, because the
model got them wrong on live calls: the appointment date, and the status of a prospect
who has actually booked a visit.
"""

from contextlib import asynccontextmanager
from datetime import datetime

import pytest

from app import worker
from app.models.db import Call, CallStatus, Lead, LeadStatus, Transcript
from app.models.db import Purpose
from app.models.schemas import LeadExtraction, Weekday

MONDAY_CALL = datetime(2026, 7, 27, 11, 11)  # 16:41 IST


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def first(self):
        return self._value


class FakeSession:
    """Returns the Call first, then the Transcript, mirroring process_extraction's order."""

    def __init__(self, call, transcript):
        self._queue = [call, transcript]
        self.added = []
        self.commits = 0

    async def execute(self, *a, **kw):
        return FakeResult(self._queue.pop(0) if self._queue else None)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for obj in self.added:
            if isinstance(obj, Lead) and obj.id is None:
                obj.id = "lead-id"

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


@pytest.fixture
def run_worker(monkeypatch):
    """Drive process_extraction with the DB and the LLM stubbed out."""

    async def _run(
        extraction: LeadExtraction, transcript: str = "Agent: hi\nProspect: hello"
    ) -> Lead:
        # The transcript is a real input now, not a placeholder: prospect-owned fields are
        # checked against it and dropped when no Prospect line says them. A test that wants
        # a budget or a locality to survive has to supply a caller who named one.
        call = Call(call_sid="sid", status=CallStatus.COMPLETED, started_at=MONDAY_CALL)
        call.id = "call-id"
        call.lead_id = None
        session = FakeSession(call, Transcript(call_id="call-id", full_text=transcript))

        @asynccontextmanager
        async def factory():
            yield session

        async def fake_extract(transcript, reference_time, call_sid):
            return extraction

        monkeypatch.setattr(worker, "AsyncSessionLocal", factory)
        monkeypatch.setattr(worker, "_extract_lead", fake_extract)

        await worker.process_extraction({}, "sid")
        leads = [o for o in session.added if isinstance(o, Lead)]
        return leads[0] if leads else None

    return _run


async def test_booked_site_visit_is_upgraded_to_hot(run_worker):
    """The model called Santosh WARM after he booked a Sunday 3PM visit."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Santosh", status=LeadStatus.WARM,
            site_visit_weekday=Weekday.SUNDAY, site_visit_at="15:00",
        )
    )
    assert lead.site_visit_time == datetime(2026, 8, 2, 15, 0)
    assert lead.status is LeadStatus.HOT


async def test_no_booking_keeps_the_model_status(run_worker):
    """The timeline is what makes WARM survivable: without a single stated fact the status is
    capped at COLD whatever the model claims, and this test would then be measuring the cap
    rather than the absence of a booking. A timeline rather than a budget because budget is
    checked against the transcript and would be dropped before the cap ever saw it."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul", status=LeadStatus.WARM,
            timeline_months=6,
        )
    )
    assert lead.site_visit_time is None
    assert lead.status is LeadStatus.WARM


async def test_partial_appointment_is_not_stored(run_worker):
    """Rahul's 'yeah sure' produced weekday=SATURDAY with no time."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul", status=LeadStatus.WARM,
            timeline_months=6,
            site_visit_weekday=Weekday.SATURDAY, site_visit_at=None,
        )
    )
    assert lead.site_visit_time is None
    # WARM survives because of the timeline, not the half-appointment: a weekday
    # with no hour is not a booking and must not raise the lead by itself.
    assert lead.status is LeadStatus.WARM


async def test_callback_resolves_independently_of_site_visit(run_worker):
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Asha", status=LeadStatus.WARM,
            callback_in_days=1, callback_at="18:00",
        )
    )
    assert lead.callback_time == datetime(2026, 7, 28, 18, 0)
    assert lead.site_visit_time is None
    assert lead.status is LeadStatus.WARM, "a callback is not a site visit"


async def test_non_prospect_creates_no_lead(run_worker):
    assert await run_worker(LeadExtraction(is_prospect=False)) is None


async def test_scalar_fields_are_carried_through(run_worker):
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Kundan", preferred_location="Whitefield",
            budget=15000000.0, timeline="3 months", status=LeadStatus.WARM,
        ),
        transcript=(
            "Agent: Which area are you looking in?\n"
            "Prospect: Whitefield, and my budget is 1.5 Crores.\n"
        ),
    )
    assert (lead.customer_name, lead.preferred_location) == ("Kundan", "Whitefield")
    assert float(lead.budget) == 15000000.0
    assert lead.timeline == "3 months"


# --- the schema instructions the model reads ------------------------------------


def test_customer_name_field_demands_latin_script():
    """It stored 'राहुल', which no CRM can search or dedupe on."""
    desc = LeadExtraction.model_fields["customer_name"].description
    assert "Latin" in desc
    assert "Transliterate" in desc, "the instruction must tell the model what to DO"
    assert "Rahul" in desc, "the concrete transliteration example is what made this stick"
    assert "Never output Devanagari" in desc


def test_appointment_fields_forbid_date_arithmetic():
    for field in ("site_visit_weekday", "callback_weekday"):
        desc = LeadExtraction.model_fields[field].description
        assert "Null if they named no weekday" in desc


def test_extraction_prompt_forbids_inventing_appointments():
    prompt = worker._build_system_prompt(datetime(2026, 7, 27, 16, 41))
    assert "NEVER invent an appointment" in prompt
    assert "Do NOT calculate any dates" in prompt


# --- the agent's pitch must not come back as the caller's requirements ---------------
#
# From a live call, verbatim:
#
#     Agent:    ...starting at 1.17 Crores.
#     Prospect: Yeah, that lie in my budget.
#
# The lead was written with budget=11700000 and preferred_location='Varthur - Sarjapur
# Road'. The prospect named neither. Sales then rings someone back believing they can spend
# money they never mentioned, about an area they were only told about. The extraction
# prompt forbids this in capitals with worked examples and gpt-4o-mini did it anyway, so
# the rule is enforced after the model rather than asked of it.

FABRICATION = (
    "Agent: We are launching a new project in Varthur - Sarjapur Road, near Dommasandra Circle.\n"
    "Prospect: Yeah, I was very much looking for that.\n"
    "Agent: Our project has 2, 3, 3.5 and 4.5 BHK configurations, starting at 1.17 Crores.\n"
    "Prospect: Yeah, that lie in my budget.\n"
)


async def test_the_asking_price_does_not_become_the_callers_budget(run_worker):
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul", budget=11700000.0,
            status=LeadStatus.WARM,
        ),
        transcript=FABRICATION,
    )
    assert lead.budget is None, "the caller never named a figure"


async def test_the_projects_own_locality_does_not_become_a_preference(run_worker):
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul",
            preferred_location="Varthur - Sarjapur Road", status=LeadStatus.WARM,
        ),
        transcript=FABRICATION,
    )
    assert lead.preferred_location is None


async def test_a_unit_type_only_the_agent_listed_is_not_a_preference(run_worker):
    """"We have 2, 3, 3.5 and 4.5 BHK" is stock being described. Picking one out of that
    list is inventing a preference."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, preferred_unit_type="3.5 BHK Presidential",
            status=LeadStatus.WARM,
        ),
        transcript=FABRICATION,
    )
    assert lead.preferred_unit_type is None


async def test_the_lead_is_still_created_when_fields_are_dropped(run_worker):
    """Dropping a fabricated field must not throw the whole lead away — they are still a
    prospect, and the call still happened."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul", budget=11700000.0,
            preferred_location="Varthur - Sarjapur Road", status=LeadStatus.WARM,
        ),
        transcript=FABRICATION,
    )
    assert lead is not None and lead.customer_name == "Rahul"


async def test_what_the_caller_really_said_survives(run_worker):
    """The guard has to be a scalpel. If it dropped genuine answers too it would be a
    worse bug than the one it replaces."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Chandan", budget=6000000.0,
            preferred_location="South Bengaluru", purpose=Purpose.INVESTMENT,
            timeline="in 2 months", timeline_months=2, status=LeadStatus.WARM,
        ),
        transcript=(
            "Agent: Which area are you looking in?\n"
            "Prospect: I am looking South Bangalore, for investment.\n"
            "Agent: What budget are you thinking of?\n"
            "Prospect: Below 60 lakhs, in 2 months.\n"
        ),
    )
    assert float(lead.budget) == 6000000.0
    assert lead.preferred_location == "South Bengaluru"
    assert lead.purpose is Purpose.INVESTMENT
    assert lead.timeline_months == 2
