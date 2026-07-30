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

    async def _run(extraction: LeadExtraction) -> Lead:
        call = Call(call_sid="sid", status=CallStatus.COMPLETED, started_at=MONDAY_CALL)
        call.id = "call-id"
        call.lead_id = None
        session = FakeSession(call, Transcript(call_id="call-id", full_text="Agent: hi\nProspect: hello"))

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
    lead = await run_worker(
        LeadExtraction(is_prospect=True, customer_name="Rahul", status=LeadStatus.WARM)
    )
    assert lead.site_visit_time is None
    assert lead.status is LeadStatus.WARM


async def test_partial_appointment_is_not_stored(run_worker):
    """Rahul's 'yeah sure' produced weekday=SATURDAY with no time."""
    lead = await run_worker(
        LeadExtraction(
            is_prospect=True, customer_name="Rahul", status=LeadStatus.WARM,
            site_visit_weekday=Weekday.SATURDAY, site_visit_at=None,
        )
    )
    assert lead.site_visit_time is None
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
        )
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
