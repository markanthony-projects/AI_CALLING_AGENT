"""What queueing a list of numbers actually does.

Both doors into the dialer now lead here — the dashboard route and the API-key route — so
this is the one place that decides whether a number gets called, and the one place that can
be wrong about it for both of them at once.

The fake session below stands in for Postgres. It is enough because everything worth pinning
here is a decision rather than a query: whether a paused campaign may be queued against, and
whether the count reported back is calls that will happen or rows that were written. The SQL
itself is covered by the predicate tests in test_dial_pump.py.
"""

import uuid

import pytest
from fastapi import HTTPException

from app.models.db import Campaign, CampaignStatus
from app.services import dial_queue


class Target:
    """Stands in for the DialEntry both routes parse into."""

    def __init__(self, number, name=None):
        self.number = number
        self.name = name


class Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeSession:
    """Answers the three queries enqueue makes, in the order it makes them:
    the suppression list, the insert, and the eligibility verdicts."""

    def __init__(self, campaign, suppressed=(), inserted=(), verdicts=()):
        self._campaign = campaign
        self._results = [Result(list(suppressed)), Result(list(inserted)), Result(list(verdicts))]
        self.committed = 0

    async def get(self, _model, _id):
        return self._campaign

    async def execute(self, _stmt):
        return self._results.pop(0)

    async def commit(self):
        self.committed += 1


def campaign(status=CampaignStatus.ACTIVE):
    return Campaign(id=uuid.uuid4(), name="Test", status=status)


@pytest.fixture(autouse=True)
def no_ceilings(monkeypatch):
    """The quota and budget gates talk to Redis and have their own tests."""

    async def allow(*_args, **_kwargs):
        return None

    monkeypatch.setattr(dial_queue, "reserve_dial_quota", allow)
    monkeypatch.setattr(dial_queue, "reserve_llm_headroom", allow)


# --- a campaign that is not running -----------------------------------------------------


async def test_a_paused_campaign_refuses_the_request():
    """Pausing has to actually stop it, or the button means nothing. Queueing against a
    paused campaign would report success and then never dial, which is indistinguishable
    from a broken dialer — the operator has no way to tell what happened."""
    db = FakeSession(campaign(CampaignStatus.PAUSED))
    with pytest.raises(HTTPException) as raised:
        await dial_queue.enqueue(db, uuid.uuid4(), [Target("+919876543210")], requested_by="test")
    assert raised.value.status_code == 409


async def test_a_paused_campaign_writes_nothing():
    """Refusing after the insert would leave a queue the pump drains the moment somebody
    un-pauses, which is not what the operator was told happened."""
    db = FakeSession(campaign(CampaignStatus.PAUSED))
    with pytest.raises(HTTPException):
        await dial_queue.enqueue(db, uuid.uuid4(), [Target("+919876543210")], requested_by="test")
    assert db.committed == 0


async def test_a_missing_campaign_is_a_404_not_a_crash():
    db = FakeSession(None)
    with pytest.raises(HTTPException) as raised:
        await dial_queue.enqueue(db, uuid.uuid4(), [Target("+919876543210")], requested_by="test")
    assert raised.value.status_code == 404


# --- what the report counts --------------------------------------------------------------


async def test_the_report_counts_calls_that_will_happen_not_rows_written(monkeypatch):
    """The distinction this whole report exists for. Two numbers are sent, one row is
    written because the other was already on the campaign — and both will be dialled."""
    monkeypatch.setattr(dial_queue, "dial_forecast", lambda _v: (2, {}))
    db = FakeSession(campaign(), inserted=["one-row"])

    report = await dial_queue.enqueue(
        db, uuid.uuid4(), [Target("+919876543210"), Target("+919876543211")], requested_by="test"
    )

    assert report.queued == 1, "one row was written"
    assert report.already_queued == 1
    assert report.will_dial == 2, "but both numbers are queued and both will be called"


async def test_a_list_that_will_place_no_calls_says_so(monkeypatch):
    """The failure that sent a dashboard dial to nobody: every number already terminal, one
    row written for none of them, and the response said queued."""
    monkeypatch.setattr(dial_queue, "dial_forecast", lambda _v: (0, {"COMPLETED": 1}))
    db = FakeSession(campaign(), inserted=[])

    report = await dial_queue.enqueue(
        db, uuid.uuid4(), [Target("+919876543210")], requested_by="test"
    )

    assert report.will_dial == 0
    assert report.held_back == {"COMPLETED": 1}


async def test_the_reason_survives_into_the_response(monkeypatch):
    """held_back is what points the operator at the row to retry; dropping it on the way out
    leaves them with a count and nowhere to go."""
    monkeypatch.setattr(dial_queue, "dial_forecast", lambda _v: (0, {"DND": 2}))
    db = FakeSession(campaign(), inserted=[])

    report = await dial_queue.enqueue(
        db, uuid.uuid4(), [Target("+919876543210")], requested_by="test"
    )
    assert report.as_response()["held_back"] == {"DND": 2}
    assert report.as_response()["will_dial"] == 0


async def test_a_suppressed_number_is_reported_as_suppressed(monkeypatch):
    """It is written as a DND row, so the insert counts it. Reporting it only as queued would
    tell the operator a do-not-call number is about to be dialled."""
    monkeypatch.setattr(dial_queue, "dial_forecast", lambda _v: (0, {"DND": 1}))
    db = FakeSession(campaign(), suppressed=["+919876543210"], inserted=["row"])

    report = await dial_queue.enqueue(
        db, uuid.uuid4(), [Target("+919876543210")], requested_by="test"
    )
    assert report.suppressed == 1
    assert report.will_dial == 0
