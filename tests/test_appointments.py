"""Appointment resolution.

gpt-4o-mini was asked to do calendar arithmetic and produced, from a Monday call:
  "On Sunday" + "3PM"      -> Friday 2026-07-31 15:00   (two days early)
  "yeah sure" (no details) -> Thursday 2026-07-30 11:00 (entirely invented)

The model now reports only what was said; these tests pin the arithmetic that replaced it.
"""

from datetime import datetime, time

import pytest

from app.models.schemas import Weekday
from app.utils.timeutils import (
    is_within_business_hours,
    parse_time_of_day,
    resolve_appointment,
)

# The real call: Monday 27 July 2026, 16:41 IST
MONDAY = datetime(2026, 7, 27, 16, 41)


# --- the two production failures -------------------------------------------------


def test_sunday_3pm_lands_on_the_actual_sunday():
    got = resolve_appointment(MONDAY, weekday=Weekday.SUNDAY, time_of_day="15:00")
    assert got == datetime(2026, 8, 2, 15, 0)
    assert got.strftime("%A") == "Sunday"


def test_vague_agreement_creates_no_appointment():
    """'yeah sure' with no day and no time must never become a booking."""
    assert resolve_appointment(MONDAY) is None
    assert resolve_appointment(MONDAY, weekday=Weekday.SATURDAY) is None


def test_a_day_without_a_time_is_not_a_booking():
    """Defaulting the missing hour is what invented the 11:00 slot."""
    assert resolve_appointment(MONDAY, weekday=Weekday.SUNDAY, time_of_day=None) is None
    assert resolve_appointment(MONDAY, in_days=3, time_of_day=None) is None


def test_a_time_without_a_day_is_not_a_booking():
    assert resolve_appointment(MONDAY, time_of_day="15:00") is None


# --- weekday arithmetic ----------------------------------------------------------


@pytest.mark.parametrize(
    "weekday,expected",
    [
        (Weekday.TUESDAY, datetime(2026, 7, 28, 11, 0)),
        (Weekday.FRIDAY, datetime(2026, 7, 31, 11, 0)),
        (Weekday.SATURDAY, datetime(2026, 8, 1, 11, 0)),
        (Weekday.SUNDAY, datetime(2026, 8, 2, 11, 0)),
    ],
)
def test_each_weekday_resolves_forward_from_monday(weekday, expected):
    got = resolve_appointment(MONDAY, weekday=weekday, time_of_day="11:00")
    assert got == expected
    assert got.strftime("%A").upper() == weekday.value


def test_same_weekday_later_today_stays_today():
    got = resolve_appointment(MONDAY, weekday=Weekday.MONDAY, time_of_day="18:00")
    assert got == datetime(2026, 7, 27, 18, 0)


def test_same_weekday_already_past_rolls_to_next_week():
    """At 16:41, 'Monday at 11' cannot mean four hours ago."""
    got = resolve_appointment(MONDAY, weekday=Weekday.MONDAY, time_of_day="11:00")
    assert got == datetime(2026, 8, 3, 11, 0)


def test_weekday_string_is_accepted_as_well_as_enum():
    assert resolve_appointment(MONDAY, weekday="sunday", time_of_day="15:00") == datetime(2026, 8, 2, 15, 0)


def test_unknown_weekday_is_refused():
    assert resolve_appointment(MONDAY, weekday="someday", time_of_day="15:00") is None


# --- relative days ---------------------------------------------------------------


@pytest.mark.parametrize(
    "in_days,expected_day",
    [(0, 27), (1, 28), (2, 29), (7, 3)],
)
def test_relative_days(in_days, expected_day):
    got = resolve_appointment(MONDAY, in_days=in_days, time_of_day="11:00")
    assert got.day == expected_day


def test_relative_days_wins_over_weekday():
    """'tomorrow' is unambiguous; a weekday guess alongside it should not override it."""
    got = resolve_appointment(MONDAY, weekday=Weekday.SUNDAY, in_days=1, time_of_day="11:00")
    assert got == datetime(2026, 7, 28, 11, 0)


# --- time parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("15:00", time(15, 0)),
        ("3 PM", time(15, 0)),
        ("3pm", time(15, 0)),
        ("3:30 pm", time(15, 30)),
        ("11:00", time(11, 0)),
        ("11 AM", time(11, 0)),
        ("12 AM", time(0, 0)),
        ("12 PM", time(12, 0)),
        ("09:45", time(9, 45)),
    ],
)
def test_time_parsing(raw, expected):
    assert parse_time_of_day(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "sometime", "evening", "25:00", "10:99", "soon"])
def test_unusable_times_are_refused(raw):
    assert parse_time_of_day(raw) is None


# --- timezone handling -----------------------------------------------------------


def test_aware_reference_returns_naive_local_time():
    """leads.* are TIMESTAMP WITHOUT TIME ZONE holding IST wall-clock time."""
    from app.utils.timeutils import IST

    aware = MONDAY.replace(tzinfo=IST)
    got = resolve_appointment(aware, weekday=Weekday.SUNDAY, time_of_day="15:00")
    assert got.tzinfo is None
    assert got == datetime(2026, 8, 2, 15, 0)


def test_naive_and_aware_references_agree():
    from app.utils.timeutils import IST

    naive = resolve_appointment(MONDAY, weekday=Weekday.SUNDAY, time_of_day="15:00")
    aware = resolve_appointment(MONDAY.replace(tzinfo=IST), weekday=Weekday.SUNDAY, time_of_day="15:00")
    assert naive == aware


# --- business hours --------------------------------------------------------------


@pytest.mark.parametrize(
    "hour,inside", [(9, False), (10, True), (15, True), (19, True), (20, False), (3, False)]
)
def test_business_hours_window(hour, inside):
    assert is_within_business_hours(datetime(2026, 8, 2, hour, 0)) is inside
