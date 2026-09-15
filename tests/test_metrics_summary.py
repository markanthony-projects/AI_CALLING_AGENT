"""Yesterday's number arrives uninvited, in one message a person reads in ten seconds."""

import asyncio
from datetime import date, datetime

import pytest

from app.services import metrics_summary as ms
from app.services.metrics_summary import DaySummary, ist_day_bounds, render


def _day(**over):
    base = dict(
        day=date(2026, 9, 14),
        calls=12,
        p50_ms=1670,
        p95_ms=1725,
        worst_ms=2088,
        first_word_ms=566,
        held_replies=1,
        tts_reconnects=0,
        llm_failures=0,
    )
    base.update(over)
    return DaySummary(**base)


def test_the_message_leads_with_the_count_and_the_typical_turn():
    text = render(_day())
    assert text.startswith("Calls Mon 14 Sep: 12 calls")
    assert "Typical turn p50 1,670 ms" in text
    assert "Typical worst turn p95 1,725 ms" in text
    assert "Slowest single turn 2,088 ms" in text
    assert "First word 566 ms" in text


def test_movement_against_the_previous_day_is_named():
    text = render(_day(), _day(day=date(2026, 9, 13), p50_ms=1400, p95_ms=1700))
    assert "p50 1,670 ms (+270 vs prev day)" in text
    assert "p95 1,725 ms (flat)" in text  # under the 50ms noise floor


def test_a_faster_day_reads_as_minus():
    text = render(_day(p50_ms=1300), _day(day=date(2026, 9, 13), p50_ms=1670))
    assert "(−370 vs prev day)" in text


def test_trouble_is_listed_only_when_there_was_some():
    assert "Trouble: 1 half-sentence replies held back" in render(_day())
    assert "Trouble: none" in render(_day(held_replies=0))
    text = render(_day(tts_reconnects=3, llm_failures=2))
    assert "3 voice reconnects" in text and "2 LLM failures" in text


def test_a_day_with_no_calls_says_so_in_one_line():
    assert render(_day(calls=0)) == "Calls Mon 14 Sep: none placed."


def test_a_missing_number_is_a_dash_not_a_crash():
    assert "p50 —" in render(_day(p50_ms=None))


def test_ist_day_bounds_are_naive_utc():
    """created_at is stored naive UTC; 14 Sep IST starts at 18:30 UTC on the 13th."""
    start, end = ist_day_bounds(date(2026, 9, 14))
    assert start == datetime(2026, 9, 13, 18, 30)
    assert end == datetime(2026, 9, 14, 18, 30)
    assert start.tzinfo is None


def test_nowhere_to_send_is_false_not_an_error(monkeypatch):
    monkeypatch.setattr(ms.settings, "METRICS_SUMMARY_WEBHOOK_URL", "")
    assert asyncio.run(ms.post("hello")) is False


def test_the_job_is_scheduled_for_nine_in_the_morning_ist():
    """arq's cron clock is UTC; 09:00 IST is 03:30 UTC."""
    from app.worker import WorkerSettings, morning_metrics_summary

    job = next(j for j in WorkerSettings.cron_jobs if j.coroutine is morning_metrics_summary)
    assert job.hour == {3} and job.minute == {30}


def test_the_summary_lines_are_visible_in_the_worker_log():
    from app.worker import _CALL_MODULES

    assert "app.services.metrics_summary" in _CALL_MODULES
