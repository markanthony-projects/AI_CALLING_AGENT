"""The latency page's per-day series, filled day by day so an empty day is a gap, not a
missing tick — the same rule the calls-per-day chart already follows."""

from datetime import date
from types import SimpleNamespace

from app.api.routes.dashboard import LatencyDay, latency_days


def _row(day, **over):
    base = dict(day=day, calls=3, p50=1670.0, p95=1725.0, worst=2088, first_word=566.0, held=1, reconnects=0, failures=0)
    base.update(over)
    return SimpleNamespace(**base)


def test_every_day_in_the_window_is_present_oldest_first():
    today = date(2026, 9, 14)
    out = latency_days([_row("2026-09-13")], 3, today)
    assert [d.date for d in out] == ["2026-09-12", "2026-09-13", "2026-09-14"]


def test_a_day_with_calls_carries_rounded_medians():
    out = latency_days([_row("2026-09-14", p50=1670.4, first_word=565.6)], 1, date(2026, 9, 14))
    day = out[0]
    assert isinstance(day, LatencyDay)
    assert day.calls == 3 and day.p50_ms == 1670 and day.p95_ms == 1725
    assert day.worst_ms == 2088 and day.first_word_ms == 566 and day.held_replies == 1


def test_a_day_with_no_calls_is_zeros_and_nones_not_absent():
    out = latency_days([], 2, date(2026, 9, 14))
    assert all(d.calls == 0 and d.p50_ms is None and d.held_replies == 0 for d in out)


def test_the_route_is_registered_under_the_dashboard_session():
    from app.api.routes.dashboard import router

    paths = {r.path for r in router.routes}
    assert "/api/v1/dashboard/latency" in paths
