"""Ceilings on outbound dialing.

The API key is a bearer secret. MAX_CALLS = 8 caps concurrent media streams, but that check
runs when the stream opens — long after Vobiz has dialled and started billing. Without a
ceiling in front of the dial, one leaked key could place thousands of billed calls of which
eight got audio.

One request carries up to MAX_DIAL_BATCH = 500 numbers, so the quota counts numbers.
"""

import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core import ratelimit


class FakeRedis:
    """Just the counter operations the limiter uses."""

    def __init__(self, fail=False):
        self.store = {}
        self.expires = {}
        self.fail = fail

    def _check(self):
        if self.fail:
            raise RedisConnectionError("redis is down")

    async def incrby(self, key, amount):
        self._check()
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]

    async def decrby(self, key, amount):
        self._check()
        self.store[key] = self.store.get(key, 0) - amount
        return self.store[key]

    async def expire(self, key, ttl):
        self._check()
        self.expires[key] = ttl


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_arq_pool", lambda: fake)
    return fake


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setattr(ratelimit.settings, "DIAL_MAX_PER_MINUTE", 30)
    monkeypatch.setattr(ratelimit.settings, "DIAL_MAX_PER_DAY", 100)


async def test_a_batch_within_both_ceilings_is_allowed(redis, limits):
    await ratelimit.reserve_dial_quota(10)
    minute_key, day_key = ratelimit.window_keys()
    assert redis.store[minute_key] == 10
    assert redis.store[day_key] == 10


async def test_the_quota_counts_numbers_not_requests(redis, limits):
    """A per-request limit would bound nothing: one request may carry 500 numbers."""
    await ratelimit.reserve_dial_quota(20)
    with pytest.raises(HTTPException) as exc:
        await ratelimit.reserve_dial_quota(20)
    assert exc.value.status_code == 429


async def test_an_oversized_batch_is_rejected_whole_and_consumes_nothing(redis, limits):
    """A rejected caller must not drain the window for everyone else."""
    with pytest.raises(HTTPException):
        await ratelimit.reserve_dial_quota(31)
    minute_key, _ = ratelimit.window_keys()
    assert redis.store.get(minute_key, 0) == 0
    await ratelimit.reserve_dial_quota(30)


async def test_rate_limit_carries_retry_after(redis, limits):
    with pytest.raises(HTTPException) as exc:
        await ratelimit.reserve_dial_quota(31)
    assert exc.value.headers["Retry-After"] == "60"


async def test_daily_ceiling_rejects_once_exhausted(redis, monkeypatch):
    """Minute limit raised out of the way so only the daily ceiling can bind."""
    monkeypatch.setattr(ratelimit.settings, "DIAL_MAX_PER_MINUTE", 10_000)
    monkeypatch.setattr(ratelimit.settings, "DIAL_MAX_PER_DAY", 100)
    for _ in range(5):
        await ratelimit.reserve_dial_quota(20)
    with pytest.raises(HTTPException) as exc:
        await ratelimit.reserve_dial_quota(20)
    assert "Daily" in exc.value.detail
    assert int(exc.value.headers["Retry-After"]) > 60


async def test_hitting_the_daily_ceiling_refunds_the_minute_budget(redis, limits):
    """The minute window accepted the batch before the day rejected it; keeping that charge
    would cost the caller the rest of their minute for a call that never happened."""
    monkey_day = ratelimit.window_keys()[1]
    redis.store[monkey_day] = 95
    with pytest.raises(HTTPException):
        await ratelimit.reserve_dial_quota(10)
    minute_key, _ = ratelimit.window_keys()
    assert redis.store.get(minute_key, 0) == 0


async def test_redis_down_refuses_to_dial(monkeypatch, limits):
    """Fails closed. Failing open would mean unbounded spend during a Redis blip."""
    monkeypatch.setattr(ratelimit, "get_arq_pool", lambda: FakeRedis(fail=True))
    with pytest.raises(HTTPException) as exc:
        await ratelimit.reserve_dial_quota(1)
    assert exc.value.status_code == 503


async def test_uninitialised_pool_refuses_to_dial(monkeypatch, limits):
    def boom():
        raise RuntimeError("arq pool is not initialised")

    monkeypatch.setattr(ratelimit, "get_arq_pool", boom)
    with pytest.raises(HTTPException) as exc:
        await ratelimit.reserve_dial_quota(1)
    assert exc.value.status_code == 503


async def test_counters_expire_so_windows_actually_roll(redis, limits):
    await ratelimit.reserve_dial_quota(1)
    minute_key, day_key = ratelimit.window_keys()
    assert redis.expires[minute_key] == ratelimit._MINUTE_TTL
    assert redis.expires[day_key] == ratelimit._DAY_TTL


def test_the_day_window_follows_ist_not_utc():
    """Sales counts "calls today" in IST; a UTC day would roll at 5:30 AM local."""
    from datetime import datetime
    from unittest.mock import patch

    with patch.object(ratelimit, "utc_now", return_value=datetime(2026, 7, 30, 20, 0, 0)):
        _, day_key = ratelimit.window_keys()
    assert day_key == "dial:day:2026-07-31", "20:00 UTC is already the 31st in IST"


def test_retry_after_midnight_is_bounded_to_a_day():
    assert 0 < ratelimit.seconds_to_ist_midnight() <= 86400


# --- wiring ----------------------------------------------------------------------


def _dial_tree():
    import ast
    import inspect

    from app.api.routes import campaign

    return ast.parse(inspect.getsource(campaign.dial_campaign_vobiz).lstrip())


def _quota_call():
    import ast

    for n in ast.walk(_dial_tree()):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "reserve_dial_quota":
            return n
    raise AssertionError("the dial endpoint never charges the quota")


def test_the_dial_endpoint_charges_the_quota():
    assert _quota_call() is not None


def test_it_charges_the_batch_size_not_one():
    """Charging per request would bound nothing: one request carries up to 500 numbers."""
    import ast

    arg = ast.unparse(_quota_call().args[0])
    assert arg == "len(req.phone_numbers)", f"quota charged as {arg!r}"


def test_the_charge_happens_before_anything_is_dialled():
    """Once trigger_vobiz_call runs, Vobiz has billed — a later check refunds nothing."""
    import ast

    tree = _dial_tree()
    quota = min(
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "reserve_dial_quota"
    )
    dial = min(
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "trigger_vobiz_call"
    )
    assert quota < dial, "the quota is charged after the money is already spent"
