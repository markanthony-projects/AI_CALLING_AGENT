"""Can the configured models answer? Asked with a completion, before anybody is dialled.

On 10 Sep 2026 the primary 404ed while models.list still returned it; on 14 Sep the .env
still named that model, and named a fallback its provider had shut down on 16 August. Two
dead models, discovered by a prospect being hung up on. The probe asks each endpoint for
one token at startup and on a timer, and the dialer refuses to place calls while neither
can answer.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
)

from app.core import llm_probe
from app.services.llm_provider import LLMEndpoint

PRIMARY = LLMEndpoint(
    name="cerebras",
    api_key="k",
    base_url="https://api.cerebras.ai/v1",
    model="gpt-oss-120b",
    reasoning_effort="low",
)
FALLBACK = LLMEndpoint(name="openai", api_key="k", base_url="https://api.openai.com/v1", model="gpt-4o-mini")


def _error(cls, status):
    request = httpx.Request("POST", "https://x/chat/completions")
    return cls("provider said no", response=httpx.Response(status, request=request), body=None)


class _FakeClient:
    """Records the request and answers, or raises, as told."""

    def __init__(self, raises=None):
        self.raises = raises
        self.kwargs = None
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        if self.raises:
            raise self.raises
        return SimpleNamespace(choices=[])

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    holder = {}

    def _make(raises=None):
        client = _FakeClient(raises)
        holder["client"] = client
        monkeypatch.setattr(llm_probe, "_client", lambda endpoint: client)
        return client

    return _make


# --- one endpoint, one verdict ----------------------------------------------------------------


def test_a_model_that_answers_is_ok(fake_client):
    client = fake_client()
    verdict = asyncio.run(llm_probe.probe_endpoint(PRIMARY, "primary"))
    assert verdict.status == llm_probe.OK and verdict.serviceable
    assert verdict.endpoint == "cerebras/gpt-oss-120b"
    assert client.closed


def test_the_probe_sends_what_a_real_turn_would_send(fake_client):
    """A reasoning_effort this model does not take is a 400 on every turn. A probe that
    omitted it would pass a configuration that fails live."""
    client = fake_client()
    asyncio.run(llm_probe.probe_endpoint(PRIMARY, "primary"))
    assert client.kwargs["model"] == "gpt-oss-120b"
    assert client.kwargs["reasoning_effort"] == "low"
    assert client.kwargs["max_tokens"] == 1


@pytest.mark.parametrize(
    "exc, status",
    [
        (_error(NotFoundError, 404), llm_probe.MODEL_NOT_FOUND),
        (_error(AuthenticationError, 401), llm_probe.UNAUTHORIZED),
        (_error(BadRequestError, 400), llm_probe.BAD_REQUEST),
        (_error(RateLimitError, 429), llm_probe.RATE_LIMITED),
        (APIConnectionError(request=httpx.Request("POST", "https://x")), llm_probe.UNREACHABLE),
        (RuntimeError("something unforeseen"), llm_probe.UNREACHABLE),
    ],
)
def test_every_way_a_provider_can_say_no_is_named(fake_client, exc, status):
    fake_client(raises=exc)
    verdict = asyncio.run(llm_probe.probe_endpoint(PRIMARY, "primary"))
    assert verdict.status == status
    assert "provider said no" in verdict.detail or status == llm_probe.UNREACHABLE


def test_a_throttled_model_still_counts_as_one_that_can_serve(fake_client):
    """Busy is not gone. The per-turn fallover handles a 429 on a live call."""
    fake_client(raises=_error(RateLimitError, 429))
    assert asyncio.run(llm_probe.probe_endpoint(PRIMARY, "primary")).serviceable


# --- the decision -----------------------------------------------------------------------------


def _v(role, status):
    return llm_probe.Verdict(role, f"{role}/model", status)


def test_a_live_fallback_keeps_dialing_alive_when_the_primary_is_gone():
    assert llm_probe.can_serve([_v("primary", llm_probe.MODEL_NOT_FOUND), _v("fallback", llm_probe.OK)])


def test_two_dead_models_stop_dialing():
    assert not llm_probe.can_serve(
        [_v("primary", llm_probe.MODEL_NOT_FOUND), _v("fallback", llm_probe.MODEL_NOT_FOUND)]
    )


def test_no_fallback_configured_means_the_primary_alone_decides():
    assert llm_probe.can_serve([_v("primary", llm_probe.OK)])
    assert not llm_probe.can_serve([_v("primary", llm_probe.UNAUTHORIZED)])


# --- published where the dialer can read it ---------------------------------------------------


class FakeRedis:
    def __init__(self, fail=False):
        self.hash = {}
        self.ttl = None
        self.fail = fail

    def _check(self):
        if self.fail:
            raise RuntimeError("arq pool is not initialised")

    async def hset(self, key, mapping):
        self._check()
        self.hash.update({k: str(v) for k, v in mapping.items()})

    async def expire(self, key, ttl):
        self._check()
        self.ttl = ttl

    async def hgetall(self, key):
        self._check()
        return dict(self.hash)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(llm_probe, "get_arq_pool", lambda: fake)
    return fake


def test_the_verdict_is_published_with_a_ttl(redis):
    asyncio.run(llm_probe.publish([_v("primary", llm_probe.OK), _v("fallback", llm_probe.MODEL_NOT_FOUND)]))
    assert redis.hash["serviceable"] == "1"
    assert redis.hash["primary"] == "ok"
    assert redis.hash["fallback"] == "model_not_found"
    assert redis.ttl == llm_probe._TTL


def test_two_dead_models_publish_not_serviceable(redis):
    asyncio.run(llm_probe.publish([_v("primary", llm_probe.MODEL_NOT_FOUND)]))
    assert asyncio.run(llm_probe.serviceable()) is False


def test_no_verdict_yet_permits_dialing(redis):
    """Unknown is not a refusal. Refusing on missing telemetry would take the campaign down
    every time Redis blinked, and the per-call fallover still stands behind this."""
    assert asyncio.run(llm_probe.serviceable()) is True


def test_an_unreadable_redis_permits_dialing(monkeypatch):
    monkeypatch.setattr(llm_probe, "get_arq_pool", lambda: FakeRedis(fail=True))
    assert asyncio.run(llm_probe.serviceable()) is True
    assert asyncio.run(llm_probe.status()) == {}


def test_publishing_never_raises(monkeypatch):
    monkeypatch.setattr(llm_probe, "get_arq_pool", lambda: FakeRedis(fail=True))
    asyncio.run(llm_probe.publish([_v("primary", llm_probe.OK)]))  # no exception is the test


# --- the gates that read it --------------------------------------------------------------------


def test_the_pump_places_nothing_while_no_model_can_answer(monkeypatch):
    from app.services import dial_pump

    monkeypatch.setattr(dial_pump, "is_within_calling_hours", lambda now: True)

    async def _down():
        return False

    monkeypatch.setattr(dial_pump.llm_probe, "serviceable", _down)

    async def _must_not_be_asked():
        raise AssertionError("the slot count was read for a dial that must not be placed")

    monkeypatch.setattr(dial_pump.call_slots, "free_slots", _must_not_be_asked)
    monkeypatch.setattr(dial_pump, "_llm_down_last_warned", 0.0)
    assert asyncio.run(dial_pump.dial_due_contacts()) == 0


def test_the_manual_dial_route_says_why_it_refused(monkeypatch):
    from app.core import ratelimit

    async def _down():
        return False

    monkeypatch.setattr(ratelimit.llm_probe, "serviceable", _down)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert refused.value.status_code == 503
    assert "LLM_MODEL" in refused.value.detail


def test_health_reports_the_verdict_or_unknown(client):
    """No Redis under the test client, so nothing has been published: the container is
    healthy and the LLM verdict is honestly unknown rather than invented."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"]["primary"] == "unknown"
    assert body["llm"]["serviceable"] == "unknown"


# --- the log lines somebody will read ---------------------------------------------------------


def test_the_report_names_the_way_out(capsys):
    from loguru import logger

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="INFO", format="{level} {message}")
    llm_probe.report([_v("primary", llm_probe.MODEL_NOT_FOUND), _v("fallback", llm_probe.MODEL_NOT_FOUND)])
    out = capsys.readouterr().out
    assert "cannot answer (model_not_found)" in out
    assert "Dialing is stopped" in out
    assert "LLM_MODEL / LLM_FALLBACK_MODEL" in out
