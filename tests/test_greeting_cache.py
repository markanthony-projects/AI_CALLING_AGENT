"""The greeting synthesised while the phone rings is the greeting the agent will ask for.

The cache is keyed by sentence. If the worker and the agent cut or build the line
differently by one character, every call misses silently and nothing is faster. So the
first tests here are about the keys; the rest are about failing quietly.
"""

import asyncio
import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.services import greeting_cache as gc
from app.services.agent import build_opening_line, spoken
from app.utils.opening_line import build_opening_line as moved_builder

MORNING = datetime(2026, 9, 15, 4, 30, tzinfo=timezone.utc)  # 10:00 IST
PROJECT = {"name": "Abhee New Dimension", "developer_name": "Abhee", "agent_name": "Priya"}


def _settings(**over):
    base = dict(
        SARVAM_API_KEY="k",
        SARVAM_VOICE_ID="simran",
        SPEAKING_PACE=1.0,
        SARVAM_TEMPERATURE=None,
        GREETING_PRIME=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Redis:
    def __init__(self):
        self.store = {}

    async def setex(self, key, ttl, value):
        self.store[key] = (ttl, value)

    async def get(self, key):
        return self.store.get(key, (None, None))[1]

    async def delete(self, key):
        self.store.pop(key, None)


def _wav(pcm: bytes) -> bytes:
    return b"RIFF" + b"\x00" * 40 + pcm


class _Client:
    """httpx.AsyncClient stand-in that records requests and answers with a fixed wave."""

    def __init__(self, pcm=b"\x01\x02" * 400, status=200, raise_=None):
        self.requests = []
        self.pcm = pcm
        self.status = status
        self.raise_ = raise_

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        if self.raise_:
            raise self.raise_
        self.requests.append((url, json, headers))
        body = {"audios": [base64.b64encode(_wav(self.pcm)).decode()]}
        return httpx.Response(self.status, json=body)


@pytest.fixture
def redis(monkeypatch):
    fake = _Redis()

    async def client():
        return fake

    monkeypatch.setattr(gc, "get_redis_client", client)
    return fake


# ------------------------------------------------------------------ the keys


def test_the_builders_the_agent_exports_are_the_moved_ones():
    assert build_opening_line is moved_builder


def test_the_cached_sentences_are_exactly_what_the_agent_will_queue():
    line = build_opening_line(
        PROJECT["name"], "Rahul", MORNING, developer_name="Abhee", agent_name="Priya"
    )
    queued = [f.text for f in spoken(line, append_to_context=False)]
    assert gc.opening_sentences(PROJECT, "Rahul", MORNING) == queued
    assert len(queued) >= 2


def test_the_agent_builds_the_line_with_the_same_arguments():
    """startup_greeting passes project, customer, developer and agent name and lets the
    builder take the time itself. So does the cache — a `now` passed on one side only
    would shift the greeting by a timezone."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    call = src[src.index("opening_line = build_opening_line(") :]
    call = call[: call.index(")") + 1]
    assert "now" not in call
    assert "developer_name=developer_name" in call and "agent_name=agent_name" in call


# ------------------------------------------------------------------ the request


def test_the_rest_request_matches_the_websocket_config():
    """The greeting must be in the same voice as the rest of the call."""
    from app.services.voice import build_tts

    settings = _settings(SARVAM_TEMPERATURE=0.3)
    live = build_tts(SimpleNamespace(**vars(settings), TTS_SPARE_SOCKET=False))._config_payload()
    rest = gc.payload("Hello, Good morning.", settings)

    assert rest["speaker"] == live["speaker"]
    assert rest["pace"] == live["pace"]
    assert rest["model"] == live["model"]
    assert rest["target_language_code"] == live["target_language_code"]
    assert rest["enable_preprocessing"] == live["enable_preprocessing"]
    assert rest["temperature"] == live["temperature"] == 0.3
    # The live socket learns its rate from the StartFrame — the transport's 16kHz output —
    # so before start() it still says the constructor default. The REST request has to
    # say what the transport plays, which the next test pins in the agent.
    assert rest["sample_rate"] == 16000


def test_the_rate_is_the_rate_the_transport_plays():
    import inspect

    from app.services import agent
    from app.utils import primed_speech

    assert "audio_out_sample_rate=16000" in inspect.getsource(agent.run_voice_agent)
    assert gc.SAMPLE_RATE == primed_speech.SAMPLE_RATE == 16000


def test_temperature_stays_out_of_the_request_when_unset():
    assert "temperature" not in gc.payload("Hello.", _settings())


def test_the_text_gets_the_same_dash_treatment_as_the_engine_input():
    from app.utils.dashes import spoken_punctuation

    text = "Hello — Good morning."
    assert gc.payload(text, _settings())["text"] == spoken_punctuation(text)


def test_the_wave_header_is_stripped():
    pcm = b"\x10\x20" * 100
    data = {"audios": [base64.b64encode(_wav(pcm)).decode()]}
    assert gc.pcm_from_response(data) == pcm
    assert gc.pcm_from_response({"audios": []}) is None


# ------------------------------------------------------------------ round trip


def test_prime_then_recall_round_trips_every_sentence(monkeypatch, redis):
    client = _Client()
    monkeypatch.setattr(gc.httpx, "AsyncClient", lambda **kw: client)

    cached = asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings()))
    expected = gc.opening_sentences(PROJECT, "Rahul")
    assert cached == len(expected)
    assert [r[1]["text"] for r in client.requests] == [
        gc.spoken_punctuation(s) for s in expected
    ]
    assert all(r[2]["api-subscription-key"] == "k" for r in client.requests)

    ttl, raw = redis.store["greeting:c1"]
    assert ttl == gc._TTL_SECONDS
    assert set(json.loads(raw)) == set(expected)

    primed = asyncio.run(gc.recall_primed_greeting("c1"))
    assert primed == {s: client.pcm for s in expected}
    assert "greeting:c1" not in redis.store, "read once, then gone"
    assert asyncio.run(gc.recall_primed_greeting("c1")) == {}


def test_switched_off_means_no_request_and_nothing_stored(monkeypatch, redis):
    client = _Client()
    monkeypatch.setattr(gc.httpx, "AsyncClient", lambda **kw: client)
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings(GREETING_PRIME=False))) == 0
    assert client.requests == [] and redis.store == {}


def test_a_refusal_from_the_voice_engine_is_a_miss_not_an_error(monkeypatch, redis):
    monkeypatch.setattr(gc.httpx, "AsyncClient", lambda **kw: _Client(status=429))
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings())) == 0
    assert redis.store == {}


def test_a_network_failure_never_raises_into_the_dialer(monkeypatch, redis):
    monkeypatch.setattr(
        gc.httpx, "AsyncClient", lambda **kw: _Client(raise_=httpx.ConnectError("down"))
    )
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings())) == 0


def test_a_redis_that_is_down_never_raises_into_the_call(monkeypatch):
    async def broken():
        raise ConnectionError("redis down")

    monkeypatch.setattr(gc, "get_redis_client", broken)
    assert asyncio.run(gc.recall_primed_greeting("c1")) == {}


def test_the_dialer_primes_after_the_carrier_accepts_and_before_returning():
    """After the dial: a dial that is refused must not spend synthesis. Inside a try: the
    pump must never lose a dial to the voice engine."""
    import inspect

    from app.services import dial_pump

    src = inspect.getsource(dial_pump._place)
    assert src.index("trigger_vobiz_call(") < src.index("prime_greeting(")
    guard = src[src.rindex("try:", 0, src.index("prime_greeting(")) :]
    assert "except Exception" in guard


def test_the_worker_side_of_this_never_imports_pipecat():
    """The pump runs in the worker, which is kept free of the pipecat runtime."""
    import sys

    for name in ("app.services.greeting_cache", "app.utils.opening_line"):
        module = sys.modules[name]
        assert not any(
            getattr(v, "__module__", "").startswith("pipecat") for v in vars(module).values()
        ), f"{name} pulled pipecat in"
