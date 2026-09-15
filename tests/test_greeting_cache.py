"""The greeting synthesised while the phone rings is the greeting the agent will ask for,
in the voice the rest of the call will have.

The cache is keyed by sentence. If the worker and the agent cut or build the line
differently by one character, every call misses silently and nothing is faster. And it is
synthesised over the same websocket with the same config as the live call, because the
REST endpoint — same model, same speaker — came back about 4dB louder and the prospect
heard the call drop in volume after the first line. So the tests here are about the keys,
about the config being the live one to the byte, and about failing quietly.
"""

import asyncio
import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

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
        TTS_SPARE_SOCKET=False,
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


def _wav(pcm: bytes, rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """A real RIFF/WAVE file, header and all."""
    import struct

    block = channels * bits // 8
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", len(body)) + body


class _Socket:
    """Sarvam's websocket as the live call sees it: config, text, flush in; audio frames
    and a final event out."""

    sockets = []

    def __init__(self, pcm=b"\x01\x02" * 400, error=None, hang=False):
        self.sent = []
        self.pcm = pcm
        self.error = error
        self.hang = hang
        _Socket.sockets.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        if self.hang:
            await asyncio.sleep(10)
        if self.error:
            yield json.dumps({"type": "error", "data": {"message": self.error, "code": 422}})
            return
        half = len(self.pcm) // 2
        for piece in (self.pcm[:half], self.pcm[half:]):
            yield json.dumps({"type": "audio", "data": {"audio": base64.b64encode(piece).decode()}})
        yield json.dumps({"type": "event", "data": {"event_type": "final"}})


@pytest.fixture
def sockets(monkeypatch):
    _Socket.sockets = []
    factory = {"make": lambda: _Socket()}

    def connect(url, additional_headers=None):
        sock = factory["make"]()
        sock.url = url
        sock.headers = additional_headers
        return sock

    monkeypatch.setattr(gc, "connect", connect)
    return factory


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


# ------------------------------------------------------------------ the same voice


@pytest.mark.parametrize("temperature", [None, 0.4, 0.3])
def test_the_config_is_the_live_calls_config_to_the_byte(temperature):
    """The load-bearing one. KeepsItsVoice._config_payload() is what the live socket
    receives; the worker cannot import it, so live_config is a copy, and this holds them
    equal across every setting that reaches the voice."""
    from app.services.voice import build_tts

    settings = _settings(SARVAM_TEMPERATURE=temperature, SPEAKING_PACE=1.05)
    tts = build_tts(settings)
    tts._speech_sample_rate = str(gc.SAMPLE_RATE)  # what start() sets from the 16kHz transport
    assert gc.live_config(settings) == tts._config_payload()


def test_the_same_endpoint_and_query_as_pipecat():
    from app.services.voice import build_tts

    assert gc.WS_URL == build_tts(_settings())._websocket_url


def test_one_sentence_goes_config_text_flush_and_comes_back_as_pcm(sockets):
    pcm = asyncio.run(gc.synthesise("Hello, Good morning.", _settings(SARVAM_TEMPERATURE=0.4)))
    sock = _Socket.sockets[0]
    assert pcm == sock.pcm
    assert [m["type"] for m in sock.sent] == ["config", "text", "flush"]
    assert sock.sent[0]["data"] == gc.live_config(_settings(SARVAM_TEMPERATURE=0.4))
    assert sock.headers == {"api-subscription-key": "k"}


def test_the_text_gets_the_same_dash_treatment_as_the_engine_input(sockets):
    from app.utils.dashes import spoken_punctuation

    text = "Hello — Good morning."
    asyncio.run(gc.synthesise(text, _settings()))
    assert _Socket.sockets[0].sent[1]["data"]["text"] == spoken_punctuation(text)


def test_the_rate_is_the_rate_the_transport_plays():
    import inspect

    from app.services import agent
    from app.utils import primed_speech

    assert "audio_out_sample_rate=16000" in inspect.getsource(agent.run_voice_agent)
    assert gc.SAMPLE_RATE == primed_speech.SAMPLE_RATE == 16000
    assert gc.live_config(_settings())["speech_sample_rate"] == "16000"


def test_a_wave_header_if_one_ever_appears_is_read_not_assumed():
    pcm = b"\x10\x20" * 100
    assert gc.pcm_16k_mono(_wav(pcm)) == pcm
    assert gc.pcm_16k_mono(pcm) == pcm, "bare linear16 is what the socket sends"
    assert gc.pcm_16k_mono(b"") is None
    assert gc.wav_pcm(_wav(pcm, rate=22050)) == (pcm, 22050, 1, 16)


def test_audio_at_any_other_rate_is_a_miss_not_a_slow_deep_greeting():
    """Call 8571d93b, 15 Sep 2026: 22050Hz played at 16000 — the greeting came out 38%
    slower and five semitones down, in a voice nobody had chosen."""
    pcm = b"\x10\x20" * 100
    for wav in (_wav(pcm, rate=22050), _wav(pcm, rate=24000), _wav(pcm, channels=2), _wav(pcm, bits=8)):
        assert gc.pcm_16k_mono(wav) is None


# ------------------------------------------------------------------ round trip


def test_prime_then_recall_round_trips_every_sentence(sockets, redis):
    cached = asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings()))
    expected = gc.opening_sentences(PROJECT, "Rahul")
    assert cached == len(expected)
    assert len(_Socket.sockets) == len(expected), "one socket per sentence, in parallel"
    assert sorted(s.sent[1]["data"]["text"] for s in _Socket.sockets) == sorted(
        gc.spoken_punctuation(s) for s in expected
    )

    ttl, raw = redis.store["greeting:c1"]
    assert ttl == gc._TTL_SECONDS
    assert set(json.loads(raw)) == set(expected)

    primed = asyncio.run(gc.recall_primed_greeting("c1"))
    assert primed == {s: _Socket.sockets[0].pcm for s in expected}
    assert "greeting:c1" not in redis.store, "read once, then gone"
    assert asyncio.run(gc.recall_primed_greeting("c1")) == {}


def test_switched_off_means_no_socket_and_nothing_stored(sockets, redis):
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings(GREETING_PRIME=False))) == 0
    assert _Socket.sockets == [] and redis.store == {}


def test_a_refusal_from_the_voice_engine_is_a_miss_not_an_error(sockets, redis):
    sockets["make"] = lambda: _Socket(error="Input parameters has to be a valid dictionary")
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings())) == 0
    assert redis.store == {}


def test_a_socket_that_never_answers_is_given_up_on_within_the_budget(sockets, redis, monkeypatch):
    sockets["make"] = lambda: _Socket(hang=True)
    monkeypatch.setattr(gc, "_SYNTHESIS_BUDGET_SECS", 0.05)
    assert asyncio.run(gc.prime_greeting("c1", PROJECT, "Rahul", _settings())) == 0


def test_a_connection_failure_never_raises_into_the_dialer(redis, monkeypatch):
    def refused(url, additional_headers=None):
        raise OSError("refused")

    monkeypatch.setattr(gc, "connect", refused)
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
