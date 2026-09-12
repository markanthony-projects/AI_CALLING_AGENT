"""The half of call 5023ff25 that was never fixed.

    USER  → "Hello."   TTS reconnected (3)   AGENT → "Hi, could you..."   [interrupted]
    USER  → "Hello."   TTS reconnected (4)   AGENT → "Which BHK size..."  [interrupted]
    (the agent goes mute for thirty seconds; the prospect hangs up)

app/utils/barge_in.py fixed the first half — "Hello?" should never have interrupted at all.
This is the second half, and it was a theory in a docstring until it was read out of pipecat:

    InterruptibleTTSService._handle_interruption:  await self._disconnect(); await self._connect()
    SarvamTTSService._connect_websocket, on error: push_error(...); self._websocket = None
    SarvamTTSService._get_websocket:              raise Exception("Websocket not connected")

One failed reconnect and nothing ever opens the socket again. The pipeline keeps running,
the model keeps replying, and none of it reaches the line.
"""

import asyncio
import inspect
from pathlib import Path

import pytest

from app.services.voice import MAX_REVIVALS, KeepsItsVoice

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")


def _voice():
    return KeepsItsVoice(
        api_key="test", settings=KeepsItsVoice.Settings(model="bulbul:v3", voice="simran")
    )


def _spy(service):
    """Replace connect and the base generator; what is under test is the guard."""
    calls = {"connect": 0, "spoke": 0}

    async def connect():
        calls["connect"] += 1
        service._websocket = object()

    async def spoke(text):
        calls["spoke"] += 1
        if False:
            yield None

    service._connect = connect
    return calls


async def _say(service, text="hello"):
    async for _ in service.run_tts(text):
        pass


# --- the failure this exists for --------------------------------------------------------


def test_a_closed_socket_is_reopened_before_speaking():
    async def run():
        service = _voice()
        calls = _spy(service)
        service._websocket = None

        async def fake_super(text):
            calls["spoke"] += 1
            if False:
                yield None

        import app.services.voice as voice

        original = voice.SarvamTTSService.run_tts
        voice.SarvamTTSService.run_tts = lambda self, text: fake_super(text)
        try:
            await _say(service)
        finally:
            voice.SarvamTTSService.run_tts = original
        assert calls["connect"] == 1, "the voice stayed closed"
        assert calls["spoke"] == 1, "it reconnected and then said nothing"

    asyncio.run(run())


def test_an_open_socket_is_left_alone():
    """Reconnecting a working socket would put a handshake in front of every sentence."""

    async def run():
        service = _voice()
        calls = _spy(service)
        service._websocket = object()

        import app.services.voice as voice

        original = voice.SarvamTTSService.run_tts

        async def fake_super(text):
            calls["spoke"] += 1
            if False:
                yield None

        voice.SarvamTTSService.run_tts = lambda self, text: fake_super(text)
        try:
            await _say(service)
        finally:
            voice.SarvamTTSService.run_tts = original
        assert calls["connect"] == 0
        assert calls["spoke"] == 1

    asyncio.run(run())


def test_it_gives_up_rather_than_hammering_a_dead_service():
    """A Sarvam that is genuinely down should not be dialled once per sentence for the rest
    of a call. Past the limit the error already on the wire is the more useful signal, and a
    caller hearing nothing is better served by the call ending."""

    async def run():
        service = _voice()
        calls = {"connect": 0}

        async def connect():
            calls["connect"] += 1  # deliberately does NOT open the socket

        service._connect = connect

        import app.services.voice as voice

        original = voice.SarvamTTSService.run_tts

        async def fake_super(text):
            if False:
                yield None

        voice.SarvamTTSService.run_tts = lambda self, text: fake_super(text)
        try:
            for _ in range(MAX_REVIVALS + 4):
                service._websocket = None
                await _say(service)
        finally:
            voice.SarvamTTSService.run_tts = original
        assert calls["connect"] == MAX_REVIVALS

    asyncio.run(run())


def test_the_limit_is_small_enough_to_be_a_limit():
    """Unbounded is the same as no limit, and this is a per-call counter."""
    assert 1 <= MAX_REVIVALS <= 5


def test_the_revivals_are_countable():
    service = _voice()
    assert service.revivals == 0


# --- and the alarm ----------------------------------------------------------------------


def test_the_agent_speaks_through_this_and_not_the_plain_service():
    assert "KeepsItsVoice(" in AGENT_SRC
    assert "SarvamTTSService(\n" not in AGENT_SRC


def test_a_run_of_reconnects_is_a_warning_and_not_a_note():
    """Four in fourteen seconds preceded the mute, and every one of them was logged at INFO
    among a thousand other INFO lines."""
    assert "TTS_RECONNECT_ALARM" in AGENT_SRC
    start = AGENT_SRC.index("if reconnects >= TTS_RECONNECT_ALARM")
    block = AGENT_SRC[start : start + 400]
    assert "logger.warning" in block
    assert "preceded a call going mute" in block


def test_the_alarm_threshold_is_below_what_actually_broke():
    """Four reconnects was the call that went mute. An alarm at four would have fired as the
    voice was already gone."""
    from app.services.agent import TTS_RECONNECT_ALARM

    assert TTS_RECONNECT_ALARM < 4


def test_the_reconnect_race_is_described_where_it_happens():
    """The mechanism was a theory for two days. It is read out of pipecat in the docstring
    now, so the next person does not have to rediscover it."""
    src = inspect.getsource(__import__("app.services.voice", fromlist=["KeepsItsVoice"]))
    assert "await self._disconnect()" in src
    assert "await self._connect()" in src
    assert "self._websocket = None" in src
    assert "Websocket not connected" in src
