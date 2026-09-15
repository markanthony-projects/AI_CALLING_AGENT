"""A barge-in swaps voice sockets instead of closing and reopening one.

Pipecat's InterruptibleTTSService tears the websocket down on every barge-in while the bot
is speaking and opens a new one — a 180ms handshake the prospect waits through, and the
churn that lost call 5023ff25 its voice. With a spare open, the spare becomes live and the
old socket is closed off the hot path. Without one, the old path runs unchanged.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.tts_service import InterruptibleTTSService, TTSService

from app.services import voice
from app.services.voice import KeepsItsVoice, build_tts


class _Socket:
    def __init__(self):
        from websockets.protocol import State

        self.state = State.OPEN
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        from websockets.protocol import State

        self.closed = True
        self.state = State.CLOSED


def _service(spare=True):
    service = KeepsItsVoice(
        api_key="k",
        spare_socket=spare,
        settings=KeepsItsVoice.Settings(model="bulbul:v3", voice="simran", pace=1.0),
    )
    service._speech_sample_rate = "16000"
    return service


def _quiet(monkeypatch, service):
    """Stand-ins for the task plumbing the swap uses, so the swap alone is under test."""
    tasks = []

    def create_task(coro, *a, **kw):
        coro.close()
        tasks.append("created")
        return SimpleNamespace(done=lambda: False, cancel=lambda: None)

    async def cancel_task(task, timeout=None):
        tasks.append("cancelled")

    async def stop_all_metrics():
        tasks.append("metrics stopped")

    monkeypatch.setattr(service, "create_task", create_task)
    monkeypatch.setattr(service, "cancel_task", cancel_task)
    monkeypatch.setattr(service, "stop_all_metrics", stop_all_metrics)
    monkeypatch.setattr(service, "_replenish_spare", lambda: tasks.append("replenish"))
    return tasks


@pytest.fixture
def interruptions(monkeypatch):
    """Records which base handler ran instead of running pipecat's internals."""
    ran = []

    async def base(self, frame, direction):
        ran.append("tts")

    async def interruptible(self, frame, direction):
        ran.append("interruptible")

    monkeypatch.setattr(TTSService, "_handle_interruption", base)
    monkeypatch.setattr(InterruptibleTTSService, "_handle_interruption", interruptible)
    return ran


# ------------------------------------------------------------------ the config


def test_the_spare_is_configured_exactly_as_pipecat_configures_the_live_socket():
    """Captured from the real _send_config, not restated."""
    service = KeepsItsVoice(
        api_key="k",
        settings=KeepsItsVoice.Settings(
            model="bulbul:v3", voice="simran", pace=1.05, temperature=0.4, max_chunk_length=150
        ),
    )
    service._speech_sample_rate = "16000"
    socket = _Socket()
    service._websocket = socket
    asyncio.run(service._send_config())
    sent = json.loads(socket.sent[0])
    assert sent["type"] == "config"
    assert sent["data"] == service._config_payload()
    assert sent["data"]["temperature"] == 0.4


def test_an_unset_temperature_stays_out_of_the_spare_config_too():
    assert "temperature" not in _service()._config_payload()


# ------------------------------------------------------------------ the swap


def test_a_barge_in_while_speaking_swaps_to_the_spare(monkeypatch, interruptions):
    service = _service()
    tasks = _quiet(monkeypatch, service)
    live, spare = _Socket(), _Socket()
    service._websocket, service._spare = live, spare
    service._receive_task = object()
    service._keepalive_task = object()
    service._bot_speaking = True

    async def go():
        await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)  # the old socket closes off the hot path

    asyncio.run(go())

    assert interruptions == ["tts"], "Interruptible's disconnect/connect must be skipped"
    assert service._websocket is spare
    assert service._spare is None
    assert service.swaps == 1
    assert live.closed
    assert tasks.count("cancelled") == 2 and tasks.count("created") == 2
    assert "metrics stopped" in tasks and tasks[-1] == "replenish"


def test_without_a_spare_the_old_path_runs(monkeypatch, interruptions):
    service = _service()
    _quiet(monkeypatch, service)
    service._websocket = _Socket()
    service._bot_speaking = True

    asyncio.run(service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM))

    assert interruptions == ["interruptible"]
    assert service.swaps == 0


def test_a_spare_that_has_closed_is_not_swapped_in(monkeypatch, interruptions):
    service = _service()
    _quiet(monkeypatch, service)
    service._websocket = _Socket()
    service._spare = _Socket()
    asyncio.run(service._spare.close())
    service._bot_speaking = True

    asyncio.run(service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM))

    assert interruptions == ["interruptible"]
    assert not service.spare_ready


def test_a_barge_in_while_silent_never_touches_the_sockets(monkeypatch, interruptions):
    service = _service()
    _quiet(monkeypatch, service)
    live, spare = _Socket(), _Socket()
    service._websocket, service._spare = live, spare
    service._bot_speaking = False

    asyncio.run(service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM))

    assert interruptions == ["interruptible"]
    assert service._websocket is live and service._spare is spare


# ------------------------------------------------------------------ lifecycle


def test_disconnect_closes_the_spare_as_well(monkeypatch):
    service = _service()
    spare = _Socket()
    service._spare = spare
    seen = []

    async def base_disconnect(self):
        seen.append("base")

    monkeypatch.setattr(SarvamTTSService, "_disconnect", base_disconnect)
    asyncio.run(service._disconnect())
    assert spare.closed and service._spare is None and seen == ["base"]


def test_connect_opens_a_spare_only_once_the_live_socket_is_up(monkeypatch):
    service = _service()
    calls = []
    monkeypatch.setattr(service, "_replenish_spare", lambda: calls.append("replenish"))

    async def base_connect(self):
        pass

    monkeypatch.setattr(SarvamTTSService, "_connect", base_connect)
    asyncio.run(service._connect())
    assert calls == [], "no live socket, no spare"
    service._websocket = _Socket()
    asyncio.run(service._connect())
    assert calls == ["replenish"]


def test_a_spare_that_cannot_be_opened_is_a_warning_not_a_failure(monkeypatch):
    service = _service()

    async def refused():
        raise OSError("refused")

    monkeypatch.setattr(service, "_open_socket", refused)
    asyncio.run(service._open_spare())
    assert service._spare is None and not service.spare_ready


def test_the_spare_is_off_unless_asked_for():
    """tests/test_voice_recovery.py builds the service bare; nothing there should open a
    second socket."""
    assert _service(spare=False)._spare_enabled is False
    assert KeepsItsVoice(api_key="k", settings=KeepsItsVoice.Settings(model="bulbul:v3"))._spare_enabled is False


def test_the_factory_reads_the_switch():
    settings = SimpleNamespace(
        SARVAM_API_KEY="k",
        SARVAM_VOICE_ID="simran",
        SPEAKING_PACE=1.0,
        SARVAM_TEMPERATURE=None,
        TTS_SPARE_SOCKET=False,
    )
    assert build_tts(settings)._spare_enabled is False
    settings.TTS_SPARE_SOCKET = True
    assert build_tts(settings)._spare_enabled is True
    assert isinstance(build_tts(settings), KeepsItsVoice)
    assert build_tts(settings).name == "SarvamTTSService", "the latency label must not change"


def test_the_switches_exist_with_rollback_defaults():
    from app.core.config import Settings

    for name in ("GREETING_PRIME", "TTS_PRECONNECT", "TTS_SPARE_SOCKET"):
        assert Settings.model_fields[name].default is True, name
