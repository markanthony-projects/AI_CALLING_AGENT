"""A voice socket opened at the answer webhook is the one the call speaks through.

The answer webhook arrives ~3s before the media stream. The handshake happens in that
window, the agent adopts the service when the stream opens, and if anything about that
went wrong the agent builds its own exactly as before.
"""

import asyncio
from types import SimpleNamespace

import pytest
from websockets.protocol import State

from app.services import warm_tts


class _Socket:
    def __init__(self, state=State.OPEN):
        self.state = state
        self.closed = False

    async def close(self):
        self.closed = True
        self.state = State.CLOSED


class _Service:
    """Just enough of KeepsItsVoice for the holder: a socket, and the two calls it makes."""

    def __init__(self, socket=None, fail=None):
        self._websocket = None
        self._speech_sample_rate = "0"
        self._socket = socket
        self._fail = fail
        self.connects = 0

    async def _connect_websocket(self):
        self.connects += 1
        if self._fail:
            raise self._fail
        self._websocket = self._socket

    async def _disconnect_websocket(self):
        if self._websocket:
            await self._websocket.close()
        self._websocket = None


@pytest.fixture(autouse=True)
def clean():
    warm_tts._warm.clear()
    warm_tts._pending.clear()
    yield
    warm_tts._warm.clear()
    warm_tts._pending.clear()


def _use(monkeypatch, service):
    monkeypatch.setattr(warm_tts, "build_tts", lambda settings: service)
    return service


def test_prepared_then_adopted_once(monkeypatch):
    service = _use(monkeypatch, _Service(_Socket()))

    async def go():
        assert await warm_tts.prepare("c1", SimpleNamespace()) is True
        assert warm_tts.waiting() == 1
        assert await warm_tts.adopt("c1") is service
        assert await warm_tts.adopt("c1") is None
        assert warm_tts.waiting() == 0

    asyncio.run(go())
    assert service.connects == 1


def test_the_socket_is_told_the_pipelines_sample_rate_before_it_connects(monkeypatch):
    """SarvamTTSService learns its rate from the StartFrame. There is no StartFrame yet."""
    seen = {}

    class Recording(_Service):
        async def _connect_websocket(self):
            seen["rate"] = self._speech_sample_rate
            await super()._connect_websocket()

    _use(monkeypatch, Recording(_Socket()))
    asyncio.run(warm_tts.prepare("c1", SimpleNamespace()))
    assert seen["rate"] == "16000"


def test_a_handshake_that_fails_leaves_nothing_behind(monkeypatch):
    _use(monkeypatch, _Service(fail=OSError("refused")))
    assert asyncio.run(warm_tts.prepare("c1", SimpleNamespace())) is False
    assert asyncio.run(warm_tts.adopt("c1")) is None


def test_a_handshake_that_quietly_left_no_socket_is_not_adopted(monkeypatch):
    """SarvamTTSService._connect_websocket swallows its own errors and sets None."""
    _use(monkeypatch, _Service(socket=None))
    assert asyncio.run(warm_tts.prepare("c1", SimpleNamespace())) is False


def test_a_socket_that_closed_while_waiting_is_not_adopted(monkeypatch):
    socket = _Socket()
    _use(monkeypatch, _Service(socket))
    asyncio.run(warm_tts.prepare("c1", SimpleNamespace()))
    socket.state = State.CLOSED
    assert asyncio.run(warm_tts.adopt("c1")) is None


def test_a_stream_that_never_opens_closes_the_socket(monkeypatch):
    socket = _Socket()
    _use(monkeypatch, _Service(socket))
    monkeypatch.setattr(warm_tts, "MAX_WAIT_SECS", 0.01)

    async def go():
        await warm_tts.prepare("c1", SimpleNamespace())
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert socket.closed
    assert warm_tts.waiting() == 0


def test_the_webhook_warms_after_replying_and_the_agent_adopts_it():
    import inspect

    from app.api.routes import webhook
    from app.services import agent

    answer = inspect.getsource(webhook.vobiz_answer)
    assert "warm_tts.begin(call_sid, settings)" in answer
    assert "if settings.TTS_PRECONNECT" in answer
    assert "warm_tts=await warm_tts.adopt(call_sid)" in inspect.getsource(webhook._handle_call)

    run = inspect.getsource(agent.run_voice_agent)
    assert 'startup.mark("tts (warm)")' in run
    assert "tts = build_tts(settings)" in run, "the cold path is the factory too"


class _Slow(_Service):
    """A handshake that takes a moment, like the real one."""

    def __init__(self, socket, delay):
        super().__init__(socket)
        self._delay = delay

    async def _connect_websocket(self):
        await asyncio.sleep(self._delay)
        await super()._connect_websocket()


def test_a_handshake_still_in_flight_is_waited_for(monkeypatch):
    """Call 81bdc87a: the stream opened while the socket was still connecting, the agent
    found nothing, built its own, and the warmed socket sat unused until it expired."""
    service = _use(monkeypatch, _Slow(_Socket(), delay=0.05))

    async def go():
        warm_tts.begin("c1", SimpleNamespace())
        await asyncio.sleep(0)  # the handshake has started, not finished
        assert warm_tts.waiting() == 0
        return await warm_tts.adopt("c1", wait=0.5)

    assert asyncio.run(go()) is service


def test_a_handshake_that_will_not_finish_in_time_is_a_cold_start(monkeypatch):
    _use(monkeypatch, _Slow(_Socket(), delay=0.5))

    async def go():
        warm_tts.begin("c1", SimpleNamespace())
        await asyncio.sleep(0)
        return await warm_tts.adopt("c1", wait=0.02)

    assert asyncio.run(go()) is None


def test_the_wait_is_shorter_than_the_cold_path_it_replaces():
    assert warm_tts.ADOPT_WAIT_SECS <= 0.4
