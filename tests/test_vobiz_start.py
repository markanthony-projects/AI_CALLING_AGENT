"""The greeting waits for Vobiz to name the stream, and audio sent before that is counted.

Call 07709de3, 15 Sep 2026: greeting primed, socket warm, first word 246ms after the socket
opened — and twelve seconds of silence before the prospect hung up. Outbound audio carries
the streamId Vobiz sends in its "start" message; anything sent before that carries ours.
"""

import asyncio
import base64
import inspect
import json

import pytest
from pipecat.frames.frames import OutputAudioRawFrame

from app.utils.vobiz_serializer import VobizSerializer


def _audio():
    return OutputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1)


def test_the_start_message_names_the_stream_and_releases_the_wait():
    s = VobizSerializer(stream_sid="call-1")

    async def go():
        assert await s.wait_for_start(0.01) is False
        await s.deserialize(json.dumps({"event": "start", "streamId": "vobiz-9"}))
        assert await s.wait_for_start(0.01) is True
        out = json.loads(await s.serialize(_audio()))
        return out

    out = asyncio.run(go())
    assert out["streamId"] == "vobiz-9"
    assert s.sent_before_start == 0


def test_audio_before_start_goes_out_under_our_id_and_is_counted(monkeypatch):
    from app.utils import vobiz_serializer as module

    warnings = []
    monkeypatch.setattr(module.logger, "warning", lambda msg: warnings.append(msg))
    s = VobizSerializer(stream_sid="call-1")

    async def go():
        first = json.loads(await s.serialize(_audio()))
        await s.serialize(_audio())
        return first

    first = asyncio.run(go())
    assert first["streamId"] == "call-1"
    assert s.sent_before_start == 2
    assert len(warnings) == 1 and "before Vobiz's start event" in warnings[0]


def test_a_wait_that_is_already_satisfied_returns_at_once():
    s = VobizSerializer(stream_sid="call-1")
    asyncio.run(s.deserialize(json.dumps({"event": "start", "streamId": "v"})))
    assert asyncio.run(s.wait_for_start(0)) is True


def test_the_greeting_waits_before_it_is_queued_and_the_wait_is_bounded():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    greeting = src[src.index("async def startup_greeting") :]
    greeting = greeting[: greeting.index("await task.queue_frames(spoken(opening_line")]
    assert "await serializer.wait_for_start(STREAM_START_WAIT_SECS)" in greeting
    assert 'startup.mark("stream started")' in greeting
    assert 0.5 <= agent.STREAM_START_WAIT_SECS <= 3.0


def test_the_start_line_can_be_seen_in_the_call_log():
    from app.main import _CALL_MODULES

    assert "app.utils.vobiz_serializer" in _CALL_MODULES


@pytest.mark.parametrize("event", ["media", "stop", "unknown"])
def test_other_events_do_not_name_the_stream(event):
    s = VobizSerializer(stream_sid="call-1")
    asyncio.run(s.deserialize(json.dumps({"event": event, "streamId": "v", "media": {"payload": ""}})))
    assert not s.started.is_set()


# --- was there a voice on the line -------------------------------------------------------


def _media(samples):
    import struct

    pcm = b"".join(struct.pack("<h", s) for s in samples)
    return json.dumps({"event": "media", "media": {"payload": base64.b64encode(pcm).decode()}})


def test_a_silent_line_is_reported_as_silent():
    """Call ecf55487, 15 Sep 2026: frames at full rate, ten seconds of hello, no turn. The
    close line now says whether those frames carried a voice."""
    import base64  # noqa: F811

    s = VobizSerializer(stream_sid="call-1")
    assert s.inbound_report() == "IN: no audio frames"

    async def go():
        for _ in range(8):  # two sampled frames of near-silence
            await s.deserialize(_media([3, -4, 2] * 100))

    asyncio.run(go())
    assert "0/2 sampled frames with voice" in s.inbound_report()
    assert "peak -" in s.inbound_report()


def test_a_voice_on_the_line_is_counted():
    s = VobizSerializer(stream_sid="call-1")

    async def go():
        for _ in range(8):
            await s.deserialize(_media([9000, -9000, 400] * 100))

    asyncio.run(go())
    assert "2/2 sampled frames with voice" in s.inbound_report()


def test_the_report_is_on_the_close_line():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "SOCKET closed | {socket.report()} | {serializer.inbound_report()}" in src
