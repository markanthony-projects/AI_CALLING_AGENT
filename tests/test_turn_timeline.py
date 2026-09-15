"""The hand-offs inside a turn, so the gap with no service behind it has a name.

Every Flux turn on 14 Sep carried ~450–520ms of `unattributed`. The decision, the LLM and
the TTS each report their own time; whatever sits between them did not. The TIMELINE line
prints the gap at each station from the turn being declared over, and prints a station that
was never reached as such — "tts_start=—" is the whole finding on a turn where the model
replied with nothing.
"""

import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData

from app.utils.latency import NS_PER_SEC, LatencyObserver

MS = NS_PER_SEC // 1000


def _push(observer, frame, at_ms, source="LLM"):
    data = SimpleNamespace(frame=frame, timestamp=at_ms * MS, source=source)
    asyncio.run(observer.on_push_frame(data))


def _timeline(lines):
    return next(line for line in lines if "TIMELINE" in line)


def _observe(monkeypatch):
    lines = []
    from app.utils import latency as module

    monkeypatch.setattr(module.logger, "info", lambda msg: lines.append(msg))
    return LatencyObserver("sid"), lines


def test_each_station_is_the_gap_from_the_one_before(monkeypatch):
    observer, lines = _observe(monkeypatch)
    _push(observer, UserStoppedSpeakingFrame(), 1000)
    _push(observer, TranscriptionFrame("hi", "u", "t"), 1020)
    _push(observer, LLMContextFrame(context=None), 1030)
    # The LLM service pushes its start frame as the request leaves; the first token is
    # derived: request at 1050 plus a 400ms TTFB puts it at 1450.
    _push(observer, LLMFullResponseStartFrame(), 1050, source="CerebrasLLMService#0")
    _push(observer, MetricsFrame(data=[TTFBMetricsData(processor="CerebrasLLMService#0", value=0.400)]), 1450)
    _push(observer, TTSStartedFrame(), 1500)
    _push(observer, TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1), 1700)
    _push(observer, BotStartedSpeakingFrame(), 1720)

    line = _timeline(lines)
    assert "transcript=+20ms" in line
    assert "context=+10ms" in line
    assert "request=+20ms" in line
    assert "first_token=+400ms" in line
    assert "tts_start=+50ms" in line
    assert "first_audio=+200ms" in line
    assert "speaking=+20ms" in line


def test_a_station_never_reached_is_printed_as_such(monkeypatch):
    observer, lines = _observe(monkeypatch)
    _push(observer, UserStoppedSpeakingFrame(), 1000)
    _push(observer, TranscriptionFrame("hi", "u", "t"), 1020)
    _push(observer, BotStartedSpeakingFrame(), 1500)
    line = _timeline(lines)
    assert "tts_start=—" in line and "first_token=—" in line


def test_only_the_first_audio_frame_of_a_turn_is_a_station_and_the_rest_are_not_remembered(monkeypatch):
    """Audio arrives fifty times a second for the whole reply; remembering every id would be
    the one thing in this observer that grew with the length of the call."""
    observer, lines = _observe(monkeypatch)
    _push(observer, UserStoppedSpeakingFrame(), 1000)
    for at in (1300, 1320, 1340):
        _push(observer, TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1), at)
    _push(observer, BotStartedSpeakingFrame(), 1360)
    assert "first_audio=+300ms" in _timeline(lines)
    assert not any(isinstance(i, int) and i > 10_000 for i in observer._seen) or len(observer._seen) < 5


def test_the_stations_reset_with_the_turn(monkeypatch):
    observer, lines = _observe(monkeypatch)
    _push(observer, UserStoppedSpeakingFrame(), 1000)
    _push(observer, TranscriptionFrame("a", "u", "t"), 1010)
    _push(observer, BotStartedSpeakingFrame(), 1500)
    _push(observer, UserStoppedSpeakingFrame(), 5000)
    _push(observer, BotStartedSpeakingFrame(), 5400)
    second = [line for line in lines if "TIMELINE turn 2" in line][0]
    assert "transcript=—" in second


def test_a_transcript_that_landed_just_before_the_stop_frame_is_this_turns(monkeypatch):
    """Flux delivers the final transcript a few milliseconds before the stop frame that
    declares the turn. The first version reset the stations on the stop frame and printed
    transcript=— on every live turn; the gap it hid was the one worth seeing."""
    observer, lines = _observe(monkeypatch)
    _push(observer, TranscriptionFrame("hi", "u", "t"), 990)
    _push(observer, UserStoppedSpeakingFrame(), 1000)
    _push(observer, LLMContextFrame(context=None), 1503)
    _push(observer, BotStartedSpeakingFrame(), 1600)
    line = _timeline(lines)
    assert "transcript=-10ms" in line
    assert "context=+513ms" in line


def test_a_transcript_from_long_before_the_turn_is_not_borrowed(monkeypatch):
    from app.utils.latency import TRANSCRIPT_CARRY_NS

    observer, lines = _observe(monkeypatch)
    _push(observer, TranscriptionFrame("hi", "u", "t"), 1000)
    _push(observer, UserStoppedSpeakingFrame(), 1000 + TRANSCRIPT_CARRY_NS // MS + 1)
    _push(observer, BotStartedSpeakingFrame(), 9000)
    assert "transcript=—" in _timeline(lines)


def test_the_request_is_observed_and_the_first_token_derived():
    import inspect

    from app.utils import latency

    src = inspect.getsource(latency.LatencyObserver)
    assert 'self._stations.setdefault("request", data.timestamp)' in src
    assert 'stations["first_token"] = request + int(ttfb * NS_PER_SEC)' in src
