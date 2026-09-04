"""The call where the agent stopped speaking and the prospect hung up.

Live call 1e770466, 4 Sep 2026. After eight clean turns:

    10:24:17  USER  → "Actually, 1st"
    10:24:19  USER  → "I would love to know more about your job."
    10:24:22  USER  → "Yeah, I was saying"
    10:24:24  USER  → "I would love to know more about this project."
    10:24:28  ERROR  Error connecting to Sarvam TTS Websocket: timed out during
                     opening handshake
    10:24:38  ERROR  ...timed out during opening handshake
    10:24:40  the caller hung up

The agent had a reply ready — it is in the log — and no voice to say it with. Pipecat's
InterruptibleTTSService reopens the websocket on every interruption ("Handles interruptions
by reconnecting the websocket when the bot is speaking and gets interrupted"), so four
barge-ins in fourteen seconds meant four reconnects, and one handshake never completed.

None of that was visible. The reconnects had no line at all and had to be inferred from a
docstring; the second failure carried no call id because only the first was logged. And the
prospect's other complaint on the same call — that the opening line started late — could
not be answered either way, because nothing measured the one stretch it lives in.

These are measurements, not a fix. What the fix is depends on whether the reconnect churn
is ours or Sarvam's, and that is exactly what none of this could tell us before.
"""

import inspect

import pytest
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed

from app.utils.latency import NS_PER_SEC, LatencyObserver


def pushed(frame, at_seconds: float) -> FramePushed:
    return FramePushed(
        source=None, destination=None, frame=frame,
        direction=None, timestamp=int(at_seconds * NS_PER_SEC),
    )


async def drive(observer, events):
    for frame, at in events:
        await observer.on_push_frame(pushed(frame, at))


def captured(level="INFO"):
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level=level)
    return seen, sink


# --- how long the caller waits for the first word -------------------------------------


async def test_the_opening_line_is_measured_from_queue_to_audible():
    """The stretch nothing covered. Everything before it is logged to the millisecond and
    everything after it is covered per turn, but the first thing the prospect waits for —
    synthesis, then the trip out to the carrier — was invisible."""
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            (TTSSpeakFrame("Hi, Good afternoon Rahul."), 1.0),
            (BotStartedSpeakingFrame(), 2.4),
        ])
    finally:
        logger.remove(sink)
    line = next(m for m in seen if "GREETING audible" in m)
    assert "1400ms" in line


async def test_it_is_not_reported_as_a_turn():
    """It is not one. There is no preceding user turn, so folding it into the per-turn
    numbers would put a figure that includes no thinking time into the p50."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        (TTSSpeakFrame("Hi."), 1.0),
        (BotStartedSpeakingFrame(), 2.4),
    ])
    assert obs.turns == []


async def test_a_later_spoken_line_cannot_claim_to_be_the_greeting():
    """The goodbye and the recovery lines are queued exactly the same way. Measured as
    greetings they would report the wait for a sentence nobody was waiting on."""
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            (TTSSpeakFrame("Hi."), 1.0),
            (BotStartedSpeakingFrame(), 1.5),
            (UserStoppedSpeakingFrame(), 5.0),
            (BotStartedSpeakingFrame(), 5.8),
            (TTSSpeakFrame("Thank you for your time."), 9.0),
            (BotStartedSpeakingFrame(), 9.4),
        ])
    finally:
        logger.remove(sink)
    assert len([m for m in seen if "GREETING audible" in m]) == 1


async def test_the_turn_measurement_still_works_around_it():
    obs = LatencyObserver("sid")
    await drive(obs, [
        (TTSSpeakFrame("Hi."), 1.0),
        (BotStartedSpeakingFrame(), 1.5),
        (UserStoppedSpeakingFrame(), 5.0),
        (BotStartedSpeakingFrame(), 5.8),
    ])
    assert obs.turns == pytest.approx([0.8])


async def test_a_call_that_never_speaks_first_reports_nothing():
    """The prospect can speak before the greeting is queued, and then it is cancelled. A
    zero logged there would read as an instant greeting that never happened."""
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            (UserStoppedSpeakingFrame(), 2.0),
            (BotStartedSpeakingFrame(), 2.9),
        ])
    finally:
        logger.remove(sink)
    assert not [m for m in seen if "GREETING audible" in m]
    assert obs.turns == pytest.approx([0.9])


# --- and the failure that had no line ---------------------------------------------------


def _agent_source():
    from app.services import agent

    return inspect.getsource(agent.run_voice_agent)


def test_every_voice_failure_is_logged_and_not_only_the_first():
    """Two failures, twelve seconds of silence, one log line. The second existed solely as a
    Pipecat traceback with no call id on it, and a failure nobody can attribute to a call is
    not a diagnosis."""
    src = _agent_source()
    handler = src[src.index("async def on_tts_error") : src.index("async def on_tts_error") + 900]
    assert "if _tts_failures == 1:" not in handler
    assert "TTS failing" in handler
    assert "{_tts_failures}" in handler


def test_the_reconnects_are_counted():
    """Pipecat reopens this websocket on every interruption. Four barge-ins in fourteen
    seconds meant four reconnects, and none of them appeared anywhere — the churn that
    preceded the failure had to be inferred from a docstring."""
    src = _agent_source()
    assert 'tts.event_handler("on_connected")' in src
    assert "TTS reconnected" in src


def test_opening_the_voice_at_the_start_of_a_call_is_not_called_a_reconnect():
    """Every call connects once. Logging that as a reconnect would put a line on every call
    and make the runs that matter impossible to spot."""
    src = _agent_source()
    assert "if _tts_reconnects > 1:" in src


def test_a_connection_that_cannot_be_made_says_so_with_the_call_id():
    src = _agent_source()
    assert 'tts.event_handler("on_connection_error")' in src
    assert "the caller hears silence until it comes back" in src


@pytest.mark.parametrize("event", ["on_connected", "on_connection_error", "on_error"])
def test_the_events_being_hooked_are_events_this_service_has(event):
    """A handler registered under a name the service does not raise is silence that looks
    like instrumentation."""
    from pipecat.services.sarvam.tts import SarvamTTSService

    service = SarvamTTSService(
        api_key="x", settings=SarvamTTSService.Settings(model="bulbul:v3", voice="simran")
    )
    service.event_handler(event)(lambda *a, **k: None)
