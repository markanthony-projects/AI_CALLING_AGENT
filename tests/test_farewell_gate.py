"""Waiting for the goodbye to actually be heard.

The closing line is the last thing the prospect hears and on a booked visit it carries the
day and the time, so cutting it off loses the confirmation the whole call was for. Two live
cutoffs shaped this:

    17:37:46.471  end_call fired
    17:37:46.896  pipeline finished        <- 425ms, for a three-second sentence

EndFrame stopping the transport in queue order caused that one. Replacing it with
EndWorkerFrame — which laps to the sink and back before becoming an EndFrame — did not fix
it, because the lap only proves the FRAMES travelled. Sarvam is a websocket TTS: run_tts
sends the text and returns, and the audio follows on a separate receive task.
"""

import asyncio

import pytest
from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from app.utils.farewell import (
    MAX_FAREWELL_WAIT_SECS,
    FarewellGate,
    farewell_timeout,
)

BOOKING_READBACK = (
    "Perfect Kumar, that's Sunday at 3 PM at Lakeview Residency. "
    "I'll send you the details. Thank you!"
)


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _feed(gate, frames):
    """Drive the real process_frame, bypassing only the started-processor check."""
    captured = _Captured()
    gate.push_frame = captured.push

    async def noop(*a, **kw):
        pass

    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


def _audio():
    return TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)


# ─── the timeout ──────────────────────────────────────────────────────────────────────


def test_a_long_readback_gets_time_to_finish():
    """The whole point of letting the model write this line is that it can name the day and
    the hour. Cutting it at a fixed two seconds would lose exactly that."""
    assert farewell_timeout(BOOKING_READBACK) >= 8


def test_a_short_goodbye_does_not_hold_the_line_open():
    """Every second waited is billed by the carrier, so the ceiling follows the sentence."""
    assert farewell_timeout("Thank you, goodbye!") < farewell_timeout(BOOKING_READBACK)


def test_nothing_waits_for_ever():
    """A dead TTS never raises BotStoppedSpeakingFrame. Waiting for one that is not coming
    would keep the phone leg up indefinitely."""
    assert farewell_timeout("word " * 500) <= MAX_FAREWELL_WAIT_SECS


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_line_still_allows_for_synthesis_latency(empty):
    """on_leaked_end_call can arm the gate with no line of its own, and the goodbye it is
    waiting on is already in flight."""
    assert farewell_timeout(empty) > 0


# ─── the gate ─────────────────────────────────────────────────────────────────────────


def test_it_releases_when_the_bot_has_finished_speaking():
    gate = FarewellGate()
    gate.arm()

    async def scenario():
        waiting = asyncio.create_task(gate.wait_until_spoken(5))
        await asyncio.sleep(0)
        await _feed(gate, [_audio(), BotStoppedSpeakingFrame()])
        return await waiting

    assert asyncio.run(scenario()) is True


def test_it_gives_up_rather_than_holding_the_leg_open():
    gate = FarewellGate()
    gate.arm()
    assert asyncio.run(gate.wait_until_spoken(0.05)) is False


def test_an_earlier_turn_cannot_satisfy_a_later_wait():
    """The gate sees a BotStoppedSpeakingFrame at the end of every reply, so by the time
    end_call fires one is already sitting there from the turn before. Without arm() clearing
    it the wait would return instantly and the goodbye would be cut off — the original bug,
    reintroduced through the mechanism meant to fix it."""
    gate = FarewellGate()
    asyncio.run(_feed(gate, [BotStoppedSpeakingFrame()]))  # the reply before the goodbye
    gate.arm()
    assert asyncio.run(gate.wait_until_spoken(0.05)) is False


def test_audio_alone_is_not_the_signal():
    """TTSAudioRawFrame means the voice has started arriving, not that it has been played."""
    gate = FarewellGate()
    gate.arm()

    async def scenario():
        waiting = asyncio.create_task(gate.wait_until_spoken(0.05))
        await asyncio.sleep(0)
        await _feed(gate, [_audio(), _audio()])
        return await waiting

    assert asyncio.run(scenario()) is False


def test_every_frame_passes_through_untouched():
    """It sits after the output transport, in front of the assistant aggregator. Swallowing
    a frame here would break the transcript the lead is extracted from."""
    sent = [_audio(), BotStoppedSpeakingFrame(), _audio()]
    gate = FarewellGate()
    gate.arm()
    assert asyncio.run(_feed(gate, sent)) == sent


def test_a_second_call_is_not_released_by_the_first_ones_frame():
    """Both end paths can fire in one call. The gate is re-armed each time, so a stale set
    event cannot make the second wait return instantly."""
    gate = FarewellGate()
    gate.arm()

    async def scenario():
        await _feed(gate, [BotStoppedSpeakingFrame()])
        assert await gate.wait_until_spoken(1) is True
        gate.arm()
        return await gate.wait_until_spoken(0.05)

    assert asyncio.run(scenario()) is False
