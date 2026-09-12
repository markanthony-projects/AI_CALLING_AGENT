"""Stopping the agent for somebody who keeps talking, and not for somebody who does not.

Deepgram tells us what the caller said while they are still saying it, so the gate beside
this one counts words. Flux and Sarvam say nothing at all until the turn is over — so a word
gate has nothing to count until the caller has finished, and the agent talks straight over
the whole utterance. That is not a tuning problem, it is the absence of any signal.

So this one counts seconds. It is blunter: "ek minute ruko" and "haan haan" are the same
length and mean opposite things, and no threshold can tell them apart. What it buys is that
it works at all without a transcript, and that it is no slower than the gate it stands in
for — three words plus the time to transcribe them is about a second in practice.
"""

import asyncio

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult

from app.services.turns import SustainedSpeechBargeIn

# Long enough that a test can act between arming and firing, short enough that the suite
# does not notice. The real call runs at settings.BARGE_IN_MIN_SPEECH_SECS.
QUICK = 0.05


class _Tasks:
    """The smallest thing that satisfies BaseObject.create_task/cancel_task."""

    def create_task(self, coroutine, name=None, context=None):
        return asyncio.get_running_loop().create_task(coroutine)

    async def cancel_task(self, task, timeout=None):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _gate(secs=QUICK):
    gate = SustainedSpeechBargeIn(min_speech_secs=secs)
    await gate.setup(_Tasks())
    gate._floor_taken = []
    gate.trigger_user_turn_started = lambda *a, **k: _took_it(gate._floor_taken)
    return gate


async def _took_it(sink):
    sink.append(True)


async def _settle(times=3):
    """Let the clock task run. Longer than QUICK, still instant to a human."""
    for _ in range(times):
        await asyncio.sleep(QUICK)


# --- while the agent is silent there is no floor to take -------------------------------


def test_with_the_agent_quiet_speech_starts_the_turn_at_once():
    """No sentence to protect, so waiting would be pure latency on every ordinary reply."""

    async def run():
        gate = await _gate()
        result = await gate.process_frame(VADUserStartedSpeakingFrame())
        assert gate._floor_taken == [True]
        assert result == ProcessFrameResult.STOP, "later strategies must not also fire"

    asyncio.run(run())


# --- while the agent is speaking, they have to mean it ---------------------------------


def test_a_short_noise_over_the_agent_does_not_stop_it():
    """A cough, a click, "hello?". The agent keeps its sentence."""

    async def run():
        gate = await _gate()
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await gate.process_frame(VADUserStoppedSpeakingFrame())
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


def test_somebody_who_keeps_talking_gets_the_floor():
    async def run():
        gate = await _gate()
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        assert gate._floor_taken == [], "it fired before the threshold"
        await _settle()
        assert gate._floor_taken == [True]

    asyncio.run(run())


def test_the_threshold_is_what_decides_and_not_the_frame():
    """A gate whose clock did nothing would pass the test above by firing immediately. This
    is the one that says the waiting is real."""

    async def run():
        gate = await _gate(secs=10)
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


def test_the_clock_restarts_rather_than_accumulating():
    """Two short bursts with a gap are two short bursts. Adding them up would mean a caller
    who says "hello?" twice loses the sentence they were straining to hear."""

    async def run():
        gate = await _gate(secs=0.15)
        await gate.process_frame(BotStartedSpeakingFrame())
        for _ in range(2):
            await gate.process_frame(VADUserStartedSpeakingFrame())
            await asyncio.sleep(0.1)
            await gate.process_frame(VADUserStoppedSpeakingFrame())
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


# --- when the sentence ends on its own -------------------------------------------------


def test_the_agent_finishing_stops_the_clock():
    """There is nothing left to interrupt. Letting it run out would start a second turn for
    one utterance — the first from the clock, the next from the VAD start that follows."""

    async def run():
        gate = await _gate()
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await gate.process_frame(BotStoppedSpeakingFrame())
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


def test_after_the_agent_finishes_the_next_word_is_immediate():
    """The pairing that matters: stopping the clock must not leave the gate armed against a
    caller who is now speaking into silence."""

    async def run():
        gate = await _gate()
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await gate.process_frame(BotStoppedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        assert gate._floor_taken == [True]

    asyncio.run(run())


def test_a_finished_turn_disarms_it():
    """reset() runs on every turn start. A clock left ticking across that boundary fires
    into the next turn, where nobody is speaking."""

    async def run():
        gate = await _gate()
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await gate.reset()
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


# --- the interface the agent calls ------------------------------------------------------


def test_it_answers_relax_like_the_word_gate_does():
    """agent.py calls relax() on whichever gate it was handed and must not have to ask which
    one it got. See app/services/stt_provider.py, where that choice is made."""
    from app.services.turns import GreetingOnlyMinWords

    assert callable(SustainedSpeechBargeIn.relax)
    assert callable(GreetingOnlyMinWords.relax)


def test_relaxing_does_not_quietly_drop_the_threshold():
    """The word gate relaxes because at three words it DISCARDED shorter replies. This one
    never discards — speech under the threshold is simply not an interruption, and the words
    still arrive as their own turn. So there is nothing to trade away, and a relax() that
    silently disarmed the gate would hand every cough the floor for the rest of the call."""

    async def run():
        gate = await _gate(secs=10)
        gate.relax()
        assert gate._min_speech_secs == 10
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await _settle()
        assert gate._floor_taken == []

    asyncio.run(run())


@pytest.mark.parametrize("secs", [0.2, 0.8, 3.0])
def test_the_threshold_is_whatever_it_was_built_with(secs):
    assert SustainedSpeechBargeIn(min_speech_secs=secs)._min_speech_secs == secs


def test_a_second_start_without_a_stop_replaces_the_clock_rather_than_adding_one():
    """VAD can report speech starting again without having reported it stopped. Leaving the
    first clock running would fire the turn twice for one utterance — and the second fires
    into a turn that has already started, where nobody is speaking.

    Mutation testing found this: the restart test above puts a VAD stop between the bursts,
    which disarms by the other path and passes either way."""

    async def run():
        gate = await _gate(secs=0.15)
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await asyncio.sleep(0.1)
        await gate.process_frame(VADUserStartedSpeakingFrame())
        await _settle(6)
        assert gate._floor_taken == [True], "one utterance, one turn"

    asyncio.run(run())


# --- and a turn that nobody ever ends ----------------------------------------------------
#
# Call 7b00a8af, 12 Sep 2026:
#
#     AGENT → "It sits on 45 acres with 14 towers. Do you know Varthur?"
#     USER  → "Sorry? Hello? I want to understand what you said. Can you repeat your
#              phone? Hello? Hello?"   (Total Turn Duration: 43062ms)
#     (the prospect hung up)
#
# One turn, forty-three seconds, no inference, nothing said. The silence is what kept them
# saying "Hello?", and that is what kept the turn open.


def _stopper(secs=QUICK):
    from app.services.turns import ServiceDecidesButNotForever

    gate = ServiceDecidesButNotForever(max_open_secs=secs)
    gate._stopped = []
    gate.trigger_user_turn_stopped = lambda *a, **k: _took_it(gate._stopped)
    return gate


def test_a_turn_the_service_never_ends_is_answered_anyway():
    from pipecat.frames.frames import UserStartedSpeakingFrame

    async def run():
        gate = _stopper()
        await gate.setup(_Tasks())
        await gate.process_frame(UserStartedSpeakingFrame())
        assert gate._stopped == [], "it gave up before the ceiling"
        await _settle(6)
        assert gate._stopped == [True]

    asyncio.run(run())


def test_a_service_that_does_end_the_turn_is_left_alone():
    """The ceiling is for a turn nobody is ending. Firing on an ordinary one would cut off
    every speaker who takes a breath, which is worse than what it is here to fix.

    Asserted on the clock rather than on the outcome: the base class fires its own stop on
    that frame, so counting stops cannot tell a cancelled ceiling from a working one."""
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame

    async def run():
        gate = _stopper()
        await gate.setup(_Tasks())
        await gate.process_frame(UserStartedSpeakingFrame())
        assert gate._deadline is not None, "it never started counting"
        await gate.process_frame(UserStoppedSpeakingFrame())
        assert gate._deadline is None, "the clock outlived the turn"

    asyncio.run(run())


def test_the_clock_does_not_survive_into_the_next_turn():
    from pipecat.frames.frames import UserStartedSpeakingFrame

    async def run():
        gate = _stopper()
        await gate.setup(_Tasks())
        await gate.process_frame(UserStartedSpeakingFrame())
        await gate.reset()
        assert gate._deadline is None
        await _settle(6)
        assert gate._stopped == []

    asyncio.run(run())


def test_the_ceiling_is_actually_waited_out():
    """A ceiling that fired at once would pass the test above and cut off every caller on
    their first syllable — the exact failure it exists to avoid becoming."""
    from pipecat.frames.frames import UserStartedSpeakingFrame

    async def run():
        gate = _stopper(secs=10)
        await gate.setup(_Tasks())
        await gate.process_frame(UserStartedSpeakingFrame())
        await _settle(6)
        assert gate._stopped == []

    asyncio.run(run())


def test_the_ceiling_is_whatever_it_was_built_with():
    from app.services.turns import ServiceDecidesButNotForever

    assert ServiceDecidesButNotForever(max_open_secs=7.5)._max_open_secs == 7.5


def test_it_is_still_the_strategy_the_service_drives():
    """Subclassed rather than replaced. Flux ending the turn is the whole reason Flux is
    here; the ceiling only covers the case where it does not."""
    from pipecat.turns.user_stop import ExternalUserTurnStopStrategy

    from app.services.turns import ServiceDecidesButNotForever

    assert issubclass(ServiceDecidesButNotForever, ExternalUserTurnStopStrategy)
