"""A line that has gone quiet with the prospect still on it is asked into again.

Three calls on 15 Sep 2026 — 5a2c8245, 5e247d87, 81756200 — ended the same way: the agent
fell silent, the prospect said "Hello?" into the silence, and every "Hello?" kept the
speech service's turn open, so no reply was ever generated, and they hung up. Each had a
different first cause (a turn the service never closed, a reply that was never spoken),
and every existing backstop missed all three:

  * the dead-air nudge fires when a prospect's turn ends EMPTY — these turns never ended;
  * the reply watchdog logs when a turn's reply is late — these turns never stopped, so
    there was no turn to wait on;
  * ServiceDecidesButNotForever's 10s cap asks the turn controller to stop the turn, and
    pipecat's controller refuses while the service says the user is speaking — which,
    on a stuck turn, is exactly the state it is in. The cap has never fired on a call;
  * the idle timeout waits sixty seconds of nobody speaking, and the prospect was speaking.

So this watches the two facts that are true regardless of what the speech service
believes: no audio has left the socket for Vobiz for OPEN_LINE_SECS, and the VAD has not
heard the prospect for PROSPECT_QUIET_SECS. The first is read off the socket witness, not
off the pipeline's bot-speaking events: on call 511dfa31 that event landed after the first
sentence of a three-sentence reply, and the watchdog asked "did the line drop?" into a
reply that had finished six seconds before. Bytes leaving the socket cannot be misread. Both, and the agent asks its last
question again through the same bounded door the other nudges use. It never talks over
a prospect who is audibly speaking, and it never fires while the agent is.
"""

from typing import Optional

# How long the agent may be silent, with the prospect quiet, before it speaks into the
# line. Above any turn a healthy call takes (p95 today is 2.5s) and below the point where
# a prospect starts saying "Hello?" — which on the calls above was about ten seconds.
OPEN_LINE_SECS = 10.0
# How long the VAD must have heard nothing. The nudge must not land on someone mid-answer.
PROSPECT_QUIET_SECS = 2.0
# The loop's tick. Coarse on purpose: this is a backstop, not a metronome.
CHECK_EVERY_SECS = 1.0
# Audio left the socket this recently means the agent is still speaking. The transport
# writes a chunk every 40ms while it plays; a second without one is a finished reply.
STILL_SPEAKING_SECS = 1.0


def line_has_gone_dead(
    now: float,
    *,
    bot_speaking: bool,
    bot_stopped_at: Optional[float],
    last_voice_at: Optional[float],
    last_nudge_at: Optional[float],
    holding: bool,
    ending: bool,
    last_turn_stopped_at: Optional[float] = None,
    inference_in_flight: bool = False,
) -> bool:
    """Whether to speak into the silence now. Pure, so every branch has a test.

    Three clocks have to agree that nothing is happening: the agent's last audio, the
    agent's last nudge, and the prospect's last finished turn. The third was missing on
    call 2dbf4ee1: the prospect answered at length, the agent's last audio was eleven
    seconds old by the time their turn was declared over, and the watchdog spoke into the
    two seconds in which the reply to that answer was being generated. A turn that has
    just ended is a reply on its way, and a reply in flight is not silence.
    """
    if bot_speaking or holding or ending or inference_in_flight:
        return False
    if bot_stopped_at is None:
        return False  # the agent has not spoken yet; the greeting is on its way
    quiet_since = max(bot_stopped_at, last_nudge_at or 0.0, last_turn_stopped_at or 0.0)
    if now - quiet_since < OPEN_LINE_SECS:
        return False
    if last_voice_at is not None and now - last_voice_at < PROSPECT_QUIET_SECS:
        return False  # they are talking, or just were; not the moment
    return True
