"""Whether the agent waits for the rest of the sentence, or answers half of it.

A silence timer cannot tell a finished thought from a pause for breath. On a live call on
3 Sep 2026 one prospect said this, with a gap after each part:

    USER  "investment"        -> AGENT "That is a great goal..."
    USER  "is good or"        -> AGENT "It is a very good choice..."
    USER  "self is good?"     -> AGENT "It is wonderful for self use..."

One question, three turns, three separate answers to fragments. Every pause outlasted
TURN_SETTLE_SECS, so each fragment was a complete turn as far as the pipeline was concerned.
TurnFinalityGate does not cover this and was never meant to: it holds a reply generated
while the prospect is still audible, and here they had genuinely stopped.

Smart Turn answers the question the timer cannot. It is Pipecat's own model, bundled with
the package, run locally on CPU — no network call on the turn path.

It reads the audio, not the transcript, and that matters more here than it would elsewhere.
Hinglish transcription is the weakest part of this stack: on the same call "3 crore" came
back as "3 year" and put a timeline on the lead that nobody had said. A model listening to
intonation is not fooled by that, so this improvement does not have to wait for the speech
recognition one.

Off by default, and that default is not timidity. It was tried before and ruled "Maybe
around in 2" a finished turn on PSTN, which is the opposite failure and a worse one. This
is a later build, but the way to find out is a measured run on real calls rather than an
assumption — so it is one setting to turn on and one to turn back off, with no deploy in
either direction.
"""

from typing import Optional

from loguru import logger


def build_turn_analyzer(call_sid: str, settings) -> Optional[object]:
    """The call's end-of-turn analyzer, or None to keep the silence timer.

    None is what the transport received before this existed, so an untouched deployment
    behaves exactly as it did.

    Never raises. A model that will not load must cost the call its semantic turn detection
    and nothing else — falling back to the timer is a worse conversation, while failing here
    would be no conversation at all.
    """
    if not settings.SMART_TURN_ENABLED:
        return None

    try:
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        analyzer = LocalSmartTurnAnalyzerV3(
            cpu_count=settings.SMART_TURN_CPU_COUNT,
            params=SmartTurnParams(stop_secs=settings.SMART_TURN_STOP_SECS),
        )
    except Exception as e:
        logger.error(
            f"[{call_sid}] Smart Turn is enabled but would not load ({e.__class__.__name__}: {e}); "
            f"falling back to the silence timer for this call"
        )
        return None

    logger.info(
        f"[{call_sid}] Semantic turn detection on | smart-turn-v3 | "
        f"stop_secs={settings.SMART_TURN_STOP_SECS} cpu={settings.SMART_TURN_CPU_COUNT}"
    )
    return analyzer
