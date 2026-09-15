"""A voice socket opened while the phone is still being answered.

Vobiz's answer webhook arrives before the media stream opens — by three seconds on the
14 Sep calls, by a few hundred milliseconds on the 15 Sep ones. The voice engine's
handshake, 180ms, was paid inside the first word after that window had gone by unused. So
the webhook builds the voice service and opens its socket, this module holds it under the
call's id, and the agent adopts it when the stream opens. The service is built by the same
factory the agent uses (app/services/voice.py), so a warmed voice is the voice.

The window is not always long enough. On call 81bdc87a the stream opened while the
handshake was still in flight, the agent found nothing to adopt, built its own, and the
warmed socket sat unused until it expired. So adopting waits, briefly, for a handshake that
has started but not finished: a socket 100ms from ready is worth more than a cold start.

Best effort throughout. A handshake that fails or a stream that never opens leaves nothing
behind but a log line; the agent finds no warm socket and builds its own as before.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple

from loguru import logger
from websockets.protocol import State

from app.services.voice import KeepsItsVoice, build_tts

# What the transport plays at. SarvamTTSService takes its own rate from its constructor
# and falls back to this only when that is unset — the same rule as its start(), and the
# socket warmed here must be configured with the rate start() will label the audio with,
# or the audio plays at the wrong speed. See app/utils/voice_rate.py for the call.
TRANSPORT_SAMPLE_RATE = 16000
CONNECT_TIMEOUT_SECS = 3.0
# How long the agent will wait for a handshake that is already in flight. Under the cold
# path's own cost (183ms on 81bdc87a), so waiting can never be worse than not.
ADOPT_WAIT_SECS = 0.4
# Longer than any gap between the answer webhook and the media stream; short enough that a
# call which never streamed does not hold a socket open on Sarvam's side.
MAX_WAIT_SECS = 60.0

_warm: Dict[str, Tuple[KeepsItsVoice, float]] = {}
_pending: Dict[str, asyncio.Task] = {}


def rate_at_start(tts: KeepsItsVoice) -> int:
    """The rate start() will set — its own constructor rate, else the transport's.

    Mirrors TTSService.start: `self._init_sample_rate or frame.audio_out_sample_rate`. A
    socket configured with anything else sends audio the service then mislabels.
    """
    return tts._init_sample_rate or TRANSPORT_SAMPLE_RATE


def _socket_open(tts: KeepsItsVoice) -> bool:
    return tts._websocket is not None and tts._websocket.state is State.OPEN


def begin(call_sid: str, settings) -> asyncio.Task:
    """Start warming a socket for this call, off the webhook's own response path."""
    task = asyncio.create_task(prepare(call_sid, settings))
    _pending[call_sid] = task
    task.add_done_callback(lambda t: _pending.pop(call_sid, None) if _pending.get(call_sid) is t else None)
    return task


async def prepare(call_sid: str, settings) -> bool:
    """Open the voice socket for a call whose media stream is about to arrive."""
    tts = build_tts(settings, call_sid=call_sid)
    tts._speech_sample_rate = str(rate_at_start(tts))
    try:
        await asyncio.wait_for(tts._connect_websocket(), timeout=CONNECT_TIMEOUT_SECS)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{call_sid}] Voice socket not warmed ({type(e).__name__}: {e})")
        return False
    if not _socket_open(tts):
        logger.warning(f"[{call_sid}] Voice socket not warmed; the call will open its own")
        return False
    _warm[call_sid] = (tts, time.monotonic())
    asyncio.create_task(_expire(call_sid))
    logger.info(f"[{call_sid}] Voice socket warmed before the media stream")
    return True


async def adopt(call_sid: str, wait: float = ADOPT_WAIT_SECS) -> Optional[KeepsItsVoice]:
    """The warmed service for this call, if there is one and its socket is still open.

    A handshake still in flight is waited for, up to `wait`; one that has not started, or
    does not finish in time, is a cold start as before.
    """
    if call_sid not in _warm:
        pending = _pending.get(call_sid)
        if pending is not None and not pending.done():
            try:
                await asyncio.wait_for(asyncio.shield(pending), timeout=wait)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
    entry = _warm.pop(call_sid, None)
    if entry is None:
        return None
    tts, since = entry
    if not _socket_open(tts):
        logger.warning(f"[{call_sid}] Warmed voice socket had closed; the call will open its own")
        return None
    logger.info(
        f"[{call_sid}] Adopting the warmed voice socket ({time.monotonic() - since:.1f}s old)"
    )
    return tts


async def _expire(call_sid: str):
    await asyncio.sleep(MAX_WAIT_SECS)
    entry = _warm.pop(call_sid, None)
    if entry is None:
        return
    tts, _ = entry
    logger.warning(f"[{call_sid}] Media stream never opened; closing the warmed voice socket")
    try:
        await tts._disconnect_websocket()
    except Exception:  # noqa: BLE001
        pass


def waiting() -> int:
    """How many warmed sockets are waiting for their stream. For tests and the health line."""
    return len(_warm)
