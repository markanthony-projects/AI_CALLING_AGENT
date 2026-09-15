"""A voice socket opened while the phone is still being answered.

Vobiz's answer webhook arrives about three seconds before the media stream opens: 611ms for
the reply to become a websocket, then Vobiz's start event. On 14 Sep the voice engine's
handshake — 180ms — was paid inside the first word, after that window had gone by unused.
So the webhook builds the voice service and opens its socket, this module holds it under
the call's id, and the agent adopts it when the stream opens. The service is built by the
same factory the agent uses (app/services/voice.py), so a warmed voice is the voice.

Best effort throughout. A handshake that fails or a stream that never opens leaves nothing
behind but a log line; the agent finds no warm socket and builds its own as before.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple

from loguru import logger
from websockets.protocol import State

from app.services.voice import KeepsItsVoice, build_tts

# The pipeline's output rate. SarvamTTSService sets this from the StartFrame; a socket
# opened before the pipeline exists has to be told, or the config it sends is wrong.
SAMPLE_RATE = 16000
CONNECT_TIMEOUT_SECS = 3.0
# Longer than any gap between the answer webhook and the media stream; short enough that a
# call which never streamed does not hold a socket open on Sarvam's side.
MAX_WAIT_SECS = 60.0

_warm: Dict[str, Tuple[KeepsItsVoice, float]] = {}


def _socket_open(tts: KeepsItsVoice) -> bool:
    return tts._websocket is not None and tts._websocket.state is State.OPEN


async def prepare(call_sid: str, settings) -> bool:
    """Open the voice socket for a call whose media stream is about to arrive."""
    tts = build_tts(settings)
    tts._speech_sample_rate = str(SAMPLE_RATE)
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


def adopt(call_sid: str) -> Optional[KeepsItsVoice]:
    """The warmed service for this call, if there is one and its socket is still open."""
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
