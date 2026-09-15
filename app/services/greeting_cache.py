"""The opening line, synthesised while the phone is still ringing.

Everything the greeting says is known when the dial is placed: the project, the developer,
the name off the lead list, the agent's name, and the time of day. The only thing that
waited for the prospect to pick up was the synthesis, and on 14 Sep that was 222ms of a
566ms first word. So the worker builds the same sentences the agent will speak, has Sarvam
synthesise each one, and leaves the audio in Redis under the call's id. The agent finds it
there when the media stream opens and plays it through app/utils/primed_speech.py.

Through the same door as the rest of the call. The first version used Sarvam's REST
endpoint — same model, speaker, pace and temperature — and the greeting still did not
match what followed it: measured on 15 Sep, REST audio is about 4dB louder than the
websocket's for the same sentence, and the prospect heard the call drop in volume after
the first line. So this opens the websocket the live call opens, sends the config the live
call sends (live_config below, pinned against the real one by tests/test_greeting_cache.py)
and takes the audio back the way the live call does. What the greeting sounds like is then
whatever the call sounds like, by construction.

Best effort at every step. A synthesis that fails, a Redis that is down, a ring that
outlasts the TTL — each of those costs the 222ms it used to cost, and nothing else. The
sentences are the cache keys, so any drift between what was synthesised and what the agent
asks to say — the time of day rolling over mid-ring — is a miss, not a wrong greeting.
"""

import asyncio
import base64
import json
import struct
from typing import Dict, List, Optional, Tuple

from loguru import logger
from websockets.asyncio.client import connect

from app.services.discovery import get_redis_client
from app.utils.dashes import spoken_punctuation
from app.utils.opening_line import build_opening_line
from app.utils.sentences import sentences

# Longer than any ring the dialer allows (VOBIZ_RING_SECONDS caps at 120) plus the carrier's
# start delay, short enough that a dial nobody answered does not sit in memory.
_TTL_SECONDS = 600
# The whole opening line, every sentence on its own socket in parallel. Past this the
# greeting is synthesised live as before; the pump must never wait on the voice engine.
_SYNTHESIS_BUDGET_SECS = 4.0
# The same endpoint pipecat's SarvamTTSService connects to, with the same query.
WS_URL = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
SAMPLE_RATE = 16000


def _key(call_sid: str) -> str:
    return f"greeting:{call_sid}"


def opening_sentences(project: dict, customer_name: Optional[str], now=None) -> List[str]:
    """Exactly the sentences the agent will queue, in the order it will queue them.

    The same builder with the same arguments as startup_greeting in app/services/agent.py,
    cut by the same sentences() that spoken() cuts with — the keys must match to the byte.
    """
    line = build_opening_line(
        project.get("name") or "your project",
        customer_name,
        now,
        developer_name=project.get("developer_name"),
        agent_name=project.get("agent_name"),
    )
    return sentences(line)


def live_config(settings) -> dict:
    """The config message the live call sends, rebuilt here without pipecat.

    Every key and value mirrors KeepsItsVoice._config_payload() for a service built by
    voice.build_tts(settings) — the defaults pipecat fills in (language, preprocessing,
    buffer sizes, codec) as much as the settings this repository chooses. The worker cannot
    import that class, so this is a copy, and tests/test_greeting_cache.py holds the two
    equal: change one and the test says so.
    """
    config = {
        "target_language_code": "en-IN",
        "speaker": settings.SARVAM_VOICE_ID,
        "speech_sample_rate": str(SAMPLE_RATE),
        "enable_preprocessing": True,
        "min_buffer_size": 50,
        "max_chunk_length": 150,
        "output_audio_codec": "linear16",
        "output_audio_bitrate": "128k",
        "pace": settings.SPEAKING_PACE,
        "model": "bulbul:v3",
    }
    if settings.SARVAM_TEMPERATURE is not None:
        config["temperature"] = settings.SARVAM_TEMPERATURE
    return config


def wav_pcm(wav: bytes) -> Optional[Tuple[bytes, int, int, int]]:
    """(pcm, rate, channels, bits) read from a RIFF/WAVE file, or None if it is not one.

    Walks the chunks rather than slicing 44 bytes off: the format is what says what the
    samples are, and a header that is not read is a header that is assumed.
    """
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return None
    rate = channels = bits = None
    pcm = None
    at = 12
    while at + 8 <= len(wav):
        tag = wav[at : at + 4]
        size = struct.unpack("<I", wav[at + 4 : at + 8])[0]
        body = wav[at + 8 : at + 8 + size]
        if tag == b"fmt " and len(body) >= 16:
            _, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
        elif tag == b"data":
            # Some encoders leave the data size as 0 or 0xFFFFFFFF for streamed output.
            pcm = body if 0 < size < 0xFFFFFFFF else wav[at + 8 :]
            break
        at += 8 + size + (size & 1)
    if pcm is None or rate is None:
        return None
    return pcm, rate, channels, bits


def pcm_16k_mono(audio: bytes) -> Optional[bytes]:
    """The audio as 16kHz mono PCM16, or None if that is not what it is.

    The websocket returns linear16 frames bare; a RIFF header, if one ever appears, is
    read rather than assumed. Anything at another rate is a miss — the engine synthesises
    the greeting live, in the right voice — never a slow, deep one (call 8571d93b).
    """
    if not audio:
        return None
    if audio.startswith(b"RIFF"):
        parsed = wav_pcm(audio)
        if parsed is None:
            return None
        pcm, rate, channels, bits = parsed
        if rate != SAMPLE_RATE or channels != 1 or bits != 16:
            logger.warning(
                f"Greeting audio came back as {rate}Hz/{channels}ch/{bits}-bit, not "
                f"{SAMPLE_RATE}Hz mono 16-bit; not priming it"
            )
            return None
        return pcm or None
    return audio


async def synthesise(text: str, settings) -> Optional[bytes]:
    """One sentence, over a socket configured exactly as the live call's. None on any refusal."""
    audio = bytearray()
    async with connect(
        WS_URL, additional_headers={"api-subscription-key": settings.SARVAM_API_KEY}
    ) as socket:
        await socket.send(json.dumps({"type": "config", "data": live_config(settings)}))
        await socket.send(json.dumps({"type": "text", "data": {"text": spoken_punctuation(text)}}))
        await socket.send(json.dumps({"type": "flush"}))
        async for raw in socket:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "audio":
                audio += base64.b64decode(message["data"]["audio"])
            elif kind == "event" and message.get("data", {}).get("event_type") == "final":
                break
            elif kind == "error":
                logger.warning(f"Greeting synthesis refused: {message.get('data')}")
                return None
    return pcm_16k_mono(bytes(audio))


async def prime_greeting(
    call_sid: str, project: dict, customer_name: Optional[str], settings
) -> int:
    """Synthesise the opening line's sentences and leave them for the call. Never raises.

    Returns how many sentences were cached, for the log and the tests.
    """
    if not getattr(settings, "GREETING_PRIME", True):
        return 0
    lines = opening_sentences(project, customer_name)
    try:
        audio = await asyncio.wait_for(
            asyncio.gather(*(synthesise(s, settings) for s in lines)),
            timeout=_SYNTHESIS_BUDGET_SECS,
        )
    except Exception as e:  # noqa: BLE001 — priming is an optimisation, never a failure
        logger.warning(
            f"[{call_sid}] Greeting not primed ({type(e).__name__}: {e}); it will be synthesised live"
        )
        return 0

    cached = {s: base64.b64encode(a).decode() for s, a in zip(lines, audio) if a}
    if not cached:
        return 0
    try:
        redis = await get_redis_client()
        await redis.setex(_key(call_sid), _TTL_SECONDS, json.dumps(cached))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{call_sid}] Greeting synthesised but not stored ({e})")
        return 0
    logger.info(
        f"[{call_sid}] Greeting primed while ringing | {len(cached)}/{len(lines)} sentence(s)"
    )
    return len(cached)


async def recall_primed_greeting(call_sid: str) -> Dict[str, bytes]:
    """The cached sentences for this call, or nothing. Read once, then gone."""
    try:
        redis = await get_redis_client()
        raw = await redis.get(_key(call_sid))
        if raw is None:
            return {}
        await redis.delete(_key(call_sid))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{call_sid}] Could not read the primed greeting ({e})")
        return {}
    try:
        stored = json.loads(raw)
        return {text: base64.b64decode(b64) for text, b64 in stored.items()}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{call_sid}] Primed greeting unreadable ({e})")
        return {}
