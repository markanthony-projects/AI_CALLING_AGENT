"""The opening line, synthesised while the phone is still ringing.

Everything the greeting says is known when the dial is placed: the project, the developer,
the name off the lead list, the agent's name, and the time of day. The only thing that
waited for the prospect to pick up was the synthesis, and on 14 Sep that was 222ms of a
566ms first word. So the worker builds the same sentences the agent will speak, asks Sarvam
for each one over REST, and leaves the audio in Redis under the call's id. The agent finds
it there when the media stream opens and plays it through app/utils/primed_speech.py.

Best effort at every step. A synthesis that fails, a Redis that is down, a ring that
outlasts the TTL — each of those costs the 222ms it used to cost, and nothing else. The
sentences are the cache keys, so any drift between what was synthesised and what the agent
asks to say — the time of day rolling over mid-ring — is a miss, not a wrong greeting.
"""

import asyncio
import base64
import json
from typing import Dict, List, Optional

import httpx
from loguru import logger

from app.services.discovery import get_redis_client
from app.utils.dashes import spoken_punctuation
from app.utils.opening_line import build_opening_line
from app.utils.sentences import sentences

# Longer than any ring the dialer allows (VOBIZ_RING_SECONDS caps at 120) plus the carrier's
# start delay, short enough that a dial nobody answered does not sit in memory.
_TTL_SECONDS = 600
# The whole opening line, all sentences in parallel. Past this the greeting is synthesised
# live as before; the pump must never wait on the voice engine.
_SYNTHESIS_BUDGET_SECS = 4.0
_URL = "https://api.sarvam.ai/text-to-speech"
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


def payload(text: str, settings) -> dict:
    """The REST request, matching the websocket config the live call sends.

    Same model, same speaker, same pace, same language and preprocessing, same temperature
    when one is set — a greeting in a different voice from the rest of the call is worse
    than a greeting 222ms late. The text goes through the same dash filter the engine's
    input passes through, so the two spellings cannot differ either.
    """
    body = {
        "text": spoken_punctuation(text),
        "target_language_code": "en-IN",
        "speaker": settings.SARVAM_VOICE_ID,
        "sample_rate": SAMPLE_RATE,
        # What the live socket resolves to: pipecat's TTSSettings default is True, and the
        # Sarvam service's own False is overridden by it. Pinned against the real config in
        # tests/test_greeting_cache.py — change one, the test says so.
        "enable_preprocessing": True,
        "model": "bulbul:v3",
        "pace": settings.SPEAKING_PACE,
    }
    if settings.SARVAM_TEMPERATURE is not None:
        body["temperature"] = settings.SARVAM_TEMPERATURE
    return body


def pcm_from_response(data: dict) -> Optional[bytes]:
    """16kHz mono PCM16 out of Sarvam's reply, or None if there was no audio in it."""
    audios = data.get("audios") or []
    if not audios:
        return None
    audio = base64.b64decode(audios[0])
    if len(audio) > 44 and audio.startswith(b"RIFF"):
        audio = audio[44:]
    return audio or None


async def synthesise(text: str, settings, client: httpx.AsyncClient) -> Optional[bytes]:
    response = await client.post(
        _URL,
        json=payload(text, settings),
        headers={"api-subscription-key": settings.SARVAM_API_KEY},
    )
    if response.status_code != 200:
        logger.warning(f"Greeting synthesis refused ({response.status_code}): {response.text[:160]}")
        return None
    return pcm_from_response(response.json())


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
        async with httpx.AsyncClient(timeout=_SYNTHESIS_BUDGET_SECS) as client:
            audio = await asyncio.wait_for(
                asyncio.gather(*(synthesise(s, settings, client) for s in lines)),
                timeout=_SYNTHESIS_BUDGET_SECS,
            )
    except Exception as e:  # noqa: BLE001 — priming is an optimisation, never a failure
        logger.warning(f"[{call_sid}] Greeting not primed ({type(e).__name__}: {e}); it will be synthesised live")
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
    logger.info(f"[{call_sid}] Greeting primed while ringing | {len(cached)}/{len(lines)} sentence(s)")
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
