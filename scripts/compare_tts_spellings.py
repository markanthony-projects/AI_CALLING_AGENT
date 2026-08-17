"""Synthesise the same sentence several ways so a human can pick how it should be written.

Sarvam reads the text it is given, literally. "BHK" written solid was heard being attacked
as a single word, so it was spaced to "B H K" — and spaced it is read letter by letter,
which a live call showed sounds mechanical in a list. Both are wrong in different ways and
no amount of reasoning settles which is less wrong: the only test is listening.

Run it, play the files, pick a winner, and put that spelling in
app/utils/context_builder.py (spoken_configurations) and the SPEAKING STYLE rule in
app/prompts/agent_prompts.py.

    python scripts/compare_tts_spellings.py
    python scripts/compare_tts_spellings.py --out /tmp/tts --text "The 3 {bhk} is 1.46 Crores."

Costs one synthesis per variant against the live Sarvam account.
"""

import argparse
import asyncio
import base64
import sys
import wave
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402

ENDPOINT = "https://api.sarvam.ai/text-to-speech"

# The sentence a real call produces, with the spelling left as a slot.
DEFAULT_TEXT = "We have 2, 3, 3.5 and 4.5 {bhk} homes starting at 1.17 Crores."

# What to try. The name becomes the filename, so they sort in the order listed.
SPELLINGS = {
    "1_spaced": "B H K",  # what production says today
    "2_solid": "BHK",  # the original, said to be read as one word
    "3_dotted": "B.H.K.",
    "4_hyphenated": "B-H-K",
    "5_lowercase_dotted": "b.h.k.",
    "6_bedroom": "bedroom",  # sidesteps the acronym entirely
    "7_bhk_bedroom": "BHK bedroom",
}


async def synthesise(client: httpx.AsyncClient, text: str) -> bytes:
    response = await client.post(
        ENDPOINT,
        headers={"api-subscription-key": settings.SARVAM_API_KEY},
        json={
            # Matching app/services/agent.py so what you hear is what a caller hears.
            "text": text,
            "model": "bulbul:v3",
            "speaker": settings.SARVAM_VOICE_ID,
            "pace": 1.0,
            "target_language_code": "en-IN",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    audios = response.json().get("audios") or []
    if not audios:
        raise RuntimeError(f"Sarvam returned no audio for: {text!r}")
    return base64.b64decode(audios[0])


def duration_secs(wav_bytes: bytes) -> float:
    """How long it takes to say. A spelling that is slower is time the prospect waits."""
    import io

    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / float(w.getframerate())


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tts_samples", help="directory for the wav files")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="sentence with a {bhk} slot")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        for name, spelling in SPELLINGS.items():
            text = args.text.format(bhk=spelling)
            try:
                audio = await synthesise(client, text)
            except Exception as e:
                print(f"{name:<20} FAILED  {e}")
                continue
            path = out / f"{name}.wav"
            path.write_bytes(audio)
            print(f"{name:<20} {duration_secs(audio):5.2f}s  {path}")
            print(f"{'':<20} {text}")

    print(f"\nPlay them in order and pick one. Files are in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
