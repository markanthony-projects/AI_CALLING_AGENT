"""Split a line the system speaks into the sentences the voice engine should hear one at a time.

Pipecat hands the model's reply to the voice engine one sentence per request: the text
aggregator cuts at full stops, and each cut is a real gap the caller hears, with the next
sentence synthesised while the last one plays. A TTSSpeakFrame skips that aggregator. Its
text goes to the engine in one request, however many sentences it holds.

The opening line holds three. On a live call on 10 Sep 2026 it was the one line reported as
sounding "like a machine", flat and slow, while every model-generated reply sounded fine —
and the difference between them was not the words or the voice but this: the greeting went
to Sarvam as a single 110-character block and came back as a single 110-character breath.
Sarvam's own guidance says the same thing from the other side: "break into shorter
sentences", "a full stop is a medium pause".

So the lines the system speaks itself are cut here, before they are queued, and queued one
sentence per frame — the same shape the model's replies already arrive in.
"""

import re

from app.utils.person_name import _SALUTATIONS

# Full stops that end a word rather than a sentence. The salutations are the ones
# spoken_name can put in front of a prospect's name — "Good afternoon Mr. Rahul." must not
# become "Good afternoon Mr." and "Rahul." — and they are never the last word of a sentence,
# so guarding them costs nothing. "Pvt." is guarded for the same reason: it is always
# followed by "Ltd.". "Ltd." itself is NOT: it is the last word of "calling you from
# Prestige Pvt. Ltd." and the full stop after it is a real sentence end.
_ABBREVIATIONS = sorted(
    {s.rstrip(".") for s in _SALUTATIONS.values() if s.endswith(".")} | {"Pvt", "St", "Bros"}
)

_BOUNDARY = re.compile(
    "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"(?<=[.!?])\s+"
)


def sentences(text: str) -> list[str]:
    """The sentences of `text`, in order, each stripped, none empty.

    A sentence ends at . ! or ? followed by whitespace. A full stop inside a number ("2.5
    Crores") is followed by a digit, not whitespace, and is left alone; one after a known
    abbreviation is left alone by name.
    """
    if not text:
        return []
    return [part.strip() for part in _BOUNDARY.split(text.strip()) if part.strip()]
