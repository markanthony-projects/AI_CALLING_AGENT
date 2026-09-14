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
#
# The rest are the abbreviations a closing line can carry — a read-back with a price, a date
# and an address in it — and each of them was cut in two by the version that knew only the
# salutations: "Possession is Dec." / "2027.", "Price is Rs." / "85 Lakhs.", "Sunday 11 A.M."
# / "works." Each cut is an audible gap where a person would not pause. "Rs" and the months
# never end a sentence, so they are unconditional.
_MONTHS = {"Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"}
_ABBREVIATIONS = sorted(
    {s.rstrip(".") for s in _SALUTATIONS.values() if s.endswith(".")}
    | {"Pvt", "St", "Bros", "Rs", "Sq"}
    | _MONTHS
)

# Abbreviations that CAN end a sentence — "It costs 1.2 Cr. Shall I go on?" is two — so
# they only hold the sentence together when what follows could not start one: a digit or a
# lowercase word. "1,450 Sq. Ft. carpet" stays whole ("Sq." is unconditional above, since
# it is always followed by "Ft."); "Sarjapur Rd. Do you know it?" splits.
_CONDITIONAL = {"Rd", "Cr", "Ft", "No"}
_ENDS_CONDITIONALLY = re.compile(r"\b(?:" + "|".join(sorted(_CONDITIONAL)) + r")\.$")
_CANNOT_START_A_SENTENCE = re.compile(r"^(?:\d|[a-z])")

# Dotted times. "11 A.M. works" split into three frames — the lookbehind per abbreviation
# handles one full stop per word, and these carry two.
_DOTTED_TIME = re.compile(r"\b[AaPp]\.[Mm]\.$")

_BOUNDARY = re.compile(
    "".join(rf"(?<!\b{re.escape(a)}\.)" for a in _ABBREVIATIONS) + r"(?<=[.!?])\s+"
)


def _held_together(left: str, right: str) -> bool:
    """Whether `left` ended on an abbreviation that should not have closed the sentence."""
    if _DOTTED_TIME.search(left):
        return True
    return bool(_ENDS_CONDITIONALLY.search(left)) and bool(_CANNOT_START_A_SENTENCE.match(right))


def sentences(text: str) -> list[str]:
    """The sentences of `text`, in order, each stripped, none empty.

    A sentence ends at . ! or ? followed by whitespace. A full stop inside a number ("2.5
    Crores") is followed by a digit, not whitespace, and is left alone; one after a known
    abbreviation is left alone by name; one after an abbreviation that only sometimes ends
    a sentence ("Cr.", "Rd.") is left alone when the next word could not begin one.
    """
    if not text:
        return []
    parts = [part.strip() for part in _BOUNDARY.split(text.strip()) if part.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _held_together(merged[-1], part):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged
