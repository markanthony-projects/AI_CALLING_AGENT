"""Punctuation the voice engine reads badly, replaced before it gets there.

Live call 2eeb48a0, 10 Sep 2026. The model wrote:

    We are launching a new project in Varthur – Sarjapur Road. It is called Abhee Codename
    New Dimension – Bengaluru's first Scotland‑themed residential…

Two en-dashes and, inside "Scotland‑themed", a non-breaking hyphen (U+2011) copied out of
the campaign context. None of them is a sentence boundary, so Pipecat sent the whole thing
to Sarvam as one request; and none of them is punctuation Sarvam's guidance mentions — it
lists the comma, the full stop and the ellipsis as the pauses it honours. The prospect said
"Sorry I didn't catch that."

A dash between two clauses is a comma to the ear, so that is what it becomes. A dash between
two words is a hyphen. A dash between two NUMBERS is neither — it is a range, and the word
for it is "to". That last rule is not decoration: the first version of this file turned
"20-30 Lakhs below launch" into "20, 30 Lakhs below launch" and "1.17-2.64 Crores" into two
unrelated prices, which is the money said wrong, on every call, in the one part of the pitch
nobody can afford to have wrong.

The model's text is not rewritten beyond that: what the prospect hears is still what the
model said, in punctuation the engine can read.
"""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

# A dash between two numbers is a range, and a range is the one place where turning the dash
# into a comma changes the facts: "20-30 Lakhs below launch" became "20, 30 Lakhs below
# launch", which is two figures where the prospect was told one span, and "1.17-2.64 Crores"
# became a pair of unrelated prices. Ranges are how every price in the campaign context is
# written. Said out loud, "to" is the word — so it is the word.
#
# Checked FIRST, before anything else touches a dash, and covering the plain hyphen too:
# whatever the context was written with, "20-30 Lakhs" reaching the engine as a range and
# leaving it as two numbers is the same wrong sentence.
_RANGE_DASH = re.compile(r"(?<=\d)\s*[-–—―‑‐‒]\s*(?=\d)")
# En dash, em dash, and the horizontal bar between anything else: a clause break to the ear,
# which is a comma.
_CLAUSE_DASH = re.compile(r"\s*[–—―]\s*")
# The hyphen-like characters the engine does not read, joining two words. A prospect hears
# "Scotland-themed" either way; the engine only reads one of them.
_WORD_DASH = re.compile(r"[‑‐‒]")


def spoken_punctuation(text: str) -> str:
    """`text` with dashes the engine misreads turned into punctuation it honours."""
    if not text:
        return text
    text = _RANGE_DASH.sub(" to ", text)
    text = _WORD_DASH.sub("-", text)
    text = _CLAUSE_DASH.sub(", ", text)
    # A dash that opened a sentence, or followed a comma already there.
    text = re.sub(r"(^|[,.!?:;])\s*,\s*", r"\1 ", text).strip()
    return text


class DashFilter(BaseTextFilter):
    """Pipecat text filter: runs on every sentence after aggregation, before synthesis."""

    async def filter(self, text: str) -> str:
        return spoken_punctuation(text)
