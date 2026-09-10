"""Punctuation the voice engine reads badly, replaced before it gets there.

Live call 2eeb48a0, 10 Sep 2026. The model wrote:

    We are launching a new project in Varthur – Sarjapur Road. It is called Abhee Codename
    New Dimension – Bengaluru's first Scotland‑themed residential…

Two en-dashes and, inside "Scotland‑themed", a non-breaking hyphen (U+2011) copied out of
the campaign context. None of them is a sentence boundary, so Pipecat sent the whole thing
to Sarvam as one request; and none of them is punctuation Sarvam's guidance mentions — it
lists the comma, the full stop and the ellipsis as the pauses it honours. The prospect said
"Sorry I didn't catch that."

A dash between two clauses is a comma to the ear, so that is what it becomes. A dash
between two words is a hyphen. The model's text is not rewritten beyond that: what the
prospect hears is still what the model said, in punctuation the engine can read.
"""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

# En dash, em dash, and the horizontal bar — set off by spaces, so they join clauses.
_CLAUSE_DASH = re.compile(r"\s*[–—―]\s*")
# The same characters between letters with no spaces, and the non-breaking hyphen, join
# words. A prospect hears "Scotland-themed" either way; the engine only reads one of them.
_WORD_DASH = re.compile(r"[‑‐‒]")


def spoken_punctuation(text: str) -> str:
    """`text` with dashes the engine misreads turned into punctuation it honours."""
    if not text:
        return text
    text = _WORD_DASH.sub("-", text)
    text = _CLAUSE_DASH.sub(", ", text)
    # A dash that opened a sentence, or followed a comma already there.
    text = re.sub(r"(^|[,.!?:;])\s*,\s*", r"\1 ", text).strip()
    return text


class DashFilter(BaseTextFilter):
    """Pipecat text filter: runs on every sentence after aggregation, before synthesis."""

    async def filter(self, text: str) -> str:
        return spoken_punctuation(text)
