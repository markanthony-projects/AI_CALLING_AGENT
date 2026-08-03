"""Keeps tool-call syntax out of the caller's ear.

A live call ended with the agent reading this out loud:

    "...have a wonderful day. <function=end_call{"closing_line":"Understood, thank you
     so much for your time, have a wonderful day."}></function>"

Llama-family models on Groq sometimes emit a tool call as plain text in the content
channel instead of as a structured tool_call. Pipecat has no idea it is markup, so it goes
straight to TTS and the caller hears the raw syntax. Nothing downstream can recover from
that, so it has to be caught between the LLM and the TTS.

Once a model starts writing markup the rest of that response is off-format, so everything
from the marker onwards is dropped rather than repaired. Text before the marker is already
spoken and is left alone.
"""

import re
from typing import Awaitable, Callable, Optional, Tuple

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Words that, right after a '<', mean the model is writing markup rather than speech.
# '|' covers Llama's special tokens (<|python_tag|>), '/' the closing tags.
_MARKUP_WORDS = (
    "function",
    "function_call",
    "functioncall",
    "tool_call",
    "toolcall",
    "tool_use",
    "tooluse",
    "invoke",
    "antml",
    "python_tag",
    "|python_tag",
)

_CLOSING_LINE = re.compile(r'"closing_line"\s*:\s*"((?:[^"\\]|\\.)*)"')

# Past Latin Extended-B. Deliberately letters only, so an em dash, a curly quote or a rupee
# sign — none of which trouble the voice engine — do not cry wolf.
_LATIN_END = 0x24F


def non_latin_letters(text: str) -> str:
    """The characters in `text` that Sarvam will be asked to read in another script.

    The prompt forbids these outright because Sarvam breaks up mid-word on mixed script; a
    caller reported precisely that after the agent said "Mayur, नमस्ते Mayur". Nothing
    verified it, and the only agent text in the logs is the aggregated turn recorded
    *downstream* of this processor — so a stray token could be spoken and leave no trace.
    That is the position we were in when a caller reported hearing "hein" and "auugh" in
    audio whose transcript is clean ASCII throughout.
    """
    return "".join(dict.fromkeys(c for c in text if ord(c) > _LATIN_END and c.isalpha()))

TEXT = "text"
MARKUP = "markup"
PARTIAL = "partial"


def classify_bracket(fragment: str) -> str:
    """Decide what a fragment starting at '<' is, given only the text seen so far.

    Returns MARKUP when it is certainly tool syntax, TEXT when it certainly is not, and
    PARTIAL when the fragment is still too short to tell — the caller must then wait for
    more tokens rather than guess, because guessing TEXT speaks the markup out loud.
    """
    body = fragment[1:].lstrip()
    if fragment[1:] and not body:  # "<   " — whitespace only, still undecided
        return PARTIAL
    body = body.lstrip("/")
    low = body.lower()
    if not low:
        return PARTIAL
    for word in _MARKUP_WORDS:
        if low.startswith(word):
            return MARKUP
        if word.startswith(low):
            return PARTIAL
    return TEXT


def split_speakable(buffer: str) -> Tuple[str, str, bool]:
    """Split buffered model text into (speak now, hold back, markup started).

    Text with no '<' in it — which is every normal reply — passes through untouched, so
    the streaming TTS keeps its first-byte latency.
    """
    safe = ""
    rest = buffer
    while True:
        idx = rest.find("<")
        if idx == -1:
            return safe + rest, "", False
        verdict = classify_bracket(rest[idx:])
        if verdict == MARKUP:
            return safe + rest[:idx], rest[idx:], True
        if verdict == PARTIAL:
            return safe + rest[:idx], rest[idx:], False
        # A '<' that is genuinely part of speech; keep it and look past it.
        safe += rest[: idx + 1]
        rest = rest[idx + 1 :]


def extract_closing_line(markup: str) -> Optional[str]:
    match = _CLOSING_LINE.search(markup)
    if not match:
        return None
    try:
        return match.group(1).encode().decode("unicode_escape")
    except Exception:
        return match.group(1)


class ToolSyntaxFilter(FrameProcessor):
    """Strips leaked tool-call markup out of the LLM's spoken text.

    Sits between the LLM and the TTS, so the assistant context aggregator downstream also
    records the cleaned text — a leak that stayed in the context would teach the model
    that writing markup inline is acceptable for the rest of the call.
    """

    def __init__(
        self,
        call_sid: str,
        on_leaked_end_call: Optional[Callable[[Optional[str], bool], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._on_leaked_end_call = on_leaked_end_call
        self._buffer = ""
        self._leaked = ""
        self._suppressing = False
        self._spoke_this_response = False
        self._flagged_script = False

    def _reset(self) -> None:
        self._buffer = ""
        self._leaked = ""
        self._suppressing = False
        self._spoke_this_response = False
        self._flagged_script = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._suppressing:
                self._leaked += frame.text
                return  # dropped: the model is mid-markup, none of this is speech
            self._buffer += frame.text
            speak, held, markup_started = split_speakable(self._buffer)
            self._buffer = "" if markup_started else held
            if markup_started:
                self._suppressing = True
                self._leaked = held
            if speak:
                self._spoke_this_response = True
                self._report_script(speak)
                frame.text = speak
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._flush(direction)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _report_script(self, spoken: str) -> None:
        """Report, once per response, any script the voice engine cannot read cleanly.

        Deliberately reports rather than strips. Removing the characters would leave a
        half-word that is worse than the original, and the correct fix is upstream in the
        prompt — this exists so the next occurrence is visible instead of being something
        the caller can hear and the logs cannot show.
        """
        if self._flagged_script:
            return
        stray = non_latin_letters(spoken)
        if not stray:
            return
        self._flagged_script = True
        logger.warning(
            f"[{self._call_sid}] Non-Latin script sent to TTS ({stray!r}) in "
            f"{spoken.strip()[:80]!r} — Sarvam breaks up mid-word on mixed script, so the "
            f"caller may hear a garbled syllable here"
        )

    async def _flush(self, direction: FrameDirection) -> None:
        leaked, suppressed, spoke = self._leaked, self._suppressing, self._spoke_this_response
        # A partial '<...' left over at the end of the response was never markup after all.
        if not suppressed and self._buffer:
            frame = LLMTextFrame(self._buffer)
            self._buffer = ""
            await self.push_frame(frame, direction)
        self._reset()

        if not suppressed:
            return

        logger.error(
            f"[{self._call_sid}] Model wrote tool syntax into its spoken reply; "
            f"suppressed before TTS: {leaked[:200]!r}"
        )
        if "end_call" in leaked and self._on_leaked_end_call:
            # It meant to hang up. Honour that, but do not speak the leaked closing line
            # if a goodbye already went out in the same response — the caller would hear
            # two farewells back to back.
            await self._on_leaked_end_call(extract_closing_line(leaked), spoke)
