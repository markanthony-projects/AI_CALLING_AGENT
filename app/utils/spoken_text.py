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

# Both shapes the leak has taken: JSON ("closing_line": "...") and the bare call form
# (closing_line="..."), with either quote character. The key is unquoted in the second, so a
# pattern that insisted on the JSON shape recovered nothing from it — and the prospect got a
# generic goodbye on exactly the calls where the model had written a good one.
_CLOSING_LINE = re.compile(
    r"""["']?closing_line["']?\s*[:=]\s*(?P<q>["'])(?P<text>(?:(?!(?P=q))[^\\]|\\.)*)(?P=q)"""
)

# Markup the model writes WITHOUT a leading '<'. Everything above keys off a bracket, which
# is how Llama-on-Groq leaked. Gemma-on-Cerebras does not use brackets at all:
#
#     "Thank you for your time. node: end_call(closing_line="Have a great day!")"
#
# That reached a live caller on 2 Sep 2026. The bracket scanner never saw a '<', so nothing
# fired: no suppression, no leak log, and the text went to the TTS and into the context.
# The provider changed and the shape of the leak changed with it.
#
# Every entry carries an underscore or is a code fence, so none can occur in a spoken sales
# line. That is the whole basis for cutting on sight. A bare word like "functions" or "node"
# is deliberately NOT here: "we have a hall for functions." is a sentence this agent could
# legitimately say, and cutting there would cost the caller real speech.
_BARE_MARKERS = (
    "end_call",      # the only tool this agent exposes
    "closing_line",  # its only parameter
    "tool_call",
    "toolcall",
    "tool_use",
    "tooluse",
    "function_call",
    "functioncall",
    "```",           # a markdown code fence
    "<|",            # a special token whose word classify_bracket does not know
)

# A label left dangling in front of the call itself — the "node: " of the leak above, or the
# '{"name": ' of a JSON one. Cutting at the marker alone leaves it behind and the caller
# hears "node" for no reason.
#
# Anchored on ':' '=' or an opening bracket, never on '.', so an ordinary sentence ending
# ("...below the launch price.") cannot match it. Only ever applied at a cut point, where
# markup follows and nothing further will be appended.
_TRAILING_LABEL = re.compile(
    r"""(?:
          ["']?[A-Za-z_][A-Za-z0-9_]{0,30}["']?\s*[:=]
        | [\[\{\("'`]
        )\s*$""",
    re.X,
)

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

def trim_label_tail(spoken: str) -> str:
    """Drop the label the model left dangling in front of its tool call."""
    out = spoken.rstrip()
    # Bounded rather than `while True`: a pathological response must not spin here, and four
    # labels deep ('{"name": "') is already deeper than any leak has gone.
    for _ in range(6):
        match = _TRAILING_LABEL.search(out)
        if not match:
            break
        out = out[: match.start()].rstrip()
    return out


def _earliest_bare_marker(lowered: str) -> int:
    """Where the first bare marker starts in already-lowercased text, or -1."""
    found = -1
    for marker in _BARE_MARKERS:
        at = lowered.find(marker)
        if at != -1 and (found == -1 or at < found):
            found = at
    return found


def _starts_a_word(text: str, at: int) -> bool:
    """Whether position `at` could be the first character of a marker.

    A marker always begins a word, so a trailing "e" inside "fine" is not the start of
    `end_call` and must not be held back. Without this the last letter of almost every
    ordinary reply was held for a frame, which broke the guarantee that normal speech
    streams through untouched.
    """
    if at == 0:
        return True
    before = text[at - 1]
    return not (before.isalnum() or before == "_")


def _pending_suffix(text: str, lowered: str) -> int:
    """How many trailing characters to hold back rather than speak yet.

    Two reasons to hold. A suffix that is a proper prefix of a marker ("end_c") may become
    one on the next token. And a trailing label ("node: ") may turn out to be sitting in
    front of one — that second case is the one that matters, because the tokens arrive in
    separate frames, so by the time `end_call` is recognised the label has already gone to
    the TTS and cannot be taken back.

    Costs one frame of delay, and only on text ending in ':' '=' or an opening bracket.
    """
    held = 0
    for marker in _BARE_MARKERS:
        length = min(len(marker) - 1, len(lowered))
        while length > held:
            if lowered.endswith(marker[:length]) and _starts_a_word(lowered, len(lowered) - length):
                held = length
                break
            length -= 1
    label = _TRAILING_LABEL.search(text)
    if label:
        held = max(held, len(text) - label.start())
    return held


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
        lowered = rest.lower()
        bare = _earliest_bare_marker(lowered)
        idx = rest.find("<")

        # The bracket scanner runs first only where a bracket actually comes first. On a tie
        # the bare marker wins: that tie is '<|', whose word classify_bracket does not know.
        if idx != -1 and (bare == -1 or idx < bare):
            verdict = classify_bracket(rest[idx:])
            if verdict == MARKUP:
                return trim_label_tail(safe + rest[:idx]), rest[idx:], True
            if verdict == PARTIAL:
                return safe + rest[:idx], rest[idx:], False
            # A '<' that is genuinely part of speech; keep it and look past it.
            safe += rest[: idx + 1]
            rest = rest[idx + 1 :]
            continue

        if bare != -1:
            return trim_label_tail(safe + rest[:bare]), rest[bare:], True

        held = _pending_suffix(rest, lowered)
        if held:
            return safe + rest[: len(rest) - held], rest[len(rest) - held :], False
        return safe + rest, "", False


def extract_closing_line(markup: str) -> Optional[str]:
    match = _CLOSING_LINE.search(markup)
    if not match:
        return None
    try:
        return match.group("text").encode().decode("unicode_escape")
    except Exception:
        return match.group("text")


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
