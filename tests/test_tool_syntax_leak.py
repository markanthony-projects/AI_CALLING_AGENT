"""The caller must never hear a tool call read out loud.

From a live call:

    Agent: "Chandr, no problem at all! ... have a wonderful day.
            <function=end_call{"closing_line":"Understood, thank you so much for your
            time, have a wonderful day."}></function>"

Groq's Llama models sometimes write the tool call into the content channel instead of
emitting a structured tool_call, and Pipecat forwards content straight to TTS.
"""

import asyncio

import pytest

from app.utils.spoken_text import (
    MARKUP,
    PARTIAL,
    TEXT,
    ToolSyntaxFilter,
    classify_bracket,
    extract_closing_line,
    split_speakable,
)

LEAK = (
    'Chandr, no problem at all! Thank you so much for your time, have a wonderful day. '
    '<function=end_call{"closing_line":"Understood, thank you so much for your time, '
    'have a wonderful day."}></function>'
)


# --- the splitter ------------------------------------------------------------------


def test_the_spoken_half_survives_and_the_markup_does_not():
    speak, held, started = split_speakable(LEAK)
    assert started
    assert "<function" not in speak
    assert speak.startswith("Chandr, no problem at all!")
    assert held.startswith("<function=end_call")


@pytest.mark.parametrize(
    "text",
    [
        "Perfect, so Saturday at 11 AM. I will send you the details.",
        "The 2 B H K starts at 1.2 Crores.",
        "",
    ],
)
def test_ordinary_speech_passes_through_untouched(text):
    """Every normal reply must stream with no added latency and no rewriting."""
    assert split_speakable(text) == (text, "", False)


def test_a_less_than_sign_in_speech_is_not_markup():
    speak, held, started = split_speakable("Budget < 1 Crore is fine")
    assert not started and held == "" and speak == "Budget < 1 Crore is fine"


def test_markup_split_across_streaming_chunks_is_still_caught():
    """The leak arrives one token at a time; '<fun' alone must not be spoken and released."""
    speak, held, started = split_speakable("have a good day. <fun")
    assert not started, "not yet decidable"
    assert speak == "have a good day. "
    assert held == "<fun", "a partial marker must be held back, not spoken"

    speak2, held2, started2 = split_speakable(held + 'ction=end_call{"a":1}>')
    assert started2 and speak2 == ""


@pytest.mark.parametrize(
    "fragment,verdict",
    [
        ("<function=end_call", MARKUP),
        ("</function>", MARKUP),
        ("<tool_call>", MARKUP),
        ("<|python_tag|>", MARKUP),
        ("<f", PARTIAL),
        ("<", PARTIAL),
        ("<3 crores", TEXT),
        ("< 1 Crore", TEXT),
    ],
)
def test_bracket_classification(fragment, verdict):
    assert classify_bracket(fragment) == verdict


def test_the_closing_line_is_recovered_from_the_leak():
    assert extract_closing_line(LEAK) == (
        "Understood, thank you so much for your time, have a wonderful day."
    )


def test_no_closing_line_is_not_an_error():
    assert extract_closing_line("<tool_call>{}</tool_call>") is None


# --- the processor -----------------------------------------------------------------


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _run(filter_, chunks):
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
    from pipecat.processors.frame_processor import FrameDirection

    captured = _Captured()
    filter_.push_frame = captured.push
    # The base class needs a started processor; bypass it, the logic under test is ours.
    filter_.process_frame = filter_.__class__.process_frame.__get__(filter_)

    async def noop(*a, **kw):
        pass

    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        await filter_.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for chunk in chunks:
            await filter_.process_frame(LLMTextFrame(chunk), FrameDirection.DOWNSTREAM)
        await filter_.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


async def test_the_filter_never_pushes_markup_downstream():
    spoken = []
    filt = ToolSyntaxFilter("sid")
    frames = await _run(filt, [LEAK[:40], LEAK[40:80], LEAK[80:]])
    for f in frames:
        text = getattr(f, "text", "")
        spoken.append(text)
        assert "<function" not in text and "closing_line" not in text
    assert "".join(spoken).strip().endswith("have a wonderful day.")


async def test_a_leaked_end_call_still_hangs_up():
    seen = {}

    async def on_leak(line, already_spoke):
        seen["line"] = line
        seen["spoke"] = already_spoke

    filt = ToolSyntaxFilter("sid", on_leaked_end_call=on_leak)
    await _run(filt, [LEAK])
    assert seen["spoke"] is True, "a goodbye was already spoken; do not speak a second one"
    assert "Understood" in seen["line"]


async def test_a_leak_with_nothing_spoken_carries_the_closing_line():
    seen = {}

    async def on_leak(line, already_spoke):
        seen["line"] = line
        seen["spoke"] = already_spoke

    filt = ToolSyntaxFilter("sid", on_leaked_end_call=on_leak)
    await _run(filt, ['<function=end_call{"closing_line":"Thank you, have a good day."}>'])
    assert seen["spoke"] is False, "nothing was said, so the closing line must be spoken"
    assert seen["line"] == "Thank you, have a good day."


async def test_a_clean_response_never_fires_the_hangup():
    fired = []

    async def on_leak(line, already_spoke):
        fired.append(line)

    filt = ToolSyntaxFilter("sid", on_leaked_end_call=on_leak)
    frames = await _run(filt, ["Would you like to ", "visit the site once?"])
    assert not fired
    assert "".join(getattr(f, "text", "") for f in frames) == "Would you like to visit the site once?"


async def test_a_trailing_partial_bracket_is_flushed_not_swallowed():
    """'<' at the very end was never markup; dropping it would lose real words."""
    filt = ToolSyntaxFilter("sid")
    frames = await _run(filt, ["Budget under <"])
    assert "".join(getattr(f, "text", "") for f in frames) == "Budget under <"


async def test_the_filter_is_in_the_pipeline_between_llm_and_tts():
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    stages = next(
        ast.unparse(n.value.args[0])
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "pipeline" for t in n.targets)
    )
    names = [s.strip() for s in stages.strip("[]").split(",")]
    assert "tool_syntax_filter" in names, "leaked markup would reach the TTS again"
    assert names.index("llm") < names.index("tool_syntax_filter") < names.index("tts")


# --- script reaching the voice engine -----------------------------------------------
#
# A caller reported hearing sounds like "hein" and "auugh" in audio whose transcript is
# clean ASCII throughout. The transcript cannot settle it: the only agent text in the logs
# is the aggregated turn, recorded downstream of this processor, so anything spoken and
# then dropped leaves no trace. On that same call the prospect spoke Hindi ("हां",
# "बैंगलोर") and it entered the context in Devanagari, which is exactly the condition the
# prompt's Latin-script rule exists for and which nothing verified.


def test_devanagari_on_its_way_to_tts_is_reported():
    from app.utils.spoken_text import non_latin_letters

    assert non_latin_letters("Namaste नमस्ते") != ""


def test_each_stray_character_is_named_once():
    """A whole garbled sentence would otherwise put every repeat in the log line, and the
    useful part — which scripts leaked — gets lost in it."""
    from app.utils.spoken_text import non_latin_letters

    assert non_latin_letters("क क क ख क") == "कख"


@pytest.mark.parametrize("ordinary", ["That works well, Chandan.", "1.17 Crores", "3 B H K"])
def test_ordinary_english_is_not_flagged(ordinary):
    from app.utils.spoken_text import non_latin_letters

    assert non_latin_letters(ordinary) == ""


@pytest.mark.parametrize("punctuation", ["it is — really — fine", "\u201cquoted\u201d", "\u20b9 50"])
def test_punctuation_and_symbols_do_not_cry_wolf(punctuation):
    """An em dash, a curly quote and a rupee sign are all non-ASCII and all harmless.
    A check that fired on those would be muted within a day."""
    from app.utils.spoken_text import non_latin_letters

    assert non_latin_letters(punctuation) == ""


def test_the_stray_characters_are_named_in_the_report():
    """"contains non-Latin script" is not actionable; the characters are."""
    from app.utils.spoken_text import non_latin_letters

    assert non_latin_letters("hello नमस्ते world") .startswith("न")


def _warnings_while(chunks):
    """Drive the real filter over `chunks` and return (spoken text, warning lines).

    Behavioural rather than source-reading: an early return placed above the logger, or a
    deleted de-duplication guard, both leave every searched-for string in the source and
    are invisible to a test that only greps it.
    """
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        frames = asyncio.run(_run(ToolSyntaxFilter("sid"), chunks))
    finally:
        logger.remove(sink)
    spoken = "".join(getattr(f, "text", "") for f in frames)
    return spoken, [w for w in seen if "Non-Latin" in w]


def test_it_reports_rather_than_strips():
    """Removing the characters would leave a half-word, worse than the original, and would
    hide the prompt bug that produced them."""
    spoken, warned = _warnings_while(["Namaste ", "नमस्ते", " Mayur"])
    assert len(warned) == 1
    assert spoken == "Namaste नमस्ते Mayur", "what is spoken must not be altered"


def test_it_reports_once_per_response_not_once_per_token():
    """Streaming hands this processor a few characters at a time; one line per fragment
    would bury the call's own log."""
    _, warned = _warnings_while(["नम", "स्ते", " and ", "बैंगलोर"])
    assert len(warned) == 1


def test_a_clean_english_response_is_silent():
    spoken, warned = _warnings_while(["That works well, ", "Chandan. ", "Prices start at 1.17 Crores."])
    assert warned == []
    assert spoken.startswith("That works well")


def test_the_report_names_the_characters():
    """"contains non-Latin script" is not actionable; the characters are.

    Asserted against the exact field rather than "is the character somewhere in the line" —
    the warning also quotes the offending sentence, so the looser check passed even with
    the characters field deleted.
    """
    from app.utils.spoken_text import non_latin_letters

    _, warned = _warnings_while(["hello नमस्ते"])
    assert f"({non_latin_letters('hello नमस्ते')!r})" in warned[0]


def test_a_text_frame_before_any_response_start_does_not_crash():
    """Nothing guarantees the ordering, and an AttributeError here kills the call outright.
    Every piece of per-response state has to exist from construction, not from the first
    LLMFullResponseStartFrame."""
    from pipecat.frames.frames import LLMTextFrame
    from pipecat.processors.frame_processor import FrameDirection

    filt = ToolSyntaxFilter("sid")
    filt.push_frame = _Captured().push
    filt.process_frame = filt.__class__.process_frame.__get__(filt)

    import pipecat.processors.frame_processor as fp

    original = fp.FrameProcessor.process_frame

    async def noop(*a, **kw):
        pass

    fp.FrameProcessor.process_frame = noop
    try:
        asyncio.run(filt.process_frame(LLMTextFrame("नमस्ते"), FrameDirection.DOWNSTREAM))
    finally:
        fp.FrameProcessor.process_frame = original
