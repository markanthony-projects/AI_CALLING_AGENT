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
    trim_label_tail,
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
        "The 2 BHK starts at 1.2 Crores.",
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


@pytest.mark.parametrize("ordinary", ["That works well, Chandan.", "1.17 Crores", "3 BHK"])
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


# --- the bare form: no bracket anywhere ------------------------------------------------
#
# From a live call on 2 Sep 2026, campaign "Abhee Codename New Dimension":
#
#     AGENT → "No problem at all. I understand. Our team will send you the brochure, floor
#              plans and price details on WhatsApp. Thank you for your time.
#              node: end_call(closing_line="Thank you for your time. Have a great day!")"
#
# The prospect heard that. Everything above keys off a '<' and there is no '<' in it: no
# suppression fired, no leak was logged, and the text went to the TTS and into the context.
# The filter was written for Llama-on-Groq; the provider is Gemma-on-Cerebras now, and it
# leaks in a different shape.

BARE_LEAK = (
    "No problem at all. I understand. Our team will send you the brochure, floor plans "
    "and price details on WhatsApp. Thank you for your time. "
    'node: end_call(closing_line="Thank you for your time. Have a great day!")'
)


def test_the_production_bare_leak_is_cut():
    speak, held, started = split_speakable(BARE_LEAK)
    assert started, "a tool call with no angle bracket is still a tool call"
    assert "end_call" not in speak and "closing_line" not in speak
    assert held.startswith("end_call(")


def test_the_dangling_label_goes_too():
    """Cutting at the marker alone leaves "node: " behind, and the caller hears "node"."""
    speak, _, _ = split_speakable(BARE_LEAK)
    assert speak.endswith("Thank you for your time.")
    assert "node" not in speak


def test_the_closing_line_survives_the_bare_form():
    """The key is unquoted and the separator is '=', not ':'. A JSON-only pattern found
    nothing here, so the prospect got a generic goodbye on exactly the calls where the
    model had written a good one."""
    _, held, _ = split_speakable(BARE_LEAK)
    assert extract_closing_line(held) == "Thank you for your time. Have a great day!"


@pytest.mark.parametrize(
    "leak",
    [
        'end_call(closing_line="Bye.")',
        "end_call({'closing_line': 'Bye.'})",
        '{"name": "end_call", "arguments": {"closing_line": "Bye."}}',
        '[end_call(closing_line="Bye.")]',
        'Tool: end_call closing_line="Bye."',
        "```json\n{\"name\": \"end_call\"}\n```",
        "<|python_tag|>end_call(closing_line=\"Bye.\")",
    ],
)
def test_every_shape_of_the_call_is_suppressed(leak):
    speak, _, started = split_speakable("Thank you for your time. " + leak)
    assert started, f"not caught: {leak!r}"
    assert speak == "Thank you for your time."


def test_a_marker_split_across_streaming_frames_is_still_caught():
    """Tokens arrive separately. "end" then "_call(" must not speak the "end"."""
    filtered, buffer = [], ""
    for chunk in ["Thank you. ", "end", "_call(closing_line=", '"Bye.")']:
        buffer += chunk
        speak, held, started = split_speakable(buffer)
        filtered.append(speak)
        buffer = "" if started else held
        if started:
            break
    assert "end" not in "".join(filtered).replace("Thank you.", "")
    assert "".join(filtered).strip() == "Thank you."


def test_a_label_arriving_in_its_own_frame_is_held_back():
    """The reason the label has to be held rather than trimmed at the cut: by the time
    `end_call` arrives, a "node: " already pushed downstream is in the TTS and gone."""
    speak, held, started = split_speakable("Thank you for your time. node: ")
    assert not started
    assert speak == "Thank you for your time. "
    assert held == "node: "


async def test_the_bare_leak_never_reaches_the_tts():
    filt = ToolSyntaxFilter("sid")
    frames = await _run(filt, [BARE_LEAK[:60], BARE_LEAK[60:130], BARE_LEAK[130:]])
    spoken = "".join(getattr(f, "text", "") for f in frames)
    for forbidden in ("end_call", "closing_line", "node:", "("):
        assert forbidden not in spoken, f"caller would hear {forbidden!r} in {spoken!r}"
    assert spoken.strip().endswith("Thank you for your time.")


async def test_the_bare_leak_still_hangs_the_call_up():
    """It meant to end the call. Suppressing the text must not also lose the intent, or the
    prospect is left holding a silent line."""
    seen = {}

    async def on_leak(line, already_spoke):
        seen["line"], seen["spoke"] = line, already_spoke

    filt = ToolSyntaxFilter("sid", on_leaked_end_call=on_leak)
    await _run(filt, [BARE_LEAK])
    assert seen["spoke"] is True
    assert seen["line"] == "Thank you for your time. Have a great day!"


# --- what must NOT be cut ---------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "We have a grand clubhouse for functions and events.",
        "The call ends when you say so.",
        "Prices start at 1.17 Crores, about 20 Lakhs below the launch price.",
        "It is near Dommasandra Circle, well connected to ITPL and Whitefield.",
        "Budget < 1 Crore is fine",
        "We are open from 10 AM to 8 PM.",
        "Would 11 AM on Sunday work for you?",
        "There are 4 clubhouses and over 140 amenities.",
    ],
)
def test_ordinary_speech_is_never_cut_or_held(sentence):
    """Every marker carries an underscore or is a code fence precisely so that no sentence
    this agent could legitimately say can trip it. A false positive here is worse than the
    leak: it silently drops real speech mid-call."""
    assert split_speakable(sentence) == (sentence, "", False)


def test_a_sentence_ending_in_a_full_stop_is_not_treated_as_a_label():
    """The label trim is anchored on ':' '=' and brackets, never on '.', or every cut would
    also eat the sentence in front of it."""
    assert trim_label_tail("Prices start at 1.17 Crores.") == "Prices start at 1.17 Crores."
    assert trim_label_tail("Thank you for your time. node: ") == "Thank you for your time."
