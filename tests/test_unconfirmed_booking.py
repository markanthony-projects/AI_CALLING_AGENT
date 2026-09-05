"""The call where a visit was booked and the prospect was never told.

Live call 92b7c253, 5 Sep 2026. The best call the system has had — 15 turns, a real
conversation, a site visit agreed — and it ended like this:

    07:39:47  USER  → "Tuesday"
    07:39:47  USER  → "8PM"
    07:39:48  ERROR  Model wrote tool syntax into its spoken reply; suppressed before TTS:
                     'end_call closing_line="Perfect, so Tuesday at 8 PM at Abhee Codename
                      New Dimension. I will send you the details. Thank you for your time,
                      Rahul."/>'
    07:39:48  Ending call after leaked end_call syntax (goodbye already spoken)
    07:39:52  AGENT → "Got it. Tuesday at 8 PM. I will book that for you. <call:" [interrupted]
    07:39:53  Pipeline stopped

Two failures in four seconds.

The read-back the model had written was correct and complete, and it was thrown away —
because the handler was told only that the agent HAD spoken, and read that as "a goodbye
already went out". It had not. What went out was half a confirmation with a piece of markup
on the end. The prompt is blunt about the cost: "A prospect never told the booking is
confirmed does not turn up."

And "<call:" was spoken at all, because the markup list knew function, tool_call, invoke and
antml but not call.
"""

import asyncio
import inspect

import pytest

from app.utils.spoken_text import ToolSyntaxFilter, sounds_like_goodbye

# Verbatim from the call, with the markup the model actually emitted.
LEAK = (
    "Got it. Tuesday at 8 PM. I will book that for you. "
    '<call:end_call closing_line="Perfect, so Tuesday at 8 PM at Abhee Codename New '
    'Dimension. I will send you the details. Thank you for your time, Rahul."/>'
)


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _run(filter_, chunks):
    import pipecat.processors.frame_processor as fp
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection

    captured = _Captured()
    filter_.push_frame = captured.push
    filter_.process_frame = filter_.__class__.process_frame.__get__(filter_)

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        await filter_.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
        for chunk in chunks:
            await filter_.process_frame(LLMTextFrame(chunk), FrameDirection.DOWNSTREAM)
        await filter_.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return "".join(getattr(f, "text", "") for f in captured.frames)


# --- what the caller hears ----------------------------------------------------------------


# The same leak as the model actually streams it. This is not decoration: given the whole
# string at once the filter classifies the bracket in one go and the bare end_call marker
# suppresses everything after it. Arriving a token at a time, "<", "call", ":" are each
# judged alone — and each is ordinary text unless the list knows the word. The single-chunk
# form of this test passed with and without the fix.
STREAMED = [
    "Got it. Tuesday at 8 PM. ",
    "I will book that for you. ",
    "<", "call", ":", "end", "_call", " closing_line=",
    '"Perfect, so Tuesday at 8 PM at Abhee Codename New Dimension."', "/>",
]


def test_the_markup_is_never_spoken():
    """"<call:" reached the voice engine on a live call, one token at a time."""
    spoken = asyncio.run(_run(ToolSyntaxFilter("sid"), STREAMED))
    assert "<call" not in spoken, spoken
    assert "<" not in spoken, spoken
    assert "closing_line" not in spoken
    assert "end_call" not in spoken


def test_the_confirmation_in_front_of_it_survives_the_stream():
    spoken = asyncio.run(_run(ToolSyntaxFilter("sid"), STREAMED))
    assert spoken.strip() == "Got it. Tuesday at 8 PM. I will book that for you."


def test_the_read_back_is_recovered_from_the_streamed_markup():
    """The closing line the model wrote is the only complete confirmation on the call. It has
    to survive being taken apart into tokens."""
    seen = {}

    async def on_leak(line, spoken_already):
        seen["line"], seen["spoken"] = line, spoken_already

    asyncio.run(_run(ToolSyntaxFilter("sid", on_leaked_end_call=on_leak), STREAMED))
    assert seen["line"] == "Perfect, so Tuesday at 8 PM at Abhee Codename New Dimension."
    assert not sounds_like_goodbye(seen["spoken"]), "so the read-back must still be spoken"


def test_the_words_in_front_of_it_still_are():
    """Suppression starts at the markup, not before it. Losing the confirmation as well
    would cost more than the markup did."""
    spoken = asyncio.run(_run(ToolSyntaxFilter("sid"), [LEAK]))
    assert "Got it. Tuesday at 8 PM. I will book that for you." in spoken


def test_an_angle_bracket_in_real_speech_is_not_eaten():
    """"Budget under < 1 Crore" is a sentence. The list is checked after a '<', so adding
    "call" to it must not start swallowing ordinary words."""
    spoken = asyncio.run(_run(ToolSyntaxFilter("sid"), ["Budget under < 1 Crore is fine."]))
    assert spoken == "Budget under < 1 Crore is fine."


def test_the_word_call_on_its_own_is_ordinary_speech():
    """The agent says it constantly — "our team will call you", "should I call at 6 PM".
    Only a '<' in front of it makes it markup."""
    line = "Our property expert will call you with better options."
    assert asyncio.run(_run(ToolSyntaxFilter("sid"), [line])) == line


# --- whether the prospect has actually been said goodbye to -------------------------------


@pytest.mark.parametrize(
    "spoken",
    [
        "Chandr, no problem at all! Thank you so much for your time, have a wonderful day.",
        "Thank you for your time. Have a great day!",
        "Goodbye.",
        "Sure. Take care.",
    ],
)
def test_a_real_sign_off_is_recognised(spoken):
    assert sounds_like_goodbye(spoken) is True


@pytest.mark.parametrize(
    "spoken",
    [
        "Got it. Tuesday at 8 PM. I will book that for you.",
        "We have 2, 3, 3.5 and 4.5 BHK homes.",
        "Which day works best for you?",
        "",
        None,
    ],
)
def test_a_reply_that_is_not_a_sign_off_is_not_mistaken_for_one(spoken):
    """This is the whole fix. "The agent spoke" was standing in for "the prospect was said
    goodbye to", and on the call above those were not the same thing."""
    assert sounds_like_goodbye(spoken) is False


# --- and what the handler does with it ------------------------------------------------------


def test_the_handler_is_told_what_was_said_and_not_merely_that_something_was():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "async def on_leaked_end_call(line: Optional[str], spoken_already: str)" in src
    assert "sounds_like_goodbye(spoken_already)" in src


def test_the_filter_hands_over_the_words_rather_than_a_flag():
    from app.utils import spoken_text

    src = inspect.getsource(spoken_text.ToolSyntaxFilter._flush)
    assert "self._on_leaked_end_call(extract_closing_line(leaked), spoken_line)" in src
    assert "spoken_line = self.lead_in" in src


def test_the_read_back_is_spoken_when_no_goodbye_was():
    """The branch taken on the live call skips the closing line entirely, and it must be
    reachable ONLY when a farewell really did go out.

    Checked as the shape of the condition, not as a substring: "not sounds_like_goodbye(...)"
    contains "sounds_like_goodbye(...)" too, so a search finds an inverted branch just as
    happily as a correct one — and inverted, this speaks the goodbye twice on a real sign-off
    and stays silent on the call that started all this."""
    import ast

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_leaked_end_call"
    )
    guards = [
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "sounds_like_goodbye" in ast.unparse(n.test)
    ]
    assert len(guards) == 1, [ast.unparse(g.test) for g in guards]
    assert ast.unparse(guards[0].test) == "sounds_like_goodbye(spoken_already)"
    assert "goodbye already spoken" in ast.unparse(guards[0].body)
    assert not guards[0].orelse, "the skip must be the only branch this guards"


def test_the_words_are_read_before_the_response_is_reset():
    """_flush clears the response state partway through. Reading the spoken line after that
    would hand the handler an empty string and skip the closing line on every leak."""
    from app.utils import spoken_text

    src = inspect.getsource(spoken_text.ToolSyntaxFilter._flush)
    assert src.index("spoken_line = self.lead_in") < src.index("self._reset()")
