"""What it takes for the prospect to cut the agent off.

The greeting was being killed 0.7s in. Pipecat's default start strategies are
[VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy], and VAD starts the user's
turn on the first syllable of sound — so the "Hello?" everyone says on picking up the phone
counted as a barge-in. In the call this came from, the prospect never heard who was calling
and asked "Who are you ma'am?" two turns later.

MinWordsUserTurnStartStrategy is Pipecat's own answer: while the bot is speaking it takes
min_words to interrupt, and once the bot stops a single word starts the turn, so replies
stay instant.
"""

import ast
import asyncio
import inspect
from pathlib import Path

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy

from app.core.config import Settings

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")

BASE = dict(
    API_KEY="k" * 32,
    CALL_TOKEN_SECRET="s" * 32,
    DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
    OPENAI_API_KEY="x",
    SARVAM_API_KEY="x",
)


def _speech(text: str) -> InterimTranscriptionFrame:
    """Interim, not final: this has to be decided before the sentence is complete or the
    interruption arrives after the agent has already talked over them."""
    return InterimTranscriptionFrame(text, "user", "2026-07-31T00:00:00Z")


async def _feed(strategy, *texts, bot_speaking: bool):
    await strategy.process_frame(
        BotStartedSpeakingFrame() if bot_speaking else BotStoppedSpeakingFrame()
    )
    return [await strategy.process_frame(_speech(t)) for t in texts]


# --- the strategy itself ------------------------------------------------------------


@pytest.mark.parametrize("said", ["Hello", "Hello?", "haan", "haan boliye", "hmm okay"])
def test_a_short_reply_does_not_interrupt_the_agent(said):
    """Pickup noise and back-channel. Neither is a request for the agent to stop."""
    results = asyncio.run(_feed(MinWordsUserTurnStartStrategy(min_words=3), said, bot_speaking=True))
    assert results == [ProcessFrameResult.CONTINUE]


@pytest.mark.parametrize(
    "said",
    [
        "ek minute ruko please",
        "kaun bol raha hai",
        "no I am not interested",
    ],
)
def test_a_real_interruption_still_lands(said):
    """The prospect must always be able to stop the agent — this cannot become a bot that
    talks over people."""
    strategy = MinWordsUserTurnStartStrategy(min_words=3)
    assert asyncio.run(_feed(strategy, said, bot_speaking=True)) == [ProcessFrameResult.STOP]


@pytest.mark.parametrize("said", ["Hello", "haan", "yes"])
def test_one_word_answers_the_agent_when_it_is_not_speaking(said):
    """The gate is only against interrupting. A one-word answer to a question the agent
    just finished asking must not be swallowed, or every "haan" would go unheard."""
    strategy = MinWordsUserTurnStartStrategy(min_words=3)
    assert asyncio.run(_feed(strategy, said, bot_speaking=False)) == [ProcessFrameResult.STOP]


def test_the_gate_reopens_once_the_agent_stops_talking():
    """One strategy instance lives for the whole call, so the bot-speaking flag has to
    track both edges. If it stuck on, every short answer for the rest of the call would be
    dropped."""
    strategy = MinWordsUserTurnStartStrategy(min_words=3)
    assert asyncio.run(_feed(strategy, "Hello", bot_speaking=True)) == [ProcessFrameResult.CONTINUE]
    assert asyncio.run(_feed(strategy, "Hello", bot_speaking=False)) == [ProcessFrameResult.STOP]


# --- how the agent wires it ---------------------------------------------------------


def _call_named(name: str) -> ast.Call:
    for node in ast.walk(ast.parse(AGENT_SRC)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name:
            return node
    raise AssertionError(f"agent.py never calls {name}()")


def test_the_aggregator_is_given_the_strategy():
    """Configuring it anywhere else has no effect — this is the object that owns the turn
    controller."""
    params = next(kw.value for kw in _call_named("LLMUserAggregator").keywords if kw.arg == "params")
    assert getattr(params.func, "id", None) == "LLMUserAggregatorParams"
    assert any(kw.arg == "user_turn_strategies" for kw in params.keywords)


def test_the_strategy_replaces_the_defaults_rather_than_joining_them():
    """The controller runs every start strategy and any one of them firing starts the turn.
    Appending to the defaults would leave VAD in place and change nothing at all."""
    start = next(
        kw.value for kw in _call_named("UserTurnStrategies").keywords if kw.arg == "start"
    )
    assert isinstance(start, ast.List), "start must be an explicit list, not the defaults"
    assert len(start.elts) == 1, "another start strategy would fire on its own and defeat this"
    assert ast.unparse(start.elts[0]) == "greeting_gate"


def test_vad_is_not_reintroduced_as_a_start_strategy():
    """Checked against the parsed tree, not the text: the comment above the call names the
    class, and a substring search would pass on that alone."""
    tree = ast.parse(AGENT_SRC)
    constructed = {
        getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "VADUserTurnStartStrategy" not in constructed
    imported = {
        alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for alias in n.names
    }
    assert "VADUserTurnStartStrategy" not in imported


def test_the_word_count_is_configurable():
    """Tunable against real calls without a redeploy, like the VAD thresholds beside it.

    Built in stt_provider since 11 Sep, not in the agent: which gate a call gets depends on
    whether its speech service produces words before the turn ends, so it is chosen where
    the service is."""
    import inspect as _inspect

    from app.services import stt_provider

    assert "settings.INTERRUPT_MIN_WORDS" in _inspect.getsource(stt_provider._word_gate)


def test_a_service_that_says_nothing_until_the_end_gets_a_gate_that_needs_no_words():
    """Flux and Sarvam both push no interim transcript. A word gate would have nothing to
    count until the caller had finished — meaning the agent talks over all of it."""
    from app.services.stt_provider import _BARGE_IN, DEEPGRAM, FLUX, PROVIDERS, SARVAM

    assert set(_BARGE_IN) == set(PROVIDERS), "a provider with no gate raises on a live call"
    assert _BARGE_IN[DEEPGRAM].__name__ == "_word_gate"
    for quiet in (SARVAM, FLUX):
        assert _BARGE_IN[quiet].__name__.startswith("_duration_gate")


# --- the setting --------------------------------------------------------------------


def test_three_words_by_default():
    assert Settings(**BASE).INTERRUPT_MIN_WORDS == 3


def test_zero_is_rejected():
    """Zero would mean an empty transcript interrupts, which is worse than the bug."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(**BASE, INTERRUPT_MIN_WORDS=0)


def test_one_is_allowed_because_it_is_the_old_behaviour():
    assert Settings(**BASE, INTERRUPT_MIN_WORDS=1).INTERRUPT_MIN_WORDS == 1


# --- the parameter that never did anything ------------------------------------------


def test_allow_interruptions_is_not_a_pipeline_parameter():
    """It was passed for months and silently dropped: PipelineParams has no such field and
    pydantic ignores extras. Anyone re-adding it would think interruptions were configured."""
    assert "allow_interruptions" not in PipelineParams.model_fields
    params = PipelineParams(allow_interruptions=True)
    assert not params.model_extra
    assert "allow_interruptions" not in params.model_dump()


def test_the_agent_no_longer_passes_it():
    tree = ast.parse(AGENT_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PipelineParams":
            assert not any(kw.arg == "allow_interruptions" for kw in node.keywords)


def test_the_strategy_is_the_one_the_installed_pipecat_ships():
    """Guards an upgrade quietly changing the semantics this whole fix rests on."""
    source = inspect.getsource(MinWordsUserTurnStartStrategy._handle_transcription)
    assert "self._min_words if self._bot_speaking else 1" in source


# --- the gate lifts once the greeting is out ----------------------------------------
#
# A flat word gate cost a live caller 37 seconds of silence. "Yeah sure." answering
# "Would you like to visit the site?" is two words, so while the bot's audio was still
# playing it was discarded outright — not deferred, discarded — and Mark said "Hello"
# twice before the agent responded at all.


def _gate():
    from app.services.turns import GreetingOnlyMinWords

    return GreetingOnlyMinWords(min_words=3)


@pytest.mark.parametrize("said", ["Yeah sure.", "Yeah,", "Hello?", "Yes please", "Okay"])
def test_while_the_greeting_plays_a_short_reply_still_does_not_interrupt(said):
    gate = _gate()
    assert asyncio.run(_feed(gate, said, bot_speaking=True)) == [ProcessFrameResult.CONTINUE]


@pytest.mark.parametrize("said", ["Yes please", "3 BHK", "Sunday", "Not interested", "nahi"])
def test_after_the_greeting_the_same_reply_gets_through(said):
    """The exact replies people give to a closing question. Dropping these is worse than
    any barge-in the gate was protecting against."""
    gate = _gate()
    gate.relax()
    assert asyncio.run(_feed(gate, said, bot_speaking=True)) == [ProcessFrameResult.STOP]


@pytest.mark.parametrize("said", ["Yeah sure.", "Yeah,", "Okay", "haan", "theek hai", "हाँ"])
def test_agreeing_along_waits_for_the_sentence_instead_of_cutting_it(said):
    """These moved out of the test above on 11 Sep 2026, and it is not a loosening.

    They were never getting through: under min_words while the bot speaks, the base class
    DISCARDS — which is the 37-second bug itself, and no relax() saves them while the bot is
    still playing out. Held, they survive and are answered the moment the sentence ends. The
    difference that matters is dropped-versus-kept, and this is the kept half.

    The reason they are held at all is that on a Hindi call this is how somebody listens.
    "haan haan", "achha", "theek hai" run right through a description, and an agent that
    stops for each of them is the naive one.
    """
    gate = _gate()
    gate.relax()
    started, dropped = [], []
    gate.trigger_user_turn_started = lambda *a, **k: _record(started)
    gate.trigger_reset_aggregation = lambda *a, **k: _record(dropped)

    async def run():
        await gate.process_frame(BotStartedSpeakingFrame())
        assert await gate.process_frame(_speech(said)) == ProcessFrameResult.CONTINUE
        assert dropped == [], "the words were thrown away"
        assert started == [], "it cut the sentence off"
        await gate.process_frame(BotStoppedSpeakingFrame())
        assert started == [True], "the sentence ended and nobody answered them"

    asyncio.run(run())


# --- except the one word that is not a reply at all ----------------------------------
#
# Call 5023ff25. The agent went mute for the last thirty seconds and the prospect hung up:
#
#     USER  "Hello."   TTS reconnected (3)   AGENT "Hi, could you..."     [interrupted]
#     USER  "Hello."   TTS reconnected (4)   AGENT "Which BHK size..."    [interrupted]
#
# Sarvam reopens its websocket on every interruption. They were saying "Hello?" because they
# could not hear, and each "Hello?" was what stopped them hearing.


@pytest.mark.parametrize("said", ["Hello?", "Hello", "hello.", "sir", "Ma'am", "Are you there?"])
def test_checking_the_line_does_not_cut_the_agent_off(said):
    """It is not somebody taking the floor. It is somebody who cannot hear, and the worst
    possible answer is to stop talking."""
    gate = _gate()
    gate.relax()
    assert asyncio.run(_feed(gate, said, bot_speaking=True)) == [ProcessFrameResult.CONTINUE]


@pytest.mark.parametrize("said", ["Hello?", "Hello", "sir", "Are you there?"])
def test_the_same_words_into_silence_are_answered_at_once(said):
    """The suppression is only about not CUTTING SOMEBODY OFF. With the agent already quiet
    there is nothing to protect, and deferring would mean waiting for a BotStoppedSpeaking
    that is never coming — a prospect saying "Hello?" into silence and getting nothing back,
    which is the 37 seconds this whole gate exists to prevent.

    Mutation testing found this: every other test here has the agent speaking."""
    gate = _gate()
    gate.relax()
    assert asyncio.run(_feed(gate, said, bot_speaking=False)) == [ProcessFrameResult.STOP]


def test_it_is_held_rather_than_dropped_and_answered_when_the_sentence_ends():
    """The whole reason the gate was relaxed to one word: at three, Pipecat DISCARDS what it
    will not act on — trigger_reset_aggregation — and a caller sat through 37 seconds of
    silence. Suppressing the interruption must not bring that back. The sentence finishes,
    and then they are answered."""
    gate = _gate()
    gate.relax()
    started = []
    gate.trigger_user_turn_started = lambda *a, **k: _record(started)
    dropped = []
    gate.trigger_reset_aggregation = lambda *a, **k: _record(dropped)

    async def run():
        await gate.process_frame(BotStartedSpeakingFrame())
        assert await gate.process_frame(_speech("Hello?")) == ProcessFrameResult.CONTINUE
        assert dropped == [], "the words were thrown away"
        assert started == [], "it interrupted after all"
        await gate.process_frame(BotStoppedSpeakingFrame())
        assert started == [True], "the sentence ended and nobody answered them"

    asyncio.run(run())


def test_a_real_reply_after_a_held_line_check_is_not_delayed_twice():
    """They say "Hello?", then say something real while the agent is still talking. The real
    words take the floor at once and must not still be marked as held."""
    gate = _gate()
    gate.relax()
    started = []
    gate.trigger_user_turn_started = lambda *a, **k: _record(started)

    async def run():
        await gate.process_frame(BotStartedSpeakingFrame())
        await gate.process_frame(_speech("Hello?"))
        assert await gate.process_frame(_speech("I want a 3 BHK")) == ProcessFrameResult.STOP
        assert started == [True]
        await gate.process_frame(BotStoppedSpeakingFrame())
        assert started == [True], "the held flag fired a second turn nobody asked for"

    asyncio.run(run())


async def _record(sink):
    sink.append(True)


def test_relaxing_is_permanent():
    """It guards one line. Re-arming it mid-call would bring the dead air back."""
    gate = _gate()
    gate.relax()
    asyncio.run(_feed(gate, "Sunday.", bot_speaking=False))
    assert asyncio.run(_feed(gate, "Sunday.", bot_speaking=True)) == [ProcessFrameResult.STOP]


def test_the_gate_is_relaxed_when_an_assistant_turn_finishes():
    """Wiring, not behaviour: a gate that is never relaxed is the original bug."""
    tree = ast.parse(AGENT_SRC)
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "on_assistant_turn_stopped"
    )
    assert "greeting_gate.relax()" in ast.unparse(handler)


def test_the_gate_is_relaxed_when_the_prospect_speaks_first():
    """Then there is no greeting at all, so nothing is being protected — but the gate would
    stay armed for the whole call and eat every short answer."""
    tree = ast.parse(AGENT_SRC)
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "startup_greeting"
    )
    assert "greeting_gate.relax()" in ast.unparse(handler)
