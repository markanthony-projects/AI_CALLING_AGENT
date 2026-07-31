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
    assert getattr(start.elts[0].func, "id", None) == "MinWordsUserTurnStartStrategy"


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
    """Tunable against real calls without a redeploy, like the VAD thresholds beside it."""
    min_words = next(
        kw.value
        for kw in _call_named("MinWordsUserTurnStartStrategy").keywords
        if kw.arg == "min_words"
    )
    assert ast.unparse(min_words) == "settings.INTERRUPT_MIN_WORDS"


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
