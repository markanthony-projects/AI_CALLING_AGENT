"""Whether the agent waits for the rest of the sentence, or answers half of it.

From a live call on 3 Sep 2026, one question and three answers:

    USER  "investment"       -> AGENT "That is a great goal..."
    USER  "is good or"       -> AGENT "It is a very good choice..."
    USER  "self is good?"    -> AGENT "It is wonderful for self use..."

Every pause outlasted TURN_SETTLE_SECS, so each fragment closed a turn. TurnFinalityGate
does not cover this and never claimed to — it holds a reply generated while the prospect is
still audible, and here they had stopped.

The switch is off by default, so most of these tests are about that default holding: a call
that nobody has reconfigured must behave exactly as it did before this module existed.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest


def fake(**overrides):
    base = dict(SMART_TURN_ENABLED=False, SMART_TURN_CPU_COUNT=1, SMART_TURN_STOP_SECS=2.0)
    base.update(overrides)
    return SimpleNamespace(**base)


# --- the default has to be invisible --------------------------------------------------------


def test_it_is_off_unless_somebody_turns_it_on():
    """It was tried before and ruled "Maybe around in 2" a finished turn on PSTN. This is a
    later build, but that has to be earned on measured calls rather than assumed."""
    from app.core.config import Settings

    assert Settings.model_fields["SMART_TURN_ENABLED"].default is False


def test_off_returns_none_which_is_what_the_transport_had_before():
    from app.utils.turn_analyzer import build_turn_analyzer

    assert build_turn_analyzer("sid", fake()) is None


def test_the_transport_is_given_whatever_it_returns():
    """None or an analyzer — either way the transport takes it, so turning this on is a
    setting rather than a deploy."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    passed = [
        ast.unparse(k.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "FastAPIWebsocketParams"
        for k in n.keywords
        if k.arg == "turn_analyzer"
    ]
    assert passed == ["turn_analyzer"], passed

    built = [
        ast.unparse(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "turn_analyzer" for t in n.targets)
    ]
    assert built == ["build_turn_analyzer(call_sid, settings)"], built


# --- and when it is on ----------------------------------------------------------------------


def test_on_builds_the_local_model():
    """Local, not the HTTP one: a network round trip on the turn path is the opposite of
    what this is for. The model ships inside the package, so nothing is downloaded."""
    from app.utils.turn_analyzer import build_turn_analyzer

    analyzer = build_turn_analyzer("sid", fake(SMART_TURN_ENABLED=True))
    assert analyzer is not None
    assert type(analyzer).__name__ == "LocalSmartTurnAnalyzerV3"


def test_the_backstop_is_carried_through():
    """A prospect who trails off mid-sentence must not hold the line open forever, so the
    model's "not finished yet" has a limit."""
    from app.utils.turn_analyzer import build_turn_analyzer

    analyzer = build_turn_analyzer("sid", fake(SMART_TURN_ENABLED=True, SMART_TURN_STOP_SECS=1.5))
    assert analyzer.params.stop_secs == 1.5


def test_the_backstop_is_tighter_than_pipecats_default():
    """Pipecat ships 3s. That is a long time to hold a phone line open on a maybe."""
    from app.core.config import Settings

    assert Settings.model_fields["SMART_TURN_STOP_SECS"].default < 3


def test_a_model_that_will_not_load_costs_the_feature_and_not_the_call():
    """Falling back to the silence timer is a worse conversation. Raising here would be no
    conversation at all, on every call, until somebody noticed."""
    import builtins

    import app.utils.turn_analyzer as module

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        # Stands in for a missing onnxruntime, a corrupt model file, or a Pipecat upgrade
        # that moved the class — every one of which reaches this code as an exception.
        if "smart_turn" in name:
            raise RuntimeError("no onnxruntime here")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = refuse
    try:
        assert module.build_turn_analyzer("sid", fake(SMART_TURN_ENABLED=True)) is None
    finally:
        builtins.__import__ = real_import


@pytest.mark.parametrize("cpus", [1, 2])
def test_the_thread_count_is_configurable(cpus):
    """Every concurrent call runs its own inference, so this is a real dial on a small box."""
    from app.utils.turn_analyzer import build_turn_analyzer

    assert build_turn_analyzer("sid", fake(SMART_TURN_ENABLED=True, SMART_TURN_CPU_COUNT=cpus))


def test_it_listens_to_audio_rather_than_the_transcript():
    """The reason this can go in before the speech-recognition work rather than after it.
    On the same call that split the sentence, "3 crore" came back as "3 year"; a model
    reading the transcript would inherit that, and this one does not."""
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

    src = inspect.getsource(LocalSmartTurnAnalyzerV3._predict_endpoint)
    assert "audio_array" in src
    assert "transcript" not in src
