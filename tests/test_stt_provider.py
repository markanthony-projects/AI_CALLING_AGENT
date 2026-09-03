"""Which ears the agent listens with, chosen from configuration rather than from source.

The model name and language were literals inside run_voice_agent. That was fine while there
was nothing to try; it stopped being fine the moment the plan called for a three-way
comparison, because running that by editing the agent and redeploying between each attempt
measures the deploy as much as the model.

The first test here is the one that matters most: the defaults must reproduce, exactly, what
was hard-coded. A refactor of a live call path earns nothing and risks everything if it
quietly changes what the agent hears.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest

from app.services.stt_provider import (
    DEEPGRAM,
    PROVIDERS,
    SARVAM,
    SttEndpoint,
    build_stt_service,
    stt_endpoint,
)


def fake(**overrides):
    base = dict(
        STT_PROVIDER="deepgram",
        STT_MODEL="nova-2-general",
        STT_LANGUAGE="hi",
        STT_ENDPOINTING_MS=300,
        DEEPGRAM_API_KEY="test",
        SARVAM_API_KEY="test",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- nothing may have changed -------------------------------------------------------------


def test_the_defaults_are_what_used_to_be_hard_coded():
    """Verbatim from the agent before this existed:

        model="nova-2-general"
        language="hi"
        endpointing=300

    An untouched deployment has to listen with exactly this.
    """
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["STT_PROVIDER"].default == "deepgram"
    assert fields["STT_MODEL"].default == "nova-2-general"
    assert fields["STT_LANGUAGE"].default == "hi"
    assert fields["STT_ENDPOINTING_MS"].default == 300


def test_the_default_endpoint_reads_back_the_same_way():
    assert stt_endpoint(fake()) == SttEndpoint("deepgram", "nova-2-general", "hi")


def test_the_agent_no_longer_builds_its_own():
    """A second construction left behind anywhere would ignore the configuration entirely."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "build_stt_service(call_sid, settings)" in src
    assert "DeepgramSTTService(" not in src


# --- choosing ------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_supported_provider_resolves(provider):
    assert stt_endpoint(fake(STT_PROVIDER=provider)).provider == provider


def test_casing_and_stray_spaces_do_not_decide_the_vendor():
    """It arrives from an env file typed by a person."""
    assert stt_endpoint(fake(STT_PROVIDER="  Deepgram ")).provider == DEEPGRAM
    assert stt_endpoint(fake(STT_MODEL=" saaras:v3 ")).model == "saaras:v3"


@pytest.mark.parametrize("bad", ["", "  ", "deepgramm", "assemblyai", "whisper"])
def test_an_unknown_provider_is_refused_rather_than_defaulted(bad):
    """Silently falling back to Deepgram would make a comparison run report the wrong
    winner, which is worse than not running it at all."""
    with pytest.raises(ValueError) as caught:
        stt_endpoint(fake(STT_PROVIDER=bad))
    assert "STT_PROVIDER" in str(caught.value)


def test_the_endpoint_prints_as_something_a_log_can_carry():
    assert str(SttEndpoint(SARVAM, "saaras:v3", "hi")) == "sarvam/saaras:v3"


# --- what actually gets built ---------------------------------------------------------------


def test_deepgram_is_built_with_the_configured_model():
    service = build_stt_service("sid", fake(STT_MODEL="nova-3", STT_LANGUAGE="multi"))
    assert type(service).__name__ == "DeepgramSTTService"


def test_sarvam_is_built_when_it_is_the_one_configured():
    service = build_stt_service(
        "sid", fake(STT_PROVIDER="sarvam", STT_MODEL="saaras:v3", STT_LANGUAGE="hi")
    )
    assert type(service).__name__ == "SarvamSTTService"


def test_every_provider_has_a_builder():
    """PROVIDERS is what stt_endpoint accepts. A name accepted there with no builder behind
    it would pass validation and then raise KeyError on a live call."""
    from app.services.stt_provider import _BUILDERS

    assert set(_BUILDERS) == set(PROVIDERS)


def test_the_sample_rate_is_not_configurable():
    """The serializer, the VAD and the turn analyzer all assume 16kHz. A provider that
    cannot take it needs resampling, not a quiet change here."""
    from app.services import stt_provider

    tree = ast.parse(inspect.getsource(stt_provider).lstrip())
    rates = [
        ast.unparse(k.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for k in n.keywords
        if k.arg == "sample_rate"
    ]
    assert rates and set(rates) == {"SAMPLE_RATE"}, rates
