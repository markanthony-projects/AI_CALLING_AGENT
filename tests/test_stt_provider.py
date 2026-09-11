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
from loguru import logger

from app.services.stt_provider import (
    DEEPGRAM,
    PROVIDERS,
    SARVAM,
    SttEndpoint,
    build_listening,
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
        # Read by the end-of-turn half of the plan: the timer providers need the settle
        # window, and Flux needs its two thresholds (None = the service's own defaults).
        TURN_SETTLE_SECS=0.4,
        STT_EOT_THRESHOLD=None,
        STT_EOT_TIMEOUT_MS=None,
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
    assert "build_listening(call_sid, settings)" in src
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
    plan = build_listening("sid", fake(STT_MODEL="nova-3", STT_LANGUAGE="multi"))
    assert type(plan.service).__name__ == "DeepgramSTTService"


def test_sarvam_is_built_when_it_is_the_one_configured():
    plan = build_listening(
        "sid", fake(STT_PROVIDER="sarvam", STT_MODEL="saaras:v3", STT_LANGUAGE="hi")
    )
    assert type(plan.service).__name__ == "SarvamSTTService"


# --- the two providers do not spell a language the same way -------------------------------


@pytest.mark.parametrize(
    "configured,sent",
    [("hi", "hi-IN"), ("en", "en-IN"), ("kn", "kn-IN"), ("ta", "ta-IN"), ("mr", "mr-IN")],
)
def test_sarvam_is_sent_the_dialect_it_expects(configured, sent):
    """Deepgram takes "hi". Sarvam wants "hi-IN", and given "hi" it logs "Language hi not
    verified" and sends the unverified code anyway — so the call is transcribed in whatever
    that turns out to mean, with nothing failing to say so."""
    from app.services.stt_provider import sarvam_language

    assert sarvam_language(configured) == sent


@pytest.mark.parametrize("already", ["hi-IN", "en-IN", "unknown"])
def test_a_code_that_is_already_sarvams_is_left_alone(already):
    """"unknown" is Sarvam's own default for saarika:v2.5 and means auto-detect. Mapping it
    to something would turn auto-detection off."""
    from app.services.stt_provider import sarvam_language

    assert sarvam_language(already) == already


def test_an_unmapped_code_passes_through_rather_than_vanishing():
    """The map covers the languages this system dials in. A code outside it is more likely a
    deliberate choice than a mistake, and sending an empty language is worse than sending an
    unusual one."""
    from app.services.stt_provider import sarvam_language

    assert sarvam_language("es-ES") == "es-ES"
    assert sarvam_language("  hi  ") == "hi-IN"


def test_the_mapping_is_applied_where_the_service_is_built():
    """A helper nothing calls is a helper that does not run on a call."""
    plan = build_listening(
        "sid", fake(STT_PROVIDER="sarvam", STT_MODEL="saarika:v2.5", STT_LANGUAGE="hi")
    )
    assert plan.service._settings.language == "hi-IN"


def test_deepgram_still_gets_the_plain_code():
    """The dialect belongs to Sarvam. Sending Deepgram "hi-IN" would change what it listens
    for on every existing call, for a switch nobody made."""
    plan = build_listening("sid", fake(STT_LANGUAGE="hi"))
    assert plan.service._settings.language == "hi"


def test_switching_provider_does_not_silently_change_the_language():
    """One STT_LANGUAGE feeds both, so the dialect has to live in the builder. Left in the
    env file, flipping STT_PROVIDER would also flip what language the call is in."""
    settings = fake(STT_LANGUAGE="hi")
    deepgram = build_listening("sid", settings)
    sarvam = build_listening(
        "sid", fake(STT_PROVIDER="sarvam", STT_MODEL="saarika:v2.5", STT_LANGUAGE="hi")
    )
    assert str(deepgram.service._settings.language).startswith("hi")
    assert str(sarvam.service._settings.language).startswith("hi")


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


# --- a service that decides the turn, and the plan that carries that fact ---------------------
#
# Measured on call ba83d740, 11 Sep 2026: p50 1289ms of which turn_decision was 602ms. Nearly
# half of everything the caller waits through, spent working out that they had finished — VAD
# stop_secs plus TURN_SETTLE_SECS, before a word reaches the model. Flux decides that on
# Deepgram's side and pushes the frames itself, so there is nothing left here to wait on.


def _plan(**over):
    return build_listening("sid", fake(**{"STT_PROVIDER": "flux", "STT_MODEL": "flux-general-multi", **over}))


def test_flux_is_a_provider_like_any_other():
    assert "flux" in PROVIDERS
    assert type(_plan().service).__name__ == "DeepgramFluxSTTService"


def test_a_transcribing_service_leaves_the_turn_to_the_timer():
    """Somebody has to guess, and a stopwatch is all there is."""
    for provider, model in (("deepgram", "nova-2-general"), ("sarvam", "saarika:v2.5")):
        plan = build_listening("sid", fake(STT_PROVIDER=provider, STT_MODEL=model))
        assert type(plan.stop_strategy).__name__ == "SpeechTimeoutUserTurnStopStrategy"


def test_a_service_that_knows_takes_the_turn_off_the_timer():
    """The point of the whole exercise. ExternalUserTurnStopStrategy has no
    user_speech_timeout at all — the 400ms window simply stops existing."""
    plan = _plan()
    assert type(plan.stop_strategy).__name__ == "ExternalUserTurnStopStrategy"
    assert not hasattr(plan.stop_strategy, "_user_speech_timeout")


def test_the_timer_window_comes_from_the_setting_that_names_it():
    plan = build_listening("sid", fake(TURN_SETTLE_SECS=0.9))
    assert plan.stop_strategy._user_speech_timeout == 0.9


def test_every_provider_says_who_owns_its_turns():
    """A provider with a builder and no turn owner would build a service and then raise
    KeyError on a live call — the same failure _BUILDERS is checked for above."""
    from app.services.stt_provider import _TURN_OWNERS

    assert set(_TURN_OWNERS) == set(PROVIDERS)


def test_flux_does_not_take_barge_in_away():
    """should_interrupt=False, and it is why this can be adopted one piece at a time. Left
    True, Flux interrupts the agent the moment it hears speech — and "Hello?" learning not
    to cut the agent off mid-sentence, earned on call 5023ff25, lives in the START strategy
    on our side. Flux is bought for the END of a turn and nothing else.

    Asserted on the built service, not on the source text: this test used to grep _flux for
    the keyword, and a mutation run showed it stayed green when the docstring above still
    said should_interrupt=False and the argument no longer did."""
    assert _plan().service._should_interrupt is False


def test_hinglish_is_described_to_it_as_both_languages():
    """The language people speak on these calls is neither Hindi nor English on its own, and
    flux-general-multi is the model that can be told so."""
    from pipecat.transcriptions.language import Language

    plan = _plan(STT_LANGUAGE="hi,en-IN")
    assert plan.service._settings.language_hints == [Language.HI, Language.EN_IN]


def test_a_single_language_still_works_unchanged():
    from pipecat.transcriptions.language import Language

    assert _plan(STT_LANGUAGE="hi").service._settings.language_hints == [Language.HI]


def test_a_language_flux_does_not_know_is_dropped_with_a_warning():
    """Flux ignores hints it cannot read, so a typo would change nothing and say nothing."""
    seen = []
    handle = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        plan = _plan(STT_LANGUAGE="hi,klingon")
    finally:
        logger.remove(handle)
    assert len(plan.service._settings.language_hints) == 1
    assert any("klingon" in line for line in seen)


def test_a_messily_written_setting_is_read_the_way_it_was_meant():
    """STT_LANGUAGE is typed by hand into an env file, and " hi , en-IN , " is what a hand
    types. The spaces and the trailing comma have to be read as two languages and nothing
    else — an empty entry survives as far as Language(""), which is dropped, but only after
    logging a warning about a language nobody wrote."""
    from pipecat.transcriptions.language import Language

    seen = []
    handle = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        plan = _plan(STT_LANGUAGE=" hi , en-IN , ")
    finally:
        logger.remove(handle)
    assert plan.service._settings.language_hints == [Language.HI, Language.EN_IN]
    assert not seen, seen


def test_the_thresholds_are_left_to_the_service_unless_somebody_sets_them():
    """Checked on the query string the service actually connects with, not on the settings
    object: the SDK normalises its own sentinel to None internally, and it is the URL that
    decides whether Deepgram hears an override or its default."""
    query = _plan().service._build_query_string()
    assert "eot_threshold" not in query
    assert "eot_timeout_ms" not in query


def test_the_thresholds_are_sent_when_somebody_does():
    query = _plan(STT_EOT_THRESHOLD=0.55, STT_EOT_TIMEOUT_MS=2500).service._build_query_string()
    assert "eot_threshold=0.55" in query
    assert "eot_timeout_ms=2500" in query


def test_the_hinglish_hints_reach_the_connection():
    """language_hints are only honoured on flux-general-multi, and the service drops them
    with a warning on any other model — so the hint and the model have to agree here."""
    query = _plan(STT_LANGUAGE="hi,en-IN").service._build_query_string()
    assert query.count("language_hint") == 2


def test_the_plan_prints_both_halves_for_the_log():
    """One line in the call log saying what we listen with AND who ends the turn — the two
    things that have to be known to read any latency number that follows."""
    assert str(_plan()) == "flux/flux-general-multi (turns: ExternalUserTurnStopStrategy)"
