"""Whether the voice keeps its character across a reply.

Reported from live calls: the voice is good, but "kuchh places me baat karte time
consistency nahi dikhti, pace, tone change ho jata hai jisse robotic sound karta hai".

The mechanism is in Pipecat's own Sarvam service. bulbul:v3 takes a `temperature`, which
its docstring describes as "Controls output randomness... Lower values = more
deterministic, higher = more random. Defaults to 0.6." We never sent one, so every call
has run at that default — and a reply is synthesised one sentence at a time, so 0.6 is
re-rolled at every full stop. Within a single answer the pace and the tone move, which is
heard as the voice slipping out of character.

None of this can be settled by a test. Whether 0.3 sounds steady or merely flat is a
judgement made by listening to a call. What is testable is that the value is ours to
choose, reaches the engine, and does not change under anyone who has not chosen it.
"""

import ast
import inspect

import pytest

from app.core.config import Settings


def test_the_dial_exists_and_is_ours():
    assert "SARVAM_TEMPERATURE" in Settings.model_fields


def test_unset_by_default_so_no_deployment_changes_a_voice_on_its_own():
    """Pipecat's docstring says Sarvam applies 0.6 when none is sent, and sending 0.6 would
    therefore change nothing. That is documentation of someone else's server, and this
    repository has already had one call's TTS killed outright by a value Sarvam did not
    like — min_buffer_size=25, rejected at connect with "Input parameters has to be a valid
    dictionary". Sending nothing cannot regress a voice."""
    assert Settings.model_fields["SARVAM_TEMPERATURE"].default is None


def _settings(**overrides):
    return Settings(
        SECRET_KEY="x",
        DATABASE_URL="postgresql+asyncpg://u:p@h/db",
        REDIS_URL="redis://localhost",
        OPENAI_API_KEY="x",
        SARVAM_API_KEY="x",
        **overrides,
    )


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.2, 2.0])
def test_a_value_sarvam_would_reject_is_refused_at_startup(bad):
    """Pipecat forwards this straight into the connect payload with no range check. An
    out-of-range value was already proved to kill TTS for a whole call — min_buffer_size=25
    came back as "Input parameters has to be a valid dictionary" and the caller heard
    silence for the rest of it. Better to fail the boot than the conversation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(SARVAM_TEMPERATURE=bad)


@pytest.mark.parametrize("good", [0.01, 0.3, 0.6, 1.0])
def test_the_whole_range_sarvam_documents_is_allowed(good):
    """0.01 to 1.0, from the model's own parameter table. Narrowing it here would rule out
    an experiment before anybody had listened to it."""
    assert _settings(SARVAM_TEMPERATURE=good).SARVAM_TEMPERATURE == good


def test_the_value_a_person_types_into_an_env_file_arrives_as_a_number():
    """It reaches the process as the string "0.3"."""
    assert _settings(SARVAM_TEMPERATURE="0.3").SARVAM_TEMPERATURE == 0.3


def _tts_settings(temperature):
    """The settings object the agent builds, resolved the way Pipecat resolves it."""
    from pipecat.services.sarvam.tts import SarvamTTSService

    tuning = {} if temperature is None else {"temperature": temperature}
    return SarvamTTSService(
        api_key="x",
        settings=SarvamTTSService.Settings(
            model="bulbul:v3", voice="simran", pace=1.0, max_chunk_length=150, **tuning
        ),
    )._settings


def test_unset_leaves_the_key_out_of_the_payload_entirely():
    """The load-bearing one. Pipecat only writes `temperature` into the connect config when
    it resolves to something other than None, so an unset setting has to survive its
    defaulting as None — otherwise "sends nothing" would quietly become "sends Pipecat's
    idea of the default"."""
    assert _tts_settings(None).temperature is None


def test_a_configured_value_does_reach_the_payload():
    assert _tts_settings(0.3).temperature == 0.3


def test_the_setting_actually_reaches_the_voice_engine():
    """A dial wired to nothing is worse than no dial: it invites somebody to change it,
    hear no difference, and conclude the voice cannot be steadied."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "SARVAM_TEMPERATURE" in ast.unparse(n.test)
    ]
    assert guards, "the setting is never consulted"

    # The polarity, not just the presence. Inverted, this sends a temperature on exactly the
    # calls nobody configured one for and sends none on the calls that did — the dial would
    # read as broken while quietly changing every other voice.
    assert ast.unparse(guards[0].test) == "settings.SARVAM_TEMPERATURE is not None"

    body = "".join(ast.unparse(n) for n in guards[0].body)
    assert "settings.SARVAM_TEMPERATURE" in body
    assert '"temperature"' in body or "'temperature'" in body
    assert not guards[0].orelse, "an else branch would send something when it is unset"

    spread = [
        ast.unparse(k.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("SarvamTTSService.Settings")
        for k in n.keywords
        if k.arg is None
    ]
    assert spread == ["tts_tuning"], f"the tuning never reaches the service: {spread}"


def test_the_model_we_send_it_to_is_one_that_accepts_it():
    """bulbul:v2 ignores temperature and takes pitch and loudness instead. Sending it there
    would be a setting that silently does nothing."""
    from pipecat.services.sarvam.tts import TTS_MODEL_CONFIGS

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    model = next(m for m in TTS_MODEL_CONFIGS if f'model="{m}"' in src)
    assert TTS_MODEL_CONFIGS[model].supports_temperature, f"{model} ignores temperature"


def test_pace_is_still_pinned():
    """Temperature moves the pace on its own. Leaving the base pace unset as well would
    make a change here impossible to attribute."""
    from app.services import agent

    assert "pace=1.0" in inspect.getsource(agent.run_voice_agent)
