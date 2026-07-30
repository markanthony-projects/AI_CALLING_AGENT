"""Prompt rules added in response to specific failures on live calls.

These assert the instruction is present, not that the model obeys it — only a real call
can show that. They exist so a prompt edit cannot silently drop a hard-won rule.
"""

import inspect

import pytest

from app.prompts.agent_prompts import get_system_prompt
from app.services import agent

PROMPT = get_system_prompt("Project Name: Test\nLocation: Test")


@pytest.mark.parametrize(
    "phrase,failure",
    [
        ("NOT THE END OF THE CALL", "hung up 457ms after the prospect agreed to a site visit"),
        ("not scheduled", "an agreed-but-unscheduled visit is a lost booking"),
        ("specific DAY", "no day was ever pinned down"),
        ("NEVER call end_call in the same turn", "the tool fired on agreement, not on goodbye"),
    ],
)
def test_closing_rules_survive(phrase, failure):
    assert phrase in PROMPT, f"lost the rule guarding: {failure}"


@pytest.mark.parametrize(
    "phrase",
    ["NEVER SPEAK ABOUT IT", "never hear you reasoning about them"],
)
def test_language_policy_is_not_spoken_aloud(phrase):
    """It told a caller 'since you've spoken a few Hindi words, I can continue in Hindi'."""
    assert phrase in PROMPT


def test_reply_length_has_a_hard_cap():
    """A 24-word sentence took ~15s to speak, which is 15s of the prospect waiting."""
    assert "HARD LIMITS" in PROMPT
    assert "35 words" in PROMPT


def test_end_call_tool_description_warns_against_agreement():
    """This docstring becomes the tool schema the model reads when deciding to hang up."""
    source = inspect.getsource(agent.run_voice_agent)
    start = source.index("async def end_call(")
    doc = source[start : source.index("pass", start)]

    assert "irreversible" in doc.lower()
    for token in ('"Yes"', '"sure"', "specific day AND time"):
        assert token in doc, f"tool description no longer warns about {token}"


def test_end_call_docstring_reaches_the_model():
    """Pipecat builds the tool schema from this docstring; an empty one tells the model nothing."""
    from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
    from pipecat.processors.aggregators.llm_context import LLMContext

    async def end_call(params: dict):
        """Hangs up the phone. Do not call this when the prospect agrees to something."""

    ctx = LLMContext(messages=[{"role": "system", "content": "x"}], tools=[end_call])
    fn = OpenAILLMAdapter().get_llm_invocation_params(ctx, convert_developer_to_user=True)["tools"][0]["function"]
    assert "agrees to something" in fn["description"]


@pytest.mark.parametrize("phrase", ["Write EVERY word in English/Latin letters", "romanise"])
def test_output_script_is_constrained_to_latin(phrase):
    """The agent said "Mayur, नमस्ते Mayur" and the caller reported the voice breaking up.
    STT returns Devanagari, the model echoes it, and Sarvam gets mixed script mid-sentence."""
    assert phrase in PROMPT


def test_tts_settings_use_no_unvalidated_tuning_values():
    """Sarvam rejects out-of-range config at connect time and TTS dies for the whole call.

    min_buffer_size=25 did exactly that. Pipecat forwards these values into the config
    payload unchecked, so anything here must be confirmed against Sarvam's API first.
    """
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "Settings"
        and getattr(getattr(n.func, "value", None), "id", "") == "SarvamTTSService"
    ]
    assert calls, "Sarvam TTS settings not found"
    supplied = {kw.arg for kw in calls[0].keywords}
    assert "min_buffer_size" not in supplied, (
        "min_buffer_size=25 was rejected by Sarvam and silenced the agent; "
        "leave it at the default until a value is verified against their API"
    )
