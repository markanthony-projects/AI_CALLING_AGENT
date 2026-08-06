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


def test_every_way_out_of_the_call_goes_through_the_tool():
    """On call 5664ace6 the agent delivered its closing line and then just sat there. The
    prospect waited eighteen seconds and hung up on us.

    The prompt was the cause and it was self-contradictory: CALL FLOW step 5 said to close
    warmly with a spoken sentence, while the TOOL section said the closing_line IS the
    goodbye and never to say one in a normal reply. The model obeyed step 5, spoke the
    goodbye, and had no reason left to call anything.

    So every branch that finishes a call must name end_call. A close described only as
    "close warmly" or "end the call" is the same trap again.
    """
    endings = [line for line in PROMPT.splitlines() if "clos" in line.lower()]
    assert endings, "the prompt no longer describes how to close a call"
    for line in endings:
        # SITE VISIT AND THE CLOSE covers the run-up to a booking, not the hangup itself.
        if "end_call" in line or "closing_line" in line:
            continue
        assert "close the call" not in line.lower() and "close warmly" not in line.lower(), (
            f"this line tells the model to finish the call without the tool that hangs "
            f"up: {line.strip()!r}"
        )


def test_the_prompt_says_what_happens_when_a_goodbye_is_merely_spoken():
    """Naming end_call in each branch is not enough on its own — the model needs the reason,
    because a spoken goodbye looks complete from where it sits."""
    assert "ONLY WAY A CALL EVER ENDS" in PROMPT
    assert "does not hang up" in PROMPT


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


# --- tone, and why it regressed -----------------------------------------------------
#
# Compressing this prompt from 3,585 to 2,641 tokens dropped five delivery instructions
# with it, and the next call came back robotic. Measured against the call before it:
# the prospect's name went from 5 replies out of 5 to 0 out of 6, and the short sentences
# stopped being warm acknowledgements ("That's great, Rahul.") and became verbless facts
# ("Near Dommasandra Circle.", "Starting price 1.17 Crores."). Sarvam synthesises each
# sentence separately, so a bare fragment gets a flat contour and the delivery goes
# mechanical. These are asserted individually because they were lost individually.


@pytest.mark.parametrize(
    "phrase,lost",
    [
        ("Tone: warm, professional, confident", "the only line that names a tone at all"),
        ("Use confident short pauses with commas", "pacing"),
        ("Do NOT read a script", "the anti-robot instruction"),
        ("real, dynamic conversation", "its other half"),
        ("Use their first name in most replies", "the name vanished from every reply"),
        ('"That works well, Chandan."', "a worked example; the bare rule was ignored"),
        ("Speak in complete sentences", "verbless fragments read flat through TTS"),
        ("NEVER jump straight to the next question", "the imperative behind ACKNOWLEDGE"),
    ],
)
def test_delivery_instructions_survive(phrase, lost):
    assert phrase in PROMPT, f"lost the rule guarding: {lost}"


def test_the_word_limits_are_not_read_as_targets():
    """The model hit 15 words by dropping verbs. The cap has to say it is a ceiling."""
    assert "ceilings, not targets" in PROMPT
    assert "never drop a verb" in PROMPT


@pytest.mark.parametrize("judgement", ["tight budget", "small budget", "good budget"])
def test_the_prompt_forbids_labelling_a_budget(judgement):
    """Told "Below 60 lakhs", the agent replied "That is a tight budget." — on the path
    whose entire purpose is capturing a requirement for a colleague to call back about."""
    assert "NEVER JUDGE THE PROSPECT" in PROMPT
    assert judgement in PROMPT, "the exact phrasing to avoid is not named"


def test_a_rejection_gets_an_acknowledgement_too():
    """"No, I'm not interested" was answered with a bare question and nothing else."""
    assert "they say no or are not interested" in PROMPT


def test_bhk_spacing_says_why():
    """The rule was in the prompt and ignored on two consecutive calls; a bare
    instruction with no reason is the first thing a compressing model drops."""
    assert "3 B H K" in PROMPT
    assert "pronounce it as a word" in PROMPT
