"""Contract tests against the installed Pipecat.

Pipecat accepts an unknown event name with only a WARNING, so a renamed or misspelled
handler is silently dead at runtime — that is how `on_client_connected` and
`on_assistant_message_added` stayed wired to nothing while every AGENT line quietly
vanished from the call logs. These tests turn that warning into a failing build.
"""

import ast
from pathlib import Path

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.tts import SarvamTTSService

from app.services.agent import GROQ_MODEL

AGENT_SOURCE = Path(__file__).resolve().parents[1] / "app" / "services" / "agent.py"

# Objects cheap enough to build without a websocket. Handlers on `transport` and `task`
# need a live transport, so they are reported as uncovered rather than silently passed.
UNCOVERABLE = {"transport", "task"}


def _declared_event_handlers() -> set[tuple[str, str]]:
    """Extract every @<obj>.event_handler("<name>") pair from agent.py."""
    tree = ast.parse(AGENT_SOURCE.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for deco in node.decorator_list:
            if (
                isinstance(deco, ast.Call)
                and isinstance(deco.func, ast.Attribute)
                and deco.func.attr == "event_handler"
                and isinstance(deco.func.value, ast.Name)
                and deco.args
                and isinstance(deco.args[0], ast.Constant)
            ):
                found.add((deco.func.value.id, deco.args[0].value))
    return found


@pytest.fixture
def pipecat_objects():
    async def end_call(params: dict):
        """Ends the call."""

    context = LLMContext(messages=[{"role": "system", "content": "x"}], tools=[end_call])
    return {
        "llm": GroqLLMService(api_key="test", settings=GroqLLMService.Settings(model=GROQ_MODEL)),
        "user_agg": LLMUserAggregator(context=context, params=LLMUserAggregatorParams()),
        "assistant_agg": LLMAssistantAggregator(context=context),
        "tts": SarvamTTSService(api_key="test", settings=SarvamTTSService.Settings(model="bulbul:v3")),
    }


def test_agent_declares_event_handlers():
    """Guards the extractor itself: if it silently matched nothing, the suite is a no-op."""
    assert _declared_event_handlers(), "no event_handler decorators found in agent.py"


def test_every_declared_event_name_exists_in_pipecat(pipecat_objects):
    declared = _declared_event_handlers()
    checked, unknown = 0, []

    for obj_name, event_name in sorted(declared):
        if obj_name in UNCOVERABLE:
            continue
        obj = pipecat_objects.get(obj_name)
        assert obj is not None, f"agent.py registers on unknown object '{obj_name}'"
        checked += 1
        if event_name not in obj._event_handlers:
            unknown.append(f"{obj_name}.{event_name}")

    assert not unknown, (
        f"agent.py registers event handlers Pipecat does not emit: {unknown}. "
        "Pipecat only logs a warning for these, so they would never fire at runtime."
    )
    assert checked, "no coverable event handlers were checked"


@pytest.mark.parametrize(
    "obj_name,event_name",
    [
        ("llm", "on_client_connected"),
        ("assistant_agg", "on_assistant_message_added"),
    ],
)
def test_known_bad_event_names_are_rejected(pipecat_objects, obj_name, event_name):
    """The names the stale build was using. If Pipecat ever adds them the guard is moot."""
    assert event_name not in pipecat_objects[obj_name]._event_handlers


def test_end_call_tool_schema_is_valid_for_groq():
    """A required-but-unfillable parameter makes Groq reject the tool call server-side."""
    from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter

    async def end_call(params: dict):
        """Ends the call. Call this only when the prospect says goodbye."""

    context = LLMContext(messages=[{"role": "system", "content": "x"}], tools=[end_call])
    tools = OpenAILLMAdapter().get_llm_invocation_params(
        context, convert_developer_to_user=True
    )["tools"]

    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "end_call"
    assert fn["description"].strip()
    # `params` is Pipecat's special first arg and must not surface to the model.
    assert fn["parameters"]["properties"] == {}
    assert fn["parameters"]["required"] == []
