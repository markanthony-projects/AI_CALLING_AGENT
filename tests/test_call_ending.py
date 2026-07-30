"""How the agent hangs up.

Ending the call used to queue EndFrame alone, so the line simply dropped on the prospect
with no goodbye. And a provider-rejected tool call was treated as a lost turn, making the
agent ask someone who had just said "no thank you" to repeat themselves.
"""

import ast
import inspect

import pytest

from app.services import agent


def _queued_frames(func) -> list[list[str]]:
    """Every task.queue_frames([...]) call in func, as lists of frame class names."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    out = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "queue_frames"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            out.append(
                [
                    getattr(el.func, "id", None)
                    for el in node.args[0].elts
                    if isinstance(el, ast.Call)
                ]
            )
    return out


def _nested(name):
    """Pull a closure-defined handler out of run_voice_agent's source."""
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in run_voice_agent")


def _frames_in(node) -> list[list[str]]:
    out = []
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "queue_frames"
            and n.args
            and isinstance(n.args[0], ast.List)
        ):
            out.append(
                [getattr(el.func, "id", None) for el in n.args[0].elts if isinstance(el, ast.Call)]
            )
    return out


def test_farewell_line_is_real_speech():
    assert agent.FAREWELL_LINE.strip()
    assert len(agent.FAREWELL_LINE.split()) >= 4


def test_agent_speaks_before_hanging_up():
    """end_call must play a closing line, not drop the caller into silence."""
    batches = _frames_in(_nested("end_call_handler"))
    assert batches, "end_call_handler queues no frames"
    frames = batches[0]
    assert "TTSSpeakFrame" in frames, "no farewell is spoken before the call ends"
    assert "EndFrame" in frames
    assert frames.index("TTSSpeakFrame") < frames.index("EndFrame"), (
        "EndFrame must be queued behind the farewell or the line is cut off"
    )


def test_rejected_tool_call_ends_the_call_instead_of_asking_to_repeat():
    handler = _nested("on_llm_error")
    src = ast.unparse(handler)
    assert "FUNCTION_CALL_FAILURE" in src, (
        "a provider-rejected tool call must be distinguished from a lost turn"
    )
    # The tool-failure branch speaks the farewell and ends; it must not fall through
    # into the generic recovery path.
    assert "FAREWELL_LINE" in src


def test_function_call_failure_marker_matches_the_provider_string():
    """Taken verbatim from the Groq error seen in production."""
    groq_error = (
        "Error during completion: Failed to call a function. "
        "Please adjust your prompt. See 'failed_generation' for more details."
    )
    assert agent.FUNCTION_CALL_FAILURE in groq_error.lower()


def test_recovery_and_farewell_lines_are_distinct():
    """Asking a departing prospect to repeat themselves is the bug being fixed."""
    assert agent.FAREWELL_LINE != agent.LLM_RECOVERY_LINE


@pytest.mark.parametrize(
    "rule",
    [
        "CRITICAL RULE FOR ATTRIBUTION",
        "Preserve the transcript's structure",
    ],
)
def test_extraction_prompt_guards_known_defects(rule):
    """Real defects from a live call: 'Whitefield' copied out of the agent's own pitch,
    and the transliteration flattening every turn into one paragraph."""
    from datetime import datetime

    from app.worker import _build_system_prompt

    assert rule in _build_system_prompt(datetime(2026, 7, 27, 12, 0, 0))
