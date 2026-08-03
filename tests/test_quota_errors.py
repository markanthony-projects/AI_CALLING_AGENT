"""A provider refusing on budget is not a failed turn.

From a live call, Groq's free tier ran out of daily tokens mid-conversation:

    [429] Rate limit reached ... tokens per day (TPD): Limit 100000, Used 98928
          Please try again in 28m37.631999999s

The generic recovery path then said "Sorry, I missed that. Could you say it once more?",
the caller repeated a sentence we had heard perfectly, the next turn failed identically,
and the call was dropped anyway. Fifteen seconds spent blaming the caller for our billing.
"""

import ast
import inspect

import pytest

from app.services import agent
from app.services.agent import (
    LLM_RECOVERY_LINE,
    LLM_SIGNOFF_LINE,
    is_quota_error,
    session_error,
)

# Verbatim from the production log.
GROQ_TPD = (
    "Error during completion: Error code: 429 - {'error': {'message': 'Rate limit reached "
    "for model `llama-3.3-70b-versatile` in organization `org_01k` service tier `on_demand` "
    "on tokens per day (TPD): Limit 100000, Used 98928, Requested 3060. Please try again in "
    "28m37.631999999s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def test_the_real_groq_message_is_recognised():
    assert is_quota_error(GROQ_TPD)


@pytest.mark.parametrize("marker", agent.LLM_QUOTA_MARKERS)
def test_every_marker_earns_its_place(marker):
    """The real Groq message trips several markers at once, so removing any one of them
    still matched it — meaning the list could rot into untested config. Each marker has to
    match on its own or it should not be in the list."""
    assert is_quota_error(f"Error during completion: {marker} (detail)")


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - rate_limit_exceeded",
        "You exceeded your current quota, please check your plan and billing details.",
        "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
        "RATE LIMIT REACHED for model x",  # providers are inconsistent about case
    ],
)
def test_other_provider_phrasings_are_recognised(message):
    assert is_quota_error(message)


@pytest.mark.parametrize(
    "message",
    [
        None,
        "",
        "Error during completion: Connection reset by peer",
        "Failed to call a function. Please adjust your prompt.",
        "Error code: 500 - internal server error",
        "context_length_exceeded",
    ],
)
def test_ordinary_failures_are_not_treated_as_quota(message):
    """These are worth one retry; misreading them as quota would end calls that could recover."""
    assert not is_quota_error(message)


def test_quota_gets_its_own_call_status_reason():
    """Reads differently in the logs on purpose: nothing is wrong with the code."""
    reason = session_error(None, llm_failures=0, llm_quota_exhausted=True)
    assert reason is not None
    assert "quota" in reason
    assert reason != session_error(None, llm_failures=99)


def test_quota_outranks_a_plain_turn_failure():
    """The first 429 also increments nothing; the reported cause must still be the quota."""
    assert "quota" in session_error(None, llm_failures=99, llm_quota_exhausted=True)


def test_a_pipeline_error_still_wins():
    """A hard transport failure is the more specific cause and must not be masked."""
    assert session_error("websocket closed", 0, llm_quota_exhausted=True) == "websocket closed"


# --- the handler ------------------------------------------------------------------


def _error_handler_source() -> str:
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "on_llm_error"
    )
    return ast.unparse(node)


def test_quota_signs_off_instead_of_asking_the_caller_to_repeat():
    src = _error_handler_source()
    assert "is_quota_error" in src, "quota errors fall through to the generic retry path"

    quota_branch = src[src.index("is_quota_error"):]
    # Up to the next branch, the quota path must sign off and end.
    head = quota_branch.split("_llm_failures += 1")[0]
    assert "LLM_SIGNOFF_LINE" in head
    # EndWorkerFrame rather than EndFrame: the sign-off is spoken first, and EndFrame
    # stops the transport as soon as it is received in queue order, cutting it off.
    assert "EndWorkerFrame" in head
    assert LLM_RECOVERY_LINE not in head
    assert "LLM_RECOVERY_LINE" not in head, (
        "asking a caller to repeat cannot help when the provider is out of budget"
    )


def test_no_counted_branch_falls_through_to_the_quota_check():
    """Counting first would let two quota errors look like two lost turns.

    Checked structurally rather than by position. A throttle branch was added above this
    one that legitimately does count — a throttled turn really is a lost turn — so "no
    increment appears earlier in the text" started failing on correct code. What actually
    matters is that nothing which increments can then reach the quota branch, i.e. every
    earlier increment sits in a branch that returns.
    """
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "on_llm_error"
    )
    quota = next(
        n for n in handler.body
        if isinstance(n, ast.If) and "is_quota_error" in ast.unparse(n.test)
    )

    for node in handler.body:
        if node is quota:
            break
        if "_llm_failures += 1" not in ast.unparse(node):
            continue
        assert isinstance(node, ast.If), "an unconditional increment reaches the quota branch"
        assert any(isinstance(s, ast.Return) for s in node.body), (
            f"branch {ast.unparse(node.test)!r} increments the counter and then falls "
            "through to the quota check, so one quota error is counted as two lost turns"
        )


def test_the_quota_guard_has_no_extra_conditions():
    """Position in the file is not enough: an added conjunction leaves the branch in place
    but makes it unreachable, and the quota falls through to the retry path anyway.

    Checked as "the test is a single call, with no boolean operator" rather than by exact
    string. The guard legitimately takes a second argument now — the delay the provider sent
    in its Retry-After header, which is the only thing distinguishing a Cerebras hiccup from
    a Cerebras dead end, since its message carries no number at all.
    """
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "on_llm_error"
    )
    guard = next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "is_quota_error" in ast.unparse(n.test)
    )
    assert isinstance(guard.test, ast.Call), (
        f"guard is {ast.unparse(guard.test)!r}; a conjunction here can gate it shut"
    )
    assert getattr(guard.test.func, "id", None) == "is_quota_error"
    assert not any(isinstance(n, ast.BoolOp) for n in ast.walk(guard.test))


def test_the_two_spoken_lines_stay_distinct():
    assert LLM_SIGNOFF_LINE != LLM_RECOVERY_LINE


def test_prompt_separates_a_line_check_from_a_brush_off():
    """The caller said only "Hello?" after seven seconds of dead air and was offered a
    callback, as though they had asked to be left alone."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("ctx")
    assert "CHECKING THE LINE" in prompt
    assert "NOT a brush-off" in prompt
    assert "NEVER offer a callback for this" in prompt
