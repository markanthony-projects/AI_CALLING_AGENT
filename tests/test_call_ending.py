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


def _body(name) -> str:
    """The function's statements without its docstring.

    These assertions compare source positions, and every one of these docstrings names the
    frames it is explaining — so matching against the whole source finds the prose, not the
    code, and the ordering claim becomes meaningless.
    """
    node = _nested(name)
    stmts = node.body
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        stmts = stmts[1:]
    return "\n".join(ast.unparse(s) for s in stmts)


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
    batches = _frames_in(_nested("say_goodbye_then_hang_up"))
    flat = [f for batch in batches for f in batch]
    assert "TTSSpeakFrame" in flat, "no farewell is spoken before the call ends"
    # EndWorkerFrame, not EndFrame. EndFrame stops the transport as soon as it is received
    # in queue order, and on a live call that cut the goodbye off 425ms after the tool
    # fired, for a sentence that takes about three seconds to speak.
    assert "EndWorkerFrame" in flat, "the farewell is cut off unless the queue flushes first"
    assert "EndFrame" not in flat
    assert flat.index("TTSSpeakFrame") < flat.index("EndWorkerFrame")


def test_the_three_steps_are_separate_and_awaited():
    """They used to be one queue_frames([Interruption, Speak, EndWorker]) call, and the
    caller heard the difference. Neither of the two worker frames takes effect on arrival:
    each makes a round trip to the sink and back first, and the worker's push loop does not
    wait for that before queueing the next frame. So the interruption meant to protect the
    farewell could land after it and cancel it instead."""
    batches = _frames_in(_nested("say_goodbye_then_hang_up"))
    assert len(batches) >= 3, "the frames must be queued in separate, awaited steps"
    assert batches[0] == ["InterruptionWorkerFrame"]
    assert "TTSSpeakFrame" in batches[1] and len(batches[1]) == 1
    assert batches[-1] == ["EndWorkerFrame"]


def test_the_interruption_is_confirmed_landed_before_the_farewell_is_spoken():
    src = _body("say_goodbye_then_hang_up")
    assert "flush_pipeline" in src
    assert src.index("flush_pipeline") < src.index("TTSSpeakFrame")


def test_it_waits_for_the_goodbye_to_be_heard_not_merely_queued():
    """EndWorkerFrame's round trip proves the FRAMES travelled. Sarvam is a websocket TTS —
    run_tts sends the text and returns, and the audio follows on a separate receive task —
    so the end signal can finish its lap while the voice is still streaming in behind it.
    BotStoppedSpeakingFrame, which FarewellGate waits on, comes off the transport's audio
    clock once the turn has actually been played out."""
    src = _body("say_goodbye_then_hang_up")
    assert "wait_until_spoken" in src
    assert src.index("wait_until_spoken") < src.index("EndWorkerFrame")


def test_the_wait_cannot_hold_the_carrier_leg_open_for_ever():
    """A dead TTS never raises BotStoppedSpeakingFrame, and every second of waiting for one
    that is not coming is billed by Vobiz."""
    src = _body("say_goodbye_then_hang_up")
    assert "farewell_timeout(line)" in src
    # And the hangup happens regardless of what the wait returned.
    assert "if not await farewell.wait_until_spoken" in src


def test_the_hangup_runs_detached_from_the_tool_call():
    """The handler runs inside the LLM service's function-call machinery, which is waiting to
    push the tool result. Holding it for the length of a spoken sentence would block the very
    pipeline that has to carry the audio."""
    src = _body("end_call_handler")
    assert "asyncio.create_task(say_goodbye_then_hang_up(line))" in src
    assert "await say_goodbye_then_hang_up" not in src


def test_only_one_hangup_per_call():
    """A model can emit a structured tool call and leaked syntax in the same turn. Two
    farewells racing would interrupt each other — the exact failure this path exists to
    prevent."""
    for name in ("end_call_handler", "on_leaked_end_call"):
        src = _body(name)
        assert "_ending" in src, f"{name} does not guard against a second hangup"


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
