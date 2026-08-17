"""The opening line must survive line noise.

A caller answered and heard nothing for seven seconds:

    12:17:14  stream open
    12:17:20  VAD fired with no transcribable speech after 5248ms — false barge-in
    12:17:21  USER → "Hello?"
    12:17:23  agent finally speaks

_user_has_spoken was set from on_user_turn_started, which fires on VAD rather than on a
transcript, so connection noise suppressed the greeting before it was ever spoken.
"""

import ast
import inspect

import pytest

from app.services import agent

TREE = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())


def _node(name):
    for n in ast.walk(TREE):
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in run_voice_agent")


def _assigns_spoken(node) -> bool:
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_user_has_spoken" for t in n.targets)
        for n in ast.walk(node)
    )


def test_vad_alone_does_not_cancel_the_greeting():
    """on_user_turn_started fires on VAD. Noise must not count as the prospect speaking."""
    assert not _assigns_spoken(_node("on_user_turn_started")), (
        "a false barge-in would suppress the opening line again"
    )


def test_a_real_transcript_does_cancel_the_greeting():
    """If they genuinely speak first, the canned line must not talk over them — the prompt
    tells the model to introduce itself in that case instead."""
    assert _assigns_spoken(_node("on_user_turn_stopped"))


def test_the_flag_is_set_only_on_a_non_empty_transcript():
    """The empty-transcript branch is the false barge-in; it must fall through to the warning."""
    node = _node("on_user_turn_stopped")
    guarded = [
        n for n in ast.walk(node)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "transcript"
        and _assigns_spoken(n)
    ]
    assert guarded, "_user_has_spoken is set outside the `if transcript:` branch"


def test_greeting_still_checks_the_flag():
    src = ast.unparse(_node("startup_greeting"))
    assert "_user_has_spoken" in src, "the greeting would now talk over a prospect who spoke first"


# --- how long the caller waits before hearing anything -------------------------------
#
# Measured on a live call, from the moment they picked up:
#
#     24.327  vobiz answer webhook (caller picked up)
#     24.938  websocket opened                            +611ms
#     25.027  voice agent starting                        + 89ms
#     27.455  Vobiz "start" event received                +2428ms
#     28.283  greeting queued to TTS                      + 828ms
#             ...plus ~240ms of Sarvam TTFB
#
# Four seconds of silence after picking up. Most of it is Vobiz and the pipeline coming up,
# but 200ms of it was ours: a sleep between the pipeline starting and the greeting being
# queued.


def test_the_greeting_is_not_delayed_by_a_sleep():
    """The caller already waits about four seconds. Nothing of ours may be added to it.

    The sleep was guarding against greeting somebody who spoke first; _user_has_spoken
    checks that directly, and GreetingOnlyMinWords covers the rest of the opening line.
    """
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    greeting = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "startup_greeting"
    )
    for node in ast.walk(greeting):
        if isinstance(node, ast.Call) and "sleep" in ast.unparse(node.func):
            pytest.fail(f"{ast.unparse(node)} delays the first thing the caller hears")


def test_the_greeting_is_still_skipped_when_the_prospect_speaks_first():
    """Removing the sleep must not remove the check it sat in front of — otherwise the
    agent talks over somebody who answered with "Hello?"."""
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    greeting = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == "startup_greeting"
    )
    src = ast.unparse(greeting)
    assert "if not _user_has_spoken:" in src
    assert "greeting_gate.relax()" in src, "the gate must still lift when they spoke first"
