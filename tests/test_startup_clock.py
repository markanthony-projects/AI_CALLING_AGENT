"""Turning the 1.7 seconds before the greeting into a number instead of a story.

Five calls, the same reading every time:

    FIRST WORD 2257ms after the stream opened | pipeline=2079ms synthesis=343ms

343ms of that is the greeting reaching the voice engine and coming back. The other 1736ms
is before it was queued at all — and the greeting is a local f-string. There is a plausible
explanation (the services open websockets on StartFrame, and on_pipeline_started fires only
once StartFrame has been through all of them) and no measurement of it, which is exactly
where turn_decision was before it turned out to be 47% of every turn.
"""

import ast
import inspect
import time
from pathlib import Path

from app.utils.startup_clock import StartupClock

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")


def _lines(clock) -> list:
    from loguru import logger

    seen = []
    handle = logger.add(lambda m: seen.append(str(m)), level="INFO")
    try:
        clock.report()
    finally:
        logger.remove(handle)
    return seen


def test_each_milestone_is_reported_as_its_own_wait():
    """The question is never "when did TTS connect". It is "what was the greeting waiting
    for", and that is the gap between marks, not the clock reading at each one."""
    clock = StartupClock("sid", stream_open_at=time.monotonic())
    clock._marks = [("tts", 100.0), ("stt", 250.0), ("pipeline", 1700.0), ("greeting", 1740.0)]
    line = _lines(clock)[0]
    assert "tts=+100ms" in line
    assert "stt=+150ms" in line
    assert "pipeline=+1450ms" in line
    assert "greeting=+40ms" in line


def test_the_headline_is_the_total():
    clock = StartupClock("sid", stream_open_at=time.monotonic())
    clock._marks = [("tts", 100.0), ("greeting", 1740.0)]
    assert "STARTUP 1740ms to the greeting" in _lines(clock)[0]


def test_it_reports_once():
    """A reconnect later in the call is a real event and not part of how long the greeting
    took. Reporting twice would make the number mean two different things."""
    clock = StartupClock("sid", stream_open_at=time.monotonic())
    clock.mark("tts")
    assert len(_lines(clock)) == 1
    assert _lines(clock) == []


def test_a_mark_after_the_report_is_ignored():
    clock = StartupClock("sid", stream_open_at=time.monotonic())
    clock.mark("tts")
    clock.report()
    clock.mark("tts")
    assert len(clock._marks) == 1


def test_without_a_stream_clock_it_says_nothing_rather_than_guessing():
    """stream_open_at is threaded from the websocket route and can be absent. A startup line
    measured from some other zero would be worse than no line."""
    clock = StartupClock("sid", stream_open_at=None)
    clock.mark("tts")
    assert clock._marks == []
    assert _lines(clock) == []


def test_nothing_marked_means_nothing_logged():
    clock = StartupClock("sid", stream_open_at=time.monotonic())
    assert _lines(clock) == []


# --- wired to the four things worth telling apart ---------------------------------------


def test_the_agent_marks_both_connections_the_pipeline_and_the_queue():
    """Four marks, and each one can be the answer. If TTS is slow the greeting genuinely had
    to wait. If STT is slow it did not, and that is a fixable second of every call."""
    for name in ("tts", "stt", "pipeline", "greeting queued"):
        assert f'startup.mark("{name}")' in AGENT_SRC, name


def test_the_line_is_emitted_where_the_greeting_is_queued():
    """Not at the end of the call: the number is about the opening, and a caller who hangs
    up during the greeting is exactly the caller whose startup time mattered most."""
    tree = ast.parse(AGENT_SRC)
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "startup_greeting"
    )
    src = ast.unparse(handler)
    assert "startup.report()" in src
    assert src.index("queue_frames") < src.index("startup.report()")


def test_it_shares_the_clock_the_first_word_line_is_measured_against():
    """Two startup numbers measured from different zeros would be unreadable together, and
    together is the only way they are read."""
    tree = ast.parse(AGENT_SRC)
    built = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in
        ("StartupClock", "LatencyObserver")
    ]
    assert len(built) == 2
    for call in built:
        assert any(
            kw.arg == "stream_open_at" and ast.unparse(kw.value) == "stream_open_at"
            for kw in call.keywords
        ), ast.unparse(call)


def test_it_changes_nothing_about_the_call():
    """Measurement only. A startup instrument that could delay the startup it measures would
    be worse than no instrument."""
    src = inspect.getsource(StartupClock)
    for forbidden in ("await", "async ", "sleep", "queue_frames", "push_frame"):
        assert forbidden not in src, forbidden


def test_the_setup_before_the_connections_is_measured_too():
    """Call f556caf9 read stt=+1013ms, and the clock was built after the services were, so
    that number was the websocket handshake AND everything before it — the call row, the
    project read, the services themselves — with no way to tell which. A number that could
    mean two very different fixes is not yet an answer."""
    assert 'startup.mark("services built")' in AGENT_SRC
    tree = ast.parse(AGENT_SRC)
    built = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "StartupClock"
    )
    services = AGENT_SRC.index('startup.mark("services built")')
    assert built.lineno < AGENT_SRC[:services].count("\n") + 1, "the clock must exist first"
    # and it must be built before the services it is timing
    assert built.lineno < AGENT_SRC[: AGENT_SRC.index("llm = build_llm_service")].count("\n") + 1
