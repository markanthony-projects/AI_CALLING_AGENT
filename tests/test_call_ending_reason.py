"""Every ending has to name itself.

On call 7defec4d the agent asked a question, its audio played out, and 43ms later the pipeline
was finished — with none of the reasons this codebase logs anywhere in the record:

    09:19:22.090  AGENT → "...Did you mean you would be free after lunch tomorrow?"
    09:19:22.133  Pipeline finished. Extracting transcript...
    09:19:22.166  Call finalised | status=COMPLETED | duration=129.7s

Every deliberate ending writes a line before it queues a frame, so an ending with no line is
one nobody chose — or one whose reason was lost. Either way the record cannot tell "the
prospect hung up" from "we hung up on the prospect", which for a sales call is the whole
question.

So the pipeline reports its own terminator, and the reason rides on the frame rather than on a
log line written somewhere earlier: a frame carries its reason wherever it ends up, and a log
line written before queueing is only adjacent to the ending in time.
"""

import ast
import inspect

import pytest

from app.services import agent
from app.services.agent import CallResult, ending_reason


def _fn(name):
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in run_voice_agent")


class _Frame:
    """Stands in for a terminating frame. Named EndFrame so the fallback is recognisable."""

    def __init__(self, reason=None):
        self.reason = reason


class EndFrame(_Frame):
    pass


class CancelFrame(_Frame):
    pass


def test_a_named_ending_is_reported_by_name():
    assert ending_reason(EndFrame(reason="the caller hung up")) == "the caller hung up"


def test_an_unnamed_ending_falls_back_to_the_mechanism():
    """Not None and not "completed". An ending nobody labelled is the case worth seeing, and
    the class at least says which mechanism fired."""
    assert ending_reason(EndFrame()) == "EndFrame"
    assert ending_reason(CancelFrame()) == "CancelFrame"


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_reason_counts_as_absent(blank):
    """An empty string reads in a log as though something was recorded when nothing was."""
    assert ending_reason(EndFrame(reason=blank)) == "EndFrame"


def test_a_frame_with_no_reason_attribute_at_all_is_handled():
    """Pipecat's frame classes vary, and this runs in the path that closes every call."""

    class Bare:
        pass

    assert ending_reason(Bare()) == "Bare"


def test_surrounding_whitespace_is_not_carried_into_the_log():
    assert ending_reason(EndFrame(reason="  end_call tool  ")) == "end_call tool"


def test_the_handler_uses_it_rather_than_reimplementing_it():
    """Behaviour above is only worth pinning if the pipeline actually goes through it."""
    assert "ending_reason(frame)" in ast.unparse(_fn("on_pipeline_finished"))


def test_the_handler_is_registered_on_the_worker():
    """A handler nobody wired up logs nothing, which is the state this fixes."""
    src = inspect.getsource(agent.run_voice_agent)
    assert '@task.event_handler("on_pipeline_finished")' in src


def test_no_ending_queues_an_anonymous_frame():
    """The point of the exercise. A bare EndFrame() gives on_pipeline_finished nothing to
    report but the class name, which is what every ending would have looked like."""
    src = inspect.getsource(agent.run_voice_agent)
    assert "EndFrame()" not in src
    assert "EndWorkerFrame()" not in src


@pytest.mark.parametrize(
    "reason",
    [
        "the caller hung up",
        "end_call tool",
        "answering machine",
        "llm quota exhausted",
        "llm turn failures exhausted",
    ],
)
def test_the_endings_that_matter_are_named(reason):
    """These are the ones an operator reading a call record needs to tell apart."""
    assert reason in inspect.getsource(agent.run_voice_agent)


def test_cancelling_carries_a_reason_too():
    """abandon_call tears the pipeline down with a CancelFrame, which carries a reason the
    same way — otherwise the one ending that means something went wrong is the one that
    arrives anonymous."""
    assert "task.cancel(reason=reason)" in ast.unparse(_fn("abandon_call"))


def test_the_reason_leaves_the_pipeline_on_the_result():
    """A log line is read once, by whoever is watching. The record is read afterwards, by
    somebody asking why a campaign's calls are short."""
    assert "end_reason" in CallResult.__dataclass_fields__
    assert "end_reason=_end_reason" in inspect.getsource(agent.run_voice_agent)


def test_an_unattributed_ending_stays_visible_rather_than_defaulting():
    """None is the honest value for an ending nothing reported, and it is the case worth
    seeing. Defaulting it to "completed" would hide exactly the call that prompted this."""
    assert CallResult(transcript="x").end_reason is None
