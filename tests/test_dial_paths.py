"""Every way a number reaches Vobiz, and the gate it has to pass first.

There were two doors. The dashboard route enqueued contacts and let the pump reserve a
carrier slot before each call. The API-key route at /api/v1/campaigns/{id}/dial/vobiz created
one background task per number — up to five hundred — and asked Vobiz to dial every one at
once. The concurrency cap still applied, but at the media websocket, which opens after the
carrier has dialled, billed us and rung a real person; on a three-slot account everyone past
the third was called, charged for, and hung up on with no Call row to say it happened.

The deployment guide pointed operators at that door.

These are structural rather than behavioural on purpose. The failure they guard against is
not a wrong answer — it is a second implementation appearing, which no amount of testing the
first one would catch. What must stay true is that exactly one function talks to the carrier,
and that it reserves a slot before it does.
"""

import ast
import inspect
from pathlib import Path

import pytest

from app.api.routes import campaign as campaign_route
from app.services import dial_pump, dial_queue

APP = Path("app")

# The one function allowed to ask the carrier for a call.
DIALER = "app/services/dial_pump.py"


def _modules_mentioning(name: str) -> set:
    return {
        str(path).replace("\\", "/")
        for path in APP.rglob("*.py")
        if name in path.read_text(encoding="utf-8")
    }


def test_only_the_pump_reaches_the_carrier():
    """The gate is worth nothing if a second caller appears beside it. dialer.py defines
    trigger_vobiz_call; dial_pump.py is the only module that may call it."""
    callers = _modules_mentioning("trigger_vobiz_call") - {"app/services/dialer.py"}
    assert callers == {DIALER}, f"something else dials Vobiz: {sorted(callers - {DIALER})}"


def test_the_slot_is_taken_before_the_dial():
    """Reserving after the dial reopens the exact hole this exists to close: the carrier has
    already rung a real person by the time we decide we had no room for them."""
    tree = ast.parse(inspect.getsource(dial_pump._place).lstrip())

    def line_of(name: str) -> int:
        lines = [
            n.lineno
            for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id == name)
            or (isinstance(n, ast.Attribute) and n.attr == name)
        ]
        assert lines, f"{name} is not referenced at all"
        return min(lines)

    assert line_of("acquire") < line_of("trigger_vobiz_call")


def test_a_refused_slot_places_no_call():
    """The branch that runs when the carrier is full must return before dialling, not fall
    through to it."""
    src = inspect.getsource(dial_pump._place)
    refused = src.index("if not await call_slots.acquire(call_sid):")
    dialled = src.index("trigger_vobiz_call")
    returned = src.index("return False", refused)
    assert returned < dialled, "a refused slot falls through to the dial"


@pytest.mark.parametrize(
    "route", ["app/api/routes/campaign.py", "app/api/routes/dashboard.py"]
)
def test_no_route_dials_by_itself(route):
    """Both doors now lead to the same queue. A route that dials directly would be back to
    checking the cap after the money is spent."""
    src = Path(route).read_text(encoding="utf-8")
    assert "trigger_vobiz_call" not in src
    assert "BackgroundTasks" not in src, f"{route} can still dial out of band"


def test_the_api_key_route_goes_through_the_queue():
    """It is the one the deployment guide tells an operator to curl, so it is the one most
    likely to be used by somebody who has not read this file."""
    src = inspect.getsource(campaign_route.dial_campaign_vobiz)
    assert "dial_queue.enqueue" in src


def test_the_queue_itself_never_dials():
    """Enqueueing is allowed to be unbounded — five hundred rows cost nothing until the pump
    takes a slot for each one. That is only true while this places no calls."""
    assert "trigger_vobiz_call" not in inspect.getsource(dial_queue)


def test_both_routes_return_the_same_shape():
    """They used to disagree: one returned call_sids for calls it had already placed, the
    other a queue report. Identifiers for calls that do not exist are what let the old
    response read as success."""
    fields = set(dial_queue.QueueReport().as_response())
    assert "will_dial" in fields
    assert "call_sids" not in fields, "no call has been placed, so there is nothing to name"
