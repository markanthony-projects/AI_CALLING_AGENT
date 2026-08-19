"""How many calls may be in flight, and where that is decided.

The cap used to be checked in one place only: when the media websocket opened. That is after
Vobiz has dialled, billed us, and rung a real person's phone. They answered and got the line
closed on them — and because the check returned before the Call row was written, there was no
record it had happened. Meanwhile the dial endpoint placed every number in the request at
once, so on a three-slot account a list of twenty meant seventeen people were called, charged
for, and hung up on invisibly.

It was also a module-level integer, which is not a count of anything: two api containers
would each admit three calls, and a restart reset it to zero while the carrier still had
calls up.

So the gate moved in front of the dial, and the count moved into Redis.
"""

import ast
import inspect

from app.api.routes import webhook
from app.core import call_slots
from app.core.config import Settings, settings
from app.services import dial_pump


# --- where the gate sits --------------------------------------------------------------


def test_a_slot_is_taken_before_the_carrier_is_asked_to_dial():
    """The whole point. Money is spent by trigger_vobiz_call; the reservation has to be
    settled before it, not after the person has already picked up."""
    src = inspect.getsource(dial_pump._place)
    assert "call_slots.acquire" in src
    assert src.index("call_slots.acquire") < src.index("trigger_vobiz_call")


def test_a_refused_reservation_places_no_call():
    """Returning False must abandon the dial, not merely log and continue."""
    tree = ast.parse(inspect.getsource(dial_pump._place).lstrip())
    guard = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "call_slots.acquire" in ast.unparse(n.test)
    )
    assert "return False" in ast.unparse(guard)


def test_an_abandoned_dial_is_not_charged_an_attempt():
    """The carrier filling up between the count and the dial is not the contact's fault. If
    the attempt stood, three such races would exhaust a number nobody ever rang."""
    guard = next(
        n for n in ast.walk(ast.parse(inspect.getsource(dial_pump._place).lstrip()))
        if isinstance(n, ast.If) and "call_slots.acquire" in ast.unparse(n.test)
    )
    body = ast.unparse(guard)
    assert "ContactStatus.PENDING" in body, "the contact is not returned to the queue"
    assert "attempts" in body, "the attempt increment is not undone"


def test_the_stream_still_refuses_over_capacity():
    """Kept as a last resort for the paths that do not come through the pump — a manual dial,
    and the browser client — and it must close before accepting, or a pipeline is spun up for
    a call being refused."""
    src = inspect.getsource(webhook._handle_call)
    assert "call_slots.acquire" in src
    assert src.index("call_slots.acquire") < src.index("websocket.accept()")
    tree = ast.parse(src.lstrip())
    guard = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "call_slots.acquire" in ast.unparse(n.test)
    )
    body = ast.unparse(guard)
    assert "close" in body and "return" in body


def test_a_refused_call_leaves_a_record():
    """It was billed and nobody could serve it. Returning before the Call row was written is
    why nobody knew this was happening."""
    guard = next(
        n for n in ast.walk(ast.parse(inspect.getsource(webhook._handle_call).lstrip()))
        if isinstance(n, ast.If) and "call_slots.acquire" in ast.unparse(n.test)
    )
    assert "_record_refused" in ast.unparse(guard)
    assert "Call(" in inspect.getsource(webhook._record_refused)


def test_the_slot_is_released_when_the_call_ends():
    """A slot that is not given back holds a third of the account's capacity until it ages
    out, so this cannot sit anywhere a pipeline exception can skip it."""
    tree = ast.parse(inspect.getsource(webhook._handle_call).lstrip())
    finallys = [
        ast.unparse(node.finalbody)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try) and node.finalbody
    ]
    assert any("call_slots.release" in f for f in finallys)


# --- how it is counted ----------------------------------------------------------------


def test_the_count_is_shared_and_not_per_process():
    """Two api containers each counting to three is six calls on a three-call account."""
    src = inspect.getsource(call_slots)
    assert "get_redis_client" in src
    assert "ZADD" in src and "ZCARD" in src


def test_the_check_and_the_insert_cannot_interleave():
    """This was five separate awaits with a comment claiming they were one. They were not,
    and the gap between reading the count and adding to it is a real race: two processes both
    see two-of-three and both add, putting four calls on a three-call account.

    A Lua script runs inside Redis with nothing able to interleave, so the count a caller acts
    on is the count at the moment it takes the slot.
    """
    assert "ZCARD" in call_slots._ACQUIRE and "ZADD" in call_slots._ACQUIRE
    assert "client.eval" in inspect.getsource(call_slots.acquire)


def test_taking_a_slot_is_one_round_trip():
    """It is on the path that opens a live call, and it was eleven round trips before the
    greeting played — five of them here."""
    src = inspect.getsource(call_slots.acquire)
    assert src.count("await client.") == 1


def test_reading_the_count_never_takes_a_slot():
    """Separate scripts on purpose. One that both counted and inserted would hand a slot to
    the dashboard every time somebody looked at it."""
    assert "ZADD" not in call_slots._ACTIVE


def test_a_leaked_slot_expires_on_its_own():
    """A process that dies mid-call cannot run its own cleanup. Without expiry the slot is
    gone for good, and three of those stop the system dialling for ever with nothing in the
    logs to say why.

    Checked on the script acquire runs, not on the module: active() sweeps too, and a
    module-wide grep passes while the sweep is missing from the one place that decides
    whether to dial.
    """
    assert "ZREMRANGEBYSCORE" in call_slots._ACQUIRE
    assert "MAX_CALL_DURATION_SECS" in inspect.getsource(call_slots._stale_after)


def test_expiry_waits_out_the_longest_possible_call():
    """Reclaiming a slot from a call still in progress would let a fourth be dialled."""
    from app.services.agent import MAX_CALL_DURATION_SECS

    assert call_slots._stale_after() > MAX_CALL_DURATION_SECS


def test_the_clock_comes_from_redis():
    """Several processes write these scores. Each using its own clock would let a few seconds
    of container drift expire live slots or keep dead ones."""
    for script in (call_slots._ACQUIRE, call_slots._ACTIVE):
        assert "redis.call('TIME')" in script
    # And no Python clock leaks in as an argument to either.
    assert "time.time()" not in inspect.getsource(call_slots)


def test_redis_being_down_means_no_capacity_rather_than_unlimited():
    """A stalled campaign is recoverable. A flood of billed calls nobody is metering is not."""
    acquire = inspect.getsource(call_slots.acquire)
    handler = acquire[acquire.index("except Exception"):]
    assert "return False" in handler
    active = inspect.getsource(call_slots.active)
    assert "MAX_CONCURRENT_CALLS" in active[active.index("except Exception"):]


def test_acquiring_twice_for_one_call_uses_one_slot():
    """The pump takes the slot, and the answer webhook refreshes it. A retried webhook must
    not consume a second."""
    assert "ZSCORE" in call_slots._ACQUIRE, "an existing holder is not recognised"
    assert "== false" in call_slots._ACQUIRE


# --- the value ------------------------------------------------------------------------


def test_the_shipped_default_matches_the_carrier():
    """Asserted on the field default, not the loaded settings, so a local .env override does
    not fail the suite — the shipped number is what a fresh deployment gets.

    The Vobiz account allows three. It sat at 4, so the application's own cap was looser than
    the carrier's and the fourth call could only ever fail.
    """
    assert Settings.model_fields["MAX_CONCURRENT_CALLS"].default == 3


def test_the_configured_value_is_not_above_the_carrier_capacity():
    """A local override is fine, above the account's limit is not."""
    assert settings.MAX_CONCURRENT_CALLS <= 3


def test_cap_must_be_positive():
    """Zero would reject every call while looking like a valid configuration."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(
            MAX_CONCURRENT_CALLS=0,
            API_KEY="k" * 32,
            CALL_TOKEN_SECRET="s" * 32,
            DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
            OPENAI_API_KEY="x",
            SARVAM_API_KEY="x",
        )
