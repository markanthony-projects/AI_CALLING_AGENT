"""Routine polling in the access log.

Docker probes /health every 30s, and every open dashboard tab asks /dashboard/live every
5s. Together that is tens of thousands of lines a day — a 40-minute capture of one real
call was already mostly live polls, with the call buried underneath.

Both drops are narrowed rather than blanket. /health only goes when it comes from
loopback, so an external probe is still visible. /dashboard/live only goes when it
returned 200: a 401 there is exactly what you want to see while debugging a session.

Record shape is taken from uvicorn's own call site:
    access_logger.info('%s - "%s %s HTTP/%s" %d',
                       client_addr, method, path_with_query, http_version, status)
"""

import logging

import pytest

from app.main import ASGILoggingMiddleware, DropRoutinePollingFilter

LIVE = "/api/v1/dashboard/live"


def _record(client: str, path: str, method: str = "GET", status: int = 200):
    """A LogRecord matching what uvicorn actually emits."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client, method, path, "1.1", status),
        exc_info=None,
    )


@pytest.fixture
def drop():
    return DropRoutinePollingFilter().filter


# --- healthcheck --------------------------------------------------------------------


def test_the_docker_probe_is_dropped(drop):
    assert drop(_record("127.0.0.1:38996", "/health")) is False


def test_an_external_health_request_still_logs(drop):
    """Someone probing /health from outside is worth seeing; Docker's own loop is not."""
    assert drop(_record("172.18.0.5:55680", "/health")) is True


def test_a_path_merely_starting_with_health_from_outside_survives(drop):
    assert drop(_record("203.0.113.9:1234", "/healthcheck-probe")) is True


# --- dashboard live poll ------------------------------------------------------------


def test_a_successful_live_poll_is_dropped(drop):
    assert drop(_record("49.207.194.207:0", LIVE)) is False


def test_the_live_poll_is_dropped_regardless_of_client(drop):
    """It arrives through nginx carrying the operator's own IP, never loopback."""
    assert drop(_record("172.18.0.5:44120", LIVE)) is False


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_a_failing_live_poll_still_logs(drop, status):
    """The whole reason to look at this endpoint's log is that it stopped returning 200."""
    assert drop(_record("49.207.194.207:0", LIVE, status=status)) is True


def test_other_dashboard_endpoints_still_log(drop):
    """Only the 5-second poll is quiet. Reads of lead data stay auditable."""
    assert drop(_record("49.207.194.207:0", "/api/v1/dashboard/leads")) is True
    assert drop(_record("49.207.194.207:0", "/api/v1/dashboard/overview?days=30")) is True


# --- everything else ----------------------------------------------------------------


def test_real_traffic_is_never_dropped(drop):
    assert drop(_record("127.0.0.1:38996", "/api/v1/campaigns/x/dial/vobiz", "POST")) is True
    assert drop(_record("172.18.0.5:46754", "/vobiz/answer/a/b")) is True


def test_a_malformed_record_is_kept(drop):
    """A filter that raises would take the whole log line down with it."""
    bad = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "boom", None, None)
    assert drop(bad) is True

    short = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, "%s %s %s", ("a", "b", LIVE), None
    )
    assert drop(short) is True


def test_the_redaction_filter_still_runs():
    """Both filters attach to the same logger; adding one must not displace the other."""
    attached = {type(f).__name__ for f in logging.getLogger("uvicorn.access").filters}
    assert "RedactCallTokenFilter" in attached
    assert "DropRoutinePollingFilter" in attached


# --- the middleware's second line ---------------------------------------------------


@pytest.mark.parametrize("path", ["/health", LIVE])
def test_middleware_is_quiet_for_the_same_paths(path):
    """Each request costs two lines, one from here and one from uvicorn. Silencing only
    one halves the noise and leaves the log just as unreadable."""
    assert path in ASGILoggingMiddleware._QUIET_PATHS


@pytest.mark.parametrize(
    "path", ["/vobiz/answer/a/b", "/api/v1/campaigns/", "/ws/vobiz/a/b", "/api/v1/dashboard/leads"]
)
def test_middleware_still_logs_everything_else(path):
    assert path not in ASGILoggingMiddleware._QUIET_PATHS
