"""Container healthcheck noise in the access log.

Docker probes /health every 30 seconds. At two lines per probe that is roughly 5,700 lines
a day, which is enough to bury the one line that matters when a call fails. The probe is
dropped; anything from outside loopback still gets logged.

Record shape is taken from uvicorn's own call site:
    access_logger.info('%s - "%s %s HTTP/%s" %d',
                       client_addr, method, path_with_query, http_version, status)
"""

import logging

import pytest

from app.main import ASGILoggingMiddleware, DropHealthCheckFilter


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
    return DropHealthCheckFilter().filter


def test_the_docker_probe_is_dropped(drop):
    assert drop(_record("127.0.0.1:38996", "/health")) is False


def test_an_external_health_request_still_logs(drop):
    """Someone probing /health from outside is worth seeing; Docker's own loop is not."""
    assert drop(_record("172.18.0.5:55680", "/health")) is True


def test_real_traffic_is_never_dropped(drop):
    assert drop(_record("127.0.0.1:38996", "/api/v1/campaigns/x/dial/vobiz", "POST")) is True
    assert drop(_record("172.18.0.5:46754", "/vobiz/answer/a/b")) is True


def test_a_path_merely_starting_with_health_from_outside_survives(drop):
    assert drop(_record("203.0.113.9:1234", "/healthcheck-probe")) is True


def test_a_malformed_record_is_kept(drop):
    """A filter that raises would take the whole log line down with it."""
    bad = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, "boom", None, None)
    assert drop(bad) is True
    assert drop(_record("127.0.0.1", "/health", status=200)) is False


def test_the_redaction_filter_still_runs():
    """Both filters attach to the same logger; adding one must not displace the other."""
    attached = {type(f).__name__ for f in logging.getLogger("uvicorn.access").filters}
    assert "RedactCallTokenFilter" in attached
    assert "DropHealthCheckFilter" in attached


def test_middleware_also_skips_health():
    """The middleware logs a second line per request; both have to be quiet."""
    assert "/health" in ASGILoggingMiddleware._QUIET_PATHS


@pytest.mark.parametrize("path", ["/vobiz/answer/a/b", "/api/v1/campaigns/", "/ws/vobiz/a/b"])
def test_middleware_still_logs_everything_else(path):
    assert path not in ASGILoggingMiddleware._QUIET_PATHS
