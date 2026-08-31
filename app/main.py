from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes.webhook import router as webhook_router
from app.api.routes.campaign import router as campaign_router
from app.api.routes.auth import router as auth_router
from app.api.routes.contacts import router as contacts_router
from app.api.routes.dashboard import router as dashboard_router
from app.core.config import settings
from sqlalchemy import text

from app.core.database import engine
from app.core.queue import init_arq_pool, close_arq_pool
from app.services.stale_calls import SWEEP_EVERY_SECONDS, reap_stale_calls
from contextlib import asynccontextmanager
from loguru import logger
import asyncio
import os
import logging
import mimetypes
import re
import sys

# ---------------------------------------------------------------------------
# Logging: quiet by default, verbose for call-critical paths only.
#
# The default loguru sink emits everything at DEBUG, which means SQLAlchemy
# query plans, arq heartbeats, pipecat audio pipeline frames, and uvicorn
# access lines for every dashboard poll all land in the same stream as the
# one line that matters when a call fails.
#
# Policy:
#   WARNING+  always visible (errors, unexpected states, stale-slot reaping)
#   INFO      only for the modules that narrate a call end-to-end
#   DEBUG     never on a running server (set LOG_LEVEL=DEBUG locally only)
# ---------------------------------------------------------------------------

_CALL_MODULES = {
    "app.services.agent",
    "app.services.dial_pump",
    "app.services.dialer",
    "app.services.extraction",
    "app.services.stale_calls",
    "app.api.routes.webhook",
    "app.worker",
}


def _log_filter(record: dict) -> bool:
    """Let WARNING+ through always; INFO only for call-path modules."""
    if record["level"].no >= 30:  # WARNING = 30
        return True
    if record["level"].no >= 20:  # INFO = 20
        return record["name"] in _CALL_MODULES
    return False


_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()  # drop the default stderr sink
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    filter=_log_filter,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    colorize=False,
    enqueue=True,
)

# Silence third-party loggers that flood the stream with internal state.
for _noisy in (
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "arq.worker",
    "arq.connections",
    "pipecat",
    "httpx",
    "httpcore",
    "asyncio",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Pipecat also emits its startup banner through loguru directly. Intercept it by
# temporarily raising the loguru level before pipecat imports, then restoring it.
# Since pipecat is already imported by the time main.py loads (it's imported by
# agent.py which is imported by webhook.py), suppress its logger name instead.
# The filter above (_CALL_MODULES) already handles this — pipecat is not in the set.

# Fix Windows registry MIME type issues for Javascript Worklets
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

_TOKEN_IN_URL = re.compile(r"([?&]token=)[^\s&\"]+")


class RedactCallTokenFilter(logging.Filter):
    """Keep call tokens out of the access log.

    Uvicorn logs the full request line including the query string, so every answer and
    stream URL was writing a live JWT to disk. The token is replayable until it expires,
    so anyone reading the logs could open a stream on our account.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _TOKEN_IN_URL.sub(r"\1[REDACTED]", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


logging.getLogger("uvicorn.access").addFilter(RedactCallTokenFilter())


# The two endpoints nothing ever reads back. Docker probes /health every 30s and every open
# dashboard tab asks /dashboard/live every 5s; together that is tens of thousands of lines a
# day. Named once so the access-log filter below and the ASGI middleware further down cannot
# drift into silencing different things — each request costs one line from each of them, and
# quieting only one halves the noise while leaving the log just as unreadable.
HEALTH_PATH = "/health"
LIVE_PATH = "/api/v1/dashboard/live"
QUIET_PATHS = frozenset({HEALTH_PATH, LIVE_PATH})


class DropRoutinePollingFilter(logging.Filter):
    """Keep the two endpoints nothing ever reads out of the access log.

    Docker probes /health every 30s, and every open dashboard tab asks /dashboard/live
    every 5s. Together that is tens of thousands of lines a day, enough to bury the one
    line that matters when a call fails — a 40-minute capture of a real call was already
    mostly live polls.

    Both are narrowed rather than blanket-dropped. /health is only dropped from loopback,
    so an external probe still appears. The live poll is only dropped when it succeeded: a
    401 or a 500 there is the interesting case and stays.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True

        client, path = args[0], args[2]
        if not isinstance(client, str) or not isinstance(path, str):
            return True

        if client.startswith("127.0.0.1") and path.startswith(HEALTH_PATH):
            return False

        status = args[4] if len(args) >= 5 else None
        if path.startswith(LIVE_PATH) and status == 200:
            return False

        return True


logging.getLogger("uvicorn.access").addFilter(DropRoutinePollingFilter())




class ASGILoggingMiddleware:
    """One line per request, before anything else has a chance to fail.

    Uvicorn's own access log is written when a response is finished, so a request that hangs,
    disconnects, or dies inside a handler leaves no trace of having arrived. This runs first
    and records the arrival, which is what makes a call that went nowhere visible at all.

    Removed in 7f736e9 along with the ASGI request lines the deployment guide and every
    debugging session rely on, while tests/test_log_noise.py kept asserting against it — which
    took the whole suite down at collection and left CI red.

    The filter above drops the same two endpoints from uvicorn's log, but only conditionally:
    /health from loopback, /dashboard/live only when it returned 200. This one is
    unconditional, because a line recording arrival has nothing interesting to say about a
    poll that failed — uvicorn's own line covers that.
    """

    _QUIET_PATHS = QUIET_PATHS

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope.get("path") not in self._QUIET_PATHS:
            logger.info(f"ASGI Request: {scope['type']} {scope.get('path')}")
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    if not settings.AUTH_ENABLED:
        logger.warning("=" * 78)
        logger.warning("AUTH_ENABLED=false — API key and call tokens are NOT enforced.")
        logger.warning("Anyone who can reach this host can place calls at your expense.")
        logger.warning("Never run this way on a public host. Unset AUTH_ENABLED for production.")
        logger.warning("=" * 78)
    if not settings.dashboard_enabled:
        logger.info(
            "Dashboard disabled: DASHBOARD_SESSION_SECRET is unset. /api/v1/auth and "
            "/api/v1/dashboard will return 503."
        )
    # Alembic owns the schema. create_all here would build tables it has no record of, so a
    # fresh database would then fail its first migration. Verify connectivity and that
    # migrations have actually been applied instead.
    try:
        async with engine.begin() as conn:
            revision = await conn.scalar(text("SELECT version_num FROM alembic_version"))
        logger.info(f"Database connected at migration {revision}.")
    except Exception as e:
        logger.error(
            "Could not read alembic_version. Run 'alembic upgrade head' before starting, "
            f"and check DATABASE_URL. Error: {e}"
        )

    # Fail fast: without the queue, every completed call would silently lose its lead extraction
    await init_arq_pool()
    logger.info("Extraction queue connected.")

    # A call whose process died never got to write its own ending, so the row says
    # IN_PROGRESS for ever and the dashboard cannot tell it from a live call. Swept on a
    # timer as well as at startup, because the process that should have cleaned up is
    # usually the one that is gone. Failures here must never stop the app serving calls.
    async def sweep_stale_calls():
        while True:
            try:
                await reap_stale_calls()
            except Exception as e:
                logger.error(f"Stale call sweep failed: {e}")
            await asyncio.sleep(SWEEP_EVERY_SECONDS)

    sweeper = asyncio.create_task(sweep_stale_calls())

    yield

    sweeper.cancel()
    await close_arq_pool()
    try:
        await engine.dispose()
    except Exception:
        pass
    logger.info("Application shutdown complete.")

# Route uvicorn's own stdlib logger through the same loguru filter so access
# lines from dashboard polls don't appear even when uvicorn emits them.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title="Indian Real Estate Sales Voice Agent",
    lifespan=lifespan,
    docs_url="/docs" if settings.DOCS_ENABLED else None,
    redoc_url="/redoc" if settings.DOCS_ENABLED else None,
    openapi_url="/openapi.json" if settings.DOCS_ENABLED else None,
)
app.add_middleware(ASGILoggingMiddleware)

# Only when the dashboard is served from its own origin. Same-origin deployments — nginx
# serving the SPA and proxying /api on one host — need no CORS at all, and adding it there
# would only widen the surface. allow_credentials is required because the session travels
# in a cookie, and it is precisely why the origin list can never be "*".
if settings.dashboard_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.dashboard_cors_origins,
        allow_credentials=True,
        # DELETE is here for a reason: removing an import batch, a campaign and a
        # do-not-call entry are all DELETE, and the dashboard is served cross-origin. Leaving
        # it out makes those three fail at the preflight with a CORS error that looks like an
        # auth problem.
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    logger.info(f"CORS enabled for dashboard origins: {settings.dashboard_cors_origins}")

# Mount static files for the web test client
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(webhook_router)
app.include_router(campaign_router)
app.include_router(auth_router)
app.include_router(dashboard_router)
# Same prefix and session auth as the dashboard, kept apart because that module is the read
# side and everything here writes. See app/api/routes/contacts.py.
app.include_router(contacts_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "auth": "enabled" if settings.AUTH_ENABLED else "DISABLED"}
