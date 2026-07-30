from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes.webhook import router as webhook_router
from app.api.routes.campaign import router as campaign_router
from app.core.config import settings
from sqlalchemy import text

from app.core.database import engine
from app.core.queue import init_arq_pool, close_arq_pool
from contextlib import asynccontextmanager
from loguru import logger
import os
import logging
import mimetypes
import re

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

class ASGILoggingMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
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

    yield

    await close_arq_pool()
    try:
        await engine.dispose()
    except Exception:
        pass
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="Indian Real Estate Sales Voice Agent",
    lifespan=lifespan,
    docs_url="/docs" if settings.DOCS_ENABLED else None,
    redoc_url="/redoc" if settings.DOCS_ENABLED else None,
    openapi_url="/openapi.json" if settings.DOCS_ENABLED else None,
)
app.add_middleware(ASGILoggingMiddleware)

# Mount static files for the web test client
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(webhook_router)
app.include_router(campaign_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "auth": "enabled" if settings.AUTH_ENABLED else "DISABLED"}
