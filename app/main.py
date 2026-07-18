from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes.webhook import router as webhook_router
from app.api.routes.campaign import router as campaign_router
from app.core.database import engine, Base
from contextlib import asynccontextmanager
from loguru import logger
import os
import mimetypes

# Fix Windows registry MIME type issues for Javascript Worklets
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

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
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas verified.")
    except Exception as e:
        logger.error(f"Warning: Could not connect to database during startup. Please configure a valid DATABASE_URL in .env. Error: {e}")
    yield
    try:
        await engine.dispose()
    except Exception:
        pass
    logger.info("Application shutdown complete.")

app = FastAPI(title="Indian Real Estate Sales Voice Agent", lifespan=lifespan)
app.add_middleware(ASGILoggingMiddleware)

# Mount static files for the web test client
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(webhook_router)
app.include_router(campaign_router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
