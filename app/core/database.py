from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    # See DB_POOL_SIZE in app/core/config.py for why these are set rather than left to the
    # default. The short version: two processes, one managed cluster, and a connection that
    # dies while idle is only discovered by the request that needs it — which on this service
    # is a live call trying to write its own row.
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Request-scoped session.

    Lives here rather than in app.api so that app.core.security can depend on it without
    core importing from the api layer — require_session needs a session to re-read the
    signed-in user, and going through Depends is what lets tests substitute one.
    """
    async with AsyncSessionLocal() as session:
        yield session
