from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator
from app.core.database import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
