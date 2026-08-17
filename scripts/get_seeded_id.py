import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.db import Campaign

async def get_campaign():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Campaign).limit(1))
        campaign = res.scalars().first()
        if campaign:
            print(f"CAMPAIGN_ID={campaign.id}")
        else:
            print("No campaign found.")

if __name__ == "__main__":
    asyncio.run(get_campaign())
