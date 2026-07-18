import asyncio
import sys
import os

# Add the root directory to sys.path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.db import Project, Campaign, CampaignStatus

async def seed_data():
    print("Seeding database...")
    async with AsyncSessionLocal() as db:
        # Create a Project
        project = Project(
            name="Lakeview Residency",
            city="Bangalore",
            locality="Whitefield",
            min_price=120.0,
            max_price=250.0,
            amenities=["Swimming Pool", "Gym", "Clubhouse", "24/7 Security"],
            usps=["Lake facing apartments", "5 mins from upcoming Metro Station"],
            possession_status="Ready to Move",
            config_json=[
                {"type": "2 BHK", "area": "1200 sqft", "price": "1.2 Cr"},
                {"type": "3 BHK", "area": "1600 sqft", "price": "1.8 Cr"},
                {"type": "Villament", "area": "2500 sqft", "price": "3.5 Cr"}
            ]
        )
        db.add(project)
        await db.flush() # To get project.id
        
        # Create a Campaign linked to the Project
        campaign = Campaign(
            project_id=project.id,
            name="Diwali Mega Sale 2026",
            status=CampaignStatus.ACTIVE
        )
        db.add(campaign)
        
        await db.commit()
        print(f"\n✅ Successfully created Project: '{project.name}'")
        print(f"   Project ID: {project.id}")
        print(f"✅ Successfully created Campaign: '{campaign.name}'")
        print(f"   Campaign ID: {campaign.id}")
        
        print("\n🚀 You can now test the browser client by navigating to:")
        print(f"http://localhost:8000/static/index.html?campaign_id={campaign.id}")

if __name__ == "__main__":
    asyncio.run(seed_data())
