"""Load Abhee Codename New Dimension and give it a campaign.

Deliberately a script and not an Alembic revision. Migrations describe the shape of the
database, and they run on every environment that ever gets built — a revision that inserts
one builder's project would recreate it on a fresh database forever, and its downgrade
would have to decide whether deleting a live project with call history is acceptable. This
is data, so it goes in by hand, once, and is safe to re-run.

Two values from the supplied JSON are corrected here rather than stored as given. Both
would otherwise be spoken to a prospect on a live call:

  min_price arrived as 117000000. The column is in LAKHS, not rupees — Lakeview stores 120
  for a project whose cheapest unit is quoted "1.2 Cr", and context_builder divides by 100
  to reach Crores. Stored as given, the agent would have offered flats at 1170000 Crores.

  config_json arrived as a dictionary of developer metadata. Every reader of that column
  expects a list of units: context_builder skips it outright unless isinstance(x, list),
  and worker._match_unit_type builds its vocabulary of valid unit types from it. A dict
  would have left the agent unable to name a single configuration and every extracted
  preferred_unit_type null, silently.
"""

import asyncio
import sys
import uuid

from sqlalchemy import select

from app.core.database import AsyncSessionLocal, engine
from app.models.db import Campaign, Project

PROJECT_NAME = "Abhee Codename New Dimension"
CAMPAIGN_NAME = "Abhee New Dimension — Pre-Launch EOI"

# Areas and prices are not in the supplied JSON, which carries only bhk_types. They come
# from the call script for this same project shared earlier; the 2 BHK price there, 1.17
# Crores, is exactly the starting price the JSON reports, which is what ties the two
# together. Without them the agent can name unit types but not size or price them.
CONFIGS = [
    {"type": "2 BHK", "area": "1181 - 1183 sqft", "price": "1.17 Cr"},
    {"type": "3 BHK Regular", "area": "1448 - 1454 sqft", "price": "1.46 Cr"},
    {"type": "3 BHK Comfort", "area": "1550 - 1558 sqft", "price": "1.56 Cr"},
    {"type": "3 BHK Luxury", "area": "1646 - 1703 sqft", "price": "1.64 - 1.74 Cr"},
    {"type": "3.5 BHK Presidential", "area": "2009 sqft", "price": "2.06 Cr"},
    {"type": "4.5 BHK Presidential", "area": "2556 sqft", "price": "2.64 Cr"},
]

AMENITIES = [
    "3-acre on-site golf course",
    "1.5-acre private lake",
    "4 clubhouses spanning over 1.5 lakh sqft",
    "Swimming pool with dining deck",
    "Gym lounge",
    "Sports courts",
    "Kids play zones",
    "Adventure play park",
    "Skywalk adventure park",
    "Master landscape garden",
    "Celebration lawn",
    "Bonfire lounge amphitheatre",
    "140 plus curated amenities",
]

USPS = [
    "Bengaluru's first Scotland-themed residential township",
    "45-acre golf township with an on-site 3-acre golf course",
    "1.5-acre private lake with uninterrupted views",
    "4 grand clubhouses spanning over 1.5 lakh sqft",
    "14 towers across 45 acres",
    "2, 3, 3.5 and 4.5 BHK configurations",
    "Pre-launch EOI pricing, around 20 to 30 Lakhs below the expected launch price",
    "RERA registered",
    "Direct connectivity to ITPL, Whitefield, Outer Ring Road and Sarjapur Road",
    # Carried as a selling point because there is no column for it and the agent must be
    # able to say it: quoting an EOI price without this is quoting a price that is not final.
    "Prices are indicative at EOI stage and exclude GST, FRC, PLC, advance maintenance, "
    "corpus and other statutory charges",
]

# Strings, not the objects the source JSON used. build_campaign_context joins these values
# straight into a sentence, so a list of dicts raises TypeError inside the call handler and
# kills the call before the agent speaks. Nothing on Lakeview exercised this: its
# nearby_facilities is null.
NEARBY = {
    "Connectivity": [
        "ITPL IT Park, 15 to 20 minutes",
        "Whitefield, 10 to 15 minutes",
        "Outer Ring Road",
        "Sarjapur Road",
    ],
    "Landmarks": [
        "Greenwood High School",
        "Dommasandra Circle",
    ],
}


async def main(attach_to: str | None) -> None:
    async with AsyncSessionLocal() as db:
        existing = (
            await db.execute(select(Project).where(Project.name == PROJECT_NAME))
        ).scalars().first()
        if existing is not None:
            print(f"Project already present: {existing.id}")
            project = existing
        else:
            project = Project(
                id=uuid.uuid4(),
                name=PROJECT_NAME,
                city="Bengaluru",
                locality="Varthur - Sarjapur Road, near Dommasandra Circle",
                # Lakhs. 117 = 1.17 Crores, 264 = 2.64 Crores (the 4.5 BHK).
                min_price=117,
                max_price=264,
                possession_status="Pre-Launch (EOI stage), launch expected mid 2025",
                rera_id="ACK/KA/RERA/1251/308/PR/130626/010325",
                amenities=AMENITIES,
                usps=USPS,
                nearby_facilities=NEARBY,
                config_json=CONFIGS,
            )
            db.add(project)
            await db.flush()
            print(f"Created project {project.id}")

        if attach_to:
            campaign = await db.get(Campaign, uuid.UUID(attach_to))
            if campaign is None:
                print(f"No campaign {attach_to}", file=sys.stderr)
                await db.rollback()
                return
            campaign.project_id = project.id
            print(f"Repointed campaign {campaign.id} ({campaign.name!r}) at {PROJECT_NAME}")
        else:
            campaign = (
                await db.execute(select(Campaign).where(Campaign.name == CAMPAIGN_NAME))
            ).scalars().first()
            if campaign is None:
                campaign = Campaign(
                    id=uuid.uuid4(), project_id=project.id, name=CAMPAIGN_NAME
                )
                db.add(campaign)
                await db.flush()
                print(f"Created campaign {campaign.id} ({CAMPAIGN_NAME!r})")
            else:
                campaign.project_id = project.id
                print(f"Campaign already present: {campaign.id}")

        await db.commit()
        campaign_id = str(campaign.id)

    # discovery.py caches campaign -> project for a day in Redis and 5 minutes in-process.
    # Without this a call started now would still be pitched the old project.
    try:
        from app.services.discovery import invalidate_project_cache

        await invalidate_project_cache(campaign_id)
        print("Cleared the campaign's project cache")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not clear the cache ({exc}); it expires on its own within a day")

    await engine.dispose()
    print(f"\nCampaign to dial with: {campaign_id}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
