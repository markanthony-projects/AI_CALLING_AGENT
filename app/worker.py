import json
from openai import AsyncOpenAI
from app.models.schemas import LeadExtraction
from app.models.db import Lead, Call, Transcript
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from arq.connections import RedisSettings
from loguru import logger
from sqlalchemy import select

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def process_extraction(ctx, transcript: str, call_sid: str):
    logger.info(f"Worker received extraction job for {call_sid}")
    async with AsyncSessionLocal() as db:
        try:
            # 1. Fetch Call Record
            query = select(Call).where(Call.call_sid == call_sid)
            result = await db.execute(query)
            call_record = result.scalars().first()
            
            if not call_record:
                logger.error(f"Cannot process transcript. Call record {call_sid} not found.")
                return
                
            # 2. Insert Transcript
            transcript_record = Transcript(call_id=call_record.id, full_text=transcript)
            db.add(transcript_record)
            await db.commit()

            # 3. Extract Lead Details
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "Extract lead details from the real estate conversation. Output as JSON matching the schema. Crucially, determine if is_prospect is true or false based on if they showed any genuine interest. If spam or wrong number, is_prospect is false."
                    },
                    {"role": "user", "content": transcript}
                ],
                response_format={"type": "json_object"}
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            lead_data = LeadExtraction(**extracted_data)
            
            if lead_data.is_prospect:
                # 4. Create Lead and Link to Call
                lead = Lead()
                for key, value in lead_data.model_dump(exclude={"is_prospect"}, exclude_unset=True).items():
                    if value is not None:
                        setattr(lead, key, value)
                        
                db.add(lead)
                await db.commit()
                await db.refresh(lead)
                
                # Link Call to Lead
                call_record.lead_id = lead.id
                await db.commit()
                
                logger.success(
                    f"[{call_sid}] 🟢 LEAD EXTRACTED\n"
                    f"   Name: {lead.name}\n"
                    f"   Budget: {lead.budget}\n"
                    f"   Timeline: {lead.timeline}\n"
                    f"   Summary: {lead.summary}"
                )
            else:
                logger.info(f"[{call_sid}] ⚪ NOT A PROSPECT: No genuine interest detected. Skipping lead creation.")
                
        except Exception as e:
            logger.error(f"[{call_sid}] Worker extraction failed: {e}")
            await db.rollback()

class WorkerSettings:
    functions = [process_extraction]
    redis_settings = RedisSettings(host="localhost", port=6379)
