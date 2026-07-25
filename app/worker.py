# Background extraction worker
import json
from datetime import datetime
from openai import AsyncOpenAI
from app.models.schemas import LeadExtraction
from app.models.db import Lead, Call, Transcript
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from loguru import logger
from sqlalchemy import select

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def process_extraction(transcript: str, call_sid: str):
    logger.info(f"[{call_sid}] Worker received extraction job. Transcript:\n{transcript}")
    async with AsyncSessionLocal() as db:
        try:
            # 1. Fetch Call Record
            query = select(Call).where(Call.call_sid == call_sid)
            result = await db.execute(query)
            call_record = result.scalars().first()
            
            if not call_record:
                logger.error(f"[{call_sid}] Cannot process transcript. Call record not found.")
                return
                
            # 2. Insert Transcript
            transcript_record = Transcript(call_id=call_record.id, full_text=transcript)
            db.add(transcript_record)
            await db.commit()

            # Base reference time for relative date/time calculations (e.g. "after 3 hours", "tomorrow at 11 AM")
            call_start_dt = call_record.started_at or datetime.utcnow()
            ref_time_str = call_start_dt.strftime("%A, %Y-%m-%d %H:%M:%S (IST / UTC+05:30)")

            system_prompt = (
                "Extract lead details from the real estate conversation.\n"
                f"BASE REFERENCE TIME FOR RELATIVE DATES & TIMES: {ref_time_str}.\n"
                "Calculate exact callback_time and site_visit_time relative to BASE REFERENCE TIME:\n"
                "- 'after X hours' -> Add X hours to BASE REFERENCE TIME.\n"
                "- 'tomorrow at 11 AM' -> Next calendar day at 11:00:00.\n"
                "- 'this Saturday' -> Nearest future Saturday at agreed time or 11:00:00.\n"
                "CRITICAL: Output callback_time and site_visit_time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Never output dummy years like 3000 or 1970.\n\n"
                "CRITICAL RULE FOR is_prospect: behaves as a universal real estate lead qualification flag. "
                "is_prospect MUST BE TRUE whenever the caller shares a budget, specifies a locality, asks about pricing, "
                "requests a callback or site visit, or shows ANY genuine interest in purchasing real estate—EVEN IF their budget is lower or higher than the project being pitched! "
                "(e.g., a caller with a 75 Lakhs budget on a 1.2 Cr project IS A VALID PROSPECT for other portfolio projects). "
                "ONLY set is_prospect to false if the call is spam, wrong number, telemarketer, or the caller explicitly refuses all real estate discussions.\n"
                "CRITICAL RULE FOR budget: ONLY output a valid numerical float (e.g. 8000000) or null. Convert Lakhs/Crores to full float (e.g. 75 Lakhs = 7500000).\n"
                "CRITICAL RULE FOR transliterated_transcript: You MUST transliterate the entire conversation transcript into English/Latin script (e.g. convert 'कौन सा प्रोजेक्ट' to 'Kaun sa project')."
            )

            # 3. Extract Lead Details & Transliterate
            import asyncio
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await openai_client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": transcript}
                        ],
                        response_format=LeadExtraction
                    )
                    lead_data = response.choices[0].message.parsed
                    break # Success, exit retry loop
                except Exception as parse_err:
                    logger.warning(f"[{call_sid}] Structured outputs parse error (attempt {attempt+1}/{max_retries}): {parse_err}")
                    if attempt == max_retries - 1:
                        # Final fallback to standard json_object if structured outputs repeatedly fails
                        response = await openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": transcript}
                            ],
                            response_format={"type": "json_object"}
                        )
                        extracted_data = json.loads(response.choices[0].message.content)
                        for wrap_key in ("lead_details", "lead", "data", "extraction"):
                            if wrap_key in extracted_data and isinstance(extracted_data[wrap_key], dict):
                                extracted_data = extracted_data[wrap_key]
                                break
                        lead_data = LeadExtraction(**extracted_data)
                    else:
                        await asyncio.sleep(2 ** attempt) # Exponential backoff

            if not lead_data or not lead_data.transliterated_transcript:
                if lead_data:
                    lead_data.transliterated_transcript = transcript
                else:
                    lead_data = LeadExtraction(is_prospect=True, transliterated_transcript=transcript)

            # Update the transcript with the transliterated version
            if lead_data.transliterated_transcript:
                transcript_record.full_text = lead_data.transliterated_transcript
                await db.commit()
            
            if lead_data.is_prospect:
                # 4. Create Lead and Link to Call
                lead = Lead()
                for key, value in lead_data.model_dump(exclude={"is_prospect", "transliterated_transcript"}, exclude_unset=True).items():
                    if value is not None:
                        # Sanitize datetime fields to offset-naive for PostgreSQL TIMESTAMP WITHOUT TIME ZONE
                        if isinstance(value, datetime) and value.tzinfo is not None:
                            value = value.replace(tzinfo=None)
                        setattr(lead, key, value)
                        
                db.add(lead)
                await db.commit()
                await db.refresh(lead)
                
                # Link Call to Lead
                call_record.lead_id = lead.id
                await db.commit()
                
                logger.success(
                    f"[{call_sid}] 🟢 LEAD EXTRACTED\n"
                    f"   Name: {lead.name if hasattr(lead, 'name') else lead.customer_name}\n"
                    f"   Budget: {lead.budget}\n"
                    f"   Timeline: {lead.timeline}"
                )
            else:
                logger.info(f"[{call_sid}] ⚪ NOT A PROSPECT: No genuine interest detected. Skipping lead creation.")
                
        except Exception as e:
            logger.error(f"[{call_sid}] Worker extraction failed: {e}")
            await db.rollback()
