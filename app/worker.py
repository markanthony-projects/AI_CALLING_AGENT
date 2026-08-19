import json
from datetime import datetime
from typing import Optional

from arq import cron
from loguru import logger
from openai import AsyncOpenAI
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal, engine
from app.core.queue import redis_settings
from app.models.db import Call, Campaign, Lead, LeadStatus, Project, Transcript
from app.models.schemas import LeadExtraction
from app.services.dial_pump import dial_due_contacts, release_stale_dialing
from app.utils.attribution import budget_is_grounded, phrase_is_grounded
from app.utils.timeutils import is_within_business_hours, resolve_appointment, to_ist, utc_now

EXTRACTION_MODEL = "gpt-4o-mini"

openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _build_system_prompt(reference_time: datetime) -> str:
    ref_time_str = reference_time.strftime("%A, %Y-%m-%d %H:%M:%S (IST / UTC+05:30)")
    return (
        "Extract lead details from the real estate conversation.\n"
        f"THE CALL TOOK PLACE ON: {ref_time_str}. Use this only to understand the conversation.\n"
        "CRITICAL: Do NOT calculate any dates. Report only what was SAID and the system will resolve the calendar:\n"
        "- 'Sunday works' -> site_visit_weekday = SUNDAY. Do not work out which date that is.\n"
        "- 'tomorrow' -> site_visit_in_days = 1. 'Today' -> 0. Use this ONLY for relative wording.\n"
        "- '3 PM' -> site_visit_at = '15:00'. Always 24-hour HH:MM.\n"
        "- Use the callback_* fields for a callback and the site_visit_* fields for a visit.\n"
        "CRITICAL: NEVER invent an appointment. If the prospect did not agree to a specific time, leave the *_at field null — "
        "a vague 'yeah sure', 'this weekend' or 'sometime' is NOT an appointment. Never fill in a default hour.\n\n"
        "CRITICAL RULE FOR is_prospect: behaves as a universal real estate lead qualification flag. "
        "is_prospect MUST BE TRUE whenever the caller shares a budget, specifies a locality, asks about pricing, "
        "requests a callback or site visit, or shows ANY genuine interest in purchasing real estate—EVEN IF their budget is lower or higher than the project being pitched! "
        "(e.g., a caller with a 75 Lakhs budget on a 1.2 Cr project IS A VALID PROSPECT for other portfolio projects). "
        "ONLY set is_prospect to false if the call is spam, wrong number, telemarketer, or the caller explicitly refuses all real estate discussions.\n"
        "CRITICAL RULE FOR budget: ONLY output a valid numerical float (e.g. 8000000) or null. Convert Lakhs/Crores to full float (e.g. 75 Lakhs = 7500000).\n"
        "CRITICAL RULE FOR ATTRIBUTION: Extract ONLY facts the Prospect stated about themselves. "
        "Every 'Agent:' line describes the LISTING — its locality, asking price, size and amenities. None of it is prospect data. "
        "Before filling any field, find the 'Prospect:' line it came from. If it only appears on an 'Agent:' line, the value is null.\n"
        "- budget: only a figure the Prospect named as their own spend. 'Our 3 BHK is priced at 1.8 Crores' and "
        "'most buyers look in the 1 to 3.5 Crores range' are the Agent quoting the ASKING PRICE — never record either as the budget. "
        "So is 'starting at 1.2 Crores', 'prices start from 90 Lakhs', 'this one is 1.8 Crores' and every other figure that follows "
        "the Agent describing what is on sale. A Prospect answering 'yes', 'okay' or 'sounds good' to any of those has agreed to keep "
        "listening — they have NOT told you their budget, and budget stays null.\n"
        "TEST BEFORE YOU FILL budget: can you point to the Prospect line where they said the number themselves? If not, it is null. "
        "A wrong budget is worse than none: sales calls the person back believing they can spend money they never mentioned.\n"
        "Asking what something costs is not having that budget. Agreeing to a site visit is not agreeing to a price. "
        "If the Prospect dodged the budget question or answered without naming a figure, budget is null.\n"
        "- preferred_unit_type: only a configuration the PROSPECT asked about or chose. 'We have 2 BHK, 3 BHK and villaments' is the "
        "Agent listing stock — picking one out of that list is inventing a preference. Null unless the Prospect named one themselves.\n"
        "- preferred_location: only a locality the Prospect named. Never copy one out of the Agent's pitch.\n"
        "CRITICAL RULE FOR transliterated_transcript: You MUST transliterate the entire conversation transcript into English/Latin script (e.g. convert 'कौन सा प्रोजेक्ट' to 'Kaun sa project'). "
        "This applies to EVERY non-Latin word including names — 'कुमार' becomes 'Kumar', 'प्रिया' becomes 'Priya'. "
        "No Devanagari character may remain anywhere in your output. "
        "Preserve the transcript's structure exactly: keep every 'Agent:' and 'Prospect:' turn on its own line, separated by newlines, in the original order. Do not merge turns into a paragraph."
    )


async def _extract_lead(transcript: str, reference_time: datetime, call_sid: str) -> LeadExtraction:
    system_prompt = _build_system_prompt(reference_time)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]

    try:
        response = await openai_client.beta.chat.completions.parse(
            model=EXTRACTION_MODEL,
            messages=messages,
            response_format=LeadExtraction,
        )
        parsed = response.choices[0].message.parsed
        if parsed is not None:
            return parsed
        raise ValueError("structured output returned no parsed payload")
    except Exception as e:
        logger.warning(f"[{call_sid}] Structured output failed, falling back to JSON mode: {e}")

    response = await openai_client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    for wrap_key in ("lead_details", "lead", "data", "extraction"):
        inner = payload.get(wrap_key)
        if isinstance(inner, dict):
            payload = inner
            break
    return LeadExtraction(**payload)


def _normalise(value: str) -> str:
    """'2bhk' / '2 B H K' / '2-BHK' all reduce to the same key."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


# What a prospect can want that this project does not sell. Asked "what kind of property
# are you looking for?" one answered "plots"; it matched none of the project's BHK configs
# and was stored as null, throwing away the single most useful fact on the record. That
# question is only ever asked on the path where the pitch has already failed, so the answer
# is by definition about some other project — matching it against this one's stock is
# backwards.
_GENERIC_UNIT_TYPES = {
    "plot": "Plot",
    "plots": "Plot",
    "site": "Plot",
    "land": "Plot",
    "villa": "Villa",
    "villas": "Villa",
    "villament": "Villament",
    "rowhouse": "Row House",
    "independenthouse": "Independent House",
    "apartment": "Apartment",
    "apartments": "Apartment",
    "flat": "Apartment",
    "flats": "Apartment",
    "penthouse": "Penthouse",
    "studio": "Studio",
    "commercial": "Commercial",
    "office": "Commercial",
}


def _generic_unit_type(spoken: str) -> Optional[str]:
    """Map free text onto a property type that exists outside this project's stock."""
    key = _normalise(spoken)
    if key in _GENERIC_UNIT_TYPES:
        return _GENERIC_UNIT_TYPES[key]
    # "looking for plots" / "a villa maybe" — the answer is rarely just the noun.
    for word, canonical in _GENERIC_UNIT_TYPES.items():
        if word in key:
            return canonical
    return None


async def _match_unit_type(db, campaign_id, spoken: Optional[str], call_sid: str) -> Optional[str]:
    """Map what the prospect said onto a unit the project actually sells.

    Free text here would give sales '2bhk', '2 BHK' and 'two bedroom' for one configuration,
    none of which can be filtered on. The project's config_json is the authoritative list,
    so anything that does not match one of its entries is dropped rather than stored.
    """
    if not spoken:
        return None

    project = (
        await db.execute(
            select(Project)
            .join(Campaign, Campaign.project_id == Project.id)
            .where(Campaign.id == campaign_id)
        )
    ).scalars().first()

    configs = project.config_json if project and isinstance(project.config_json, list) else []
    known = [c.get("type") for c in configs if isinstance(c, dict) and c.get("type")]

    target = _normalise(spoken)
    for unit in known:
        if _normalise(unit) == target:
            return unit
    # Substring both ways catches "2 BHK apartment" against "2 BHK".
    for unit in known:
        key = _normalise(unit)
        if key and (key in target or target in key):
            return unit

    generic = _generic_unit_type(spoken)
    if generic:
        logger.info(
            f"[{call_sid}] Unit type {spoken!r} is not in this project's stock; "
            f"recording it as {generic!r} for the portfolio"
        )
        return generic

    logger.warning(
        f"[{call_sid}] Unit type {spoken!r} matches none of {known or 'this project'} "
        f"and is not a property type we recognise; storing null"
    )
    return None


def _resolve(reference, weekday, in_days, time_of_day, call_sid: str, label: str):
    """Resolve one appointment intent, logging whatever we refuse to guess at."""
    resolved = resolve_appointment(
        reference, weekday=weekday, in_days=in_days, time_of_day=time_of_day
    )
    if resolved is None:
        if weekday or in_days is not None or time_of_day:
            logger.info(
                f"[{call_sid}] Partial {label} discarded "
                f"(weekday={weekday}, in_days={in_days}, at={time_of_day!r}) — not a booking"
            )
        return None

    logger.info(f"[{call_sid}] {label.capitalize()} resolved to {resolved:%A %Y-%m-%d %H:%M}")
    if not is_within_business_hours(resolved):
        logger.warning(f"[{call_sid}] {label.capitalize()} falls outside business hours: {resolved:%H:%M}")
    return resolved


def _drop_ungrounded(lead_data: LeadExtraction, transcript: str, call_sid: str) -> LeadExtraction:
    """Null any prospect-owned field the prospect is not on record as having said.

    The extraction prompt already forbids copying from the agent's pitch, in capitals, with
    worked examples. It was ignored twice — once recording the asking price as the caller's
    budget, once recording the project's own locality as the area they wanted. A wrong
    value here is worse than a missing one: sales dials someone believing facts about them
    that came out of our own script.

    Only fields the prospect owns are checked. customer_name is not one of them — the
    dial list supplies it and the prospect never has to repeat it.
    """
    checks = (
        ("budget", budget_is_grounded(lead_data.budget, transcript), lead_data.budget),
        ("preferred_location", phrase_is_grounded(lead_data.preferred_location, transcript), lead_data.preferred_location),
        ("preferred_unit_type", phrase_is_grounded(lead_data.preferred_unit_type, transcript), lead_data.preferred_unit_type),
    )
    dropped = {}
    for field, grounded, value in checks:
        if not grounded:
            dropped[field] = None
            logger.warning(
                f"[{call_sid}] Dropping {field}={value!r}: no Prospect line says it. "
                f"This is the agent's own pitch being read back as the caller's requirement."
            )
    return lead_data.model_copy(update=dropped) if dropped else lead_data


async def process_extraction(ctx: dict, call_sid: str) -> None:
    async with AsyncSessionLocal() as db:
        call_record = (await db.execute(select(Call).where(Call.call_sid == call_sid))).scalars().first()
        if call_record is None:
            logger.error(f"[{call_sid}] Call record not found; dropping extraction job.")
            return
        if call_record.lead_id is not None:
            logger.info(f"[{call_sid}] Lead already extracted; skipping.")
            return

        transcript_record = (
            await db.execute(select(Transcript).where(Transcript.call_id == call_record.id))
        ).scalars().first()
        if transcript_record is None or not transcript_record.full_text.strip():
            logger.error(f"[{call_sid}] No transcript stored; dropping extraction job.")
            return

        # started_at is naive UTC; the prompt presents it as IST, so convert or every
        # relative date the caller gave lands 5h30m early.
        reference_time = to_ist(call_record.started_at or utc_now())
        lead_data = await _extract_lead(transcript_record.full_text, reference_time, call_sid)

        # Grounded against the transliterated text where we have it: the extracted values
        # come back in Latin script, so checking them against a Devanagari transcript would
        # find nothing and discard everything the prospect said in Hindi.
        lead_data = _drop_ungrounded(
            lead_data,
            lead_data.transliterated_transcript or transcript_record.full_text,
            call_sid,
        )

        if lead_data.transliterated_transcript:
            transcript_record.full_text = lead_data.transliterated_transcript
        # Silent until now: the model romanises customer_name but has left Devanagari in the
        # transcript, and sales read the transcript.
        if not transcript_record.full_text.isascii():
            logger.warning(f"[{call_sid}] Transcript still holds non-Latin script after transliteration")

        if not lead_data.is_prospect:
            await db.commit()
            logger.info(f"[{call_sid}] Not a prospect; no lead created.")
            return

        lead = Lead(
            # Carried from the Call so a booked visit can actually be confirmed, leads can
            # be de-duplicated, and DND can be honoured.
            phone_number=call_record.phone_number,
            customer_name=lead_data.customer_name,
            preferred_location=lead_data.preferred_location,
            preferred_unit_type=await _match_unit_type(
                db, call_record.campaign_id, lead_data.preferred_unit_type, call_sid
            ),
            budget=lead_data.budget,
            purpose=lead_data.purpose,
            timeline=lead_data.timeline,
            timeline_months=lead_data.timeline_months,
            status=lead_data.status,
            # leads.* are TIMESTAMP WITHOUT TIME ZONE holding IST wall-clock time, which is
            # what sales actually dials against.
            site_visit_time=_resolve(
                reference_time, lead_data.site_visit_weekday, lead_data.site_visit_in_days,
                lead_data.site_visit_at, call_sid, "site visit",
            ),
            callback_time=_resolve(
                reference_time, lead_data.callback_weekday, lead_data.callback_in_days,
                lead_data.callback_at, call_sid, "callback",
            ),
        )

        if lead.customer_name and not lead.customer_name.isascii():
            logger.warning(f"[{call_sid}] customer_name not transliterated: {lead.customer_name!r}")

        # is_prospect already qualified them, so a missing status is the model declining to
        # answer rather than a judgement that the lead is cold. Leaving it null hides the lead
        # from every status filter sales uses, which is worse than guessing one notch high.
        if lead.status is None:
            logger.warning(f"[{call_sid}] Extraction returned no status; defaulting to WARM")
            lead.status = LeadStatus.WARM

        # A booked site visit is the strongest buying signal there is; don't leave it to the model.
        if lead.site_visit_time is not None and lead.status is not LeadStatus.HOT:
            logger.info(f"[{call_sid}] Site visit booked; upgrading lead from {lead.status} to HOT")
            lead.status = LeadStatus.HOT

        db.add(lead)
        await db.flush()
        call_record.lead_id = lead.id
        await db.commit()

        logger.success(
            f"[{call_sid}] Lead extracted | name={lead.customer_name} | budget={lead.budget} "
            f"| purpose={lead.purpose} | timeline={lead.timeline_months}mo ({lead.timeline}) "
            f"| unit={lead.preferred_unit_type} | area={lead.preferred_location} "
            f"| phone={lead.phone_number} | status={lead.status} "
            f"| site_visit={lead.site_visit_time} | callback={lead.callback_time}"
        )


async def on_shutdown(ctx: dict) -> None:
    await engine.dispose()


# --- the dial pump ---------------------------------------------------------------------


async def pump_dials(ctx: dict) -> int:
    """Place as many calls as the carrier has free slots for, and no more.

    Runs on a short timer rather than being scheduled per number. See
    app/services/dial_pump.py for why a pull model is what survives a worker restart, two
    workers, a full carrier and a paused campaign without a special case for any of them.

    Every exception is swallowed. This is the only thing that moves the queue, and a tick
    that raises would be retried by arq with a job that has already dialled — so a transient
    Postgres blip would place the same calls twice.
    """
    try:
        return await dial_due_contacts()
    except Exception as e:
        logger.error(f"Dial pump tick failed: {e}")
        return 0


async def unstick_dials(ctx: dict) -> int:
    """Return contacts stuck mid-dial to the queue.

    A dial whose call never opened a websocket leaves its row DIALING for ever, and DIALING
    is not eligible — so that number is silently never called again while its row looks busy
    rather than broken.
    """
    try:
        return await release_stale_dialing()
    except Exception as e:
        logger.error(f"Could not unstick mid-dial contacts: {e}")
        return 0


class WorkerSettings:
    functions = [process_extraction]
    cron_jobs = [
        # Every five seconds. Short because it bounds how long a freed slot sits idle, which
        # on a three-slot account is a third of the throughput; cheap because a tick with no
        # free slots and no active campaign is two indexed queries.
        cron(pump_dials, second={s for s in range(0, 60, 5)}, run_at_startup=True, max_tries=1),
        # Slower: this only cleans up after something that has already gone wrong.
        cron(unstick_dials, minute={0, 15, 30, 45}, max_tries=1),
    ]
    redis_settings = redis_settings()
    on_shutdown = on_shutdown
    max_jobs = 10
    max_tries = 4
    job_timeout = 120
    keep_result = 3600
    retry_jobs = True
    health_check_interval = 60
