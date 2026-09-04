import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response, WebSocket, WebSocketDisconnect
from loguru import logger
from sqlalchemy import select

from app.core import call_slots
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.security import issue_call_token, require_call_token, require_call_token_ws
from app.models.db import Call, CallStatus, Transcript
from app.services.agent import run_voice_agent
from app.services.call_context import recall_customer_name, recall_dialed_number
from app.services.dial_pump import recall_contact, record_carrier_outcome, record_outcome
from app.services.discovery import get_project_by_campaign
from app.services.extraction import enqueue_extraction
from app.utils.attribution import prospect_text
from app.utils.context_builder import build_campaign_context
from app.utils.timeutils import utc_now

router = APIRouter()

_HANGUP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Hangup/>
</Response>"""

_NO_INSTRUCTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
</Response>"""

# call_sids whose websocket is streaming right now. The Call row cannot answer this
# question: it is written when the stream opens, so it exists for the whole of a healthy
# call, and treating its presence as "already served" hangs up live callers.
_STREAMING_CALLS: set[str] = set()


@router.post("/vobiz/answer/{campaign_id}/{call_sid}", dependencies=[Depends(require_call_token)])
async def vobiz_answer(campaign_id: str, call_sid: str, request: Request):
    # Never hang up a call whose stream is still running. Doing so terminated live callers
    # about two seconds in, while the agent carried on talking to a dead leg.
    if call_sid in _STREAMING_CALLS:
        logger.warning(f"[{call_sid}] Answer webhook re-fetched mid-stream; returning no new instructions")
        return Response(content=_NO_INSTRUCTION_XML, media_type="application/xml")

    # A call token stays valid for its whole TTL, so a carrier retry or a replayed URL can
    # land here after the call is done. Hang up rather than hand out a second stream and
    # pay for another leg.
    async with AsyncSessionLocal() as db:
        already_served = await db.scalar(select(Call.id).where(Call.call_sid == call_sid))

    if already_served:
        logger.warning(f"[{call_sid}] Answer webhook replayed after the call ended; hanging up")
        return Response(content=_HANGUP_XML, media_type="application/xml")

    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)

    if forwarded_host:
        ws_proto = "wss" if forwarded_proto in ("https", "wss") else "ws"
        base_ws_url = f"{ws_proto}://{forwarded_host}"
    else:
        base_ws_url = settings.WEBHOOK_BASE_URL.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")

    ws_token = issue_call_token(campaign_id, call_sid)
    ws_url = f"{base_ws_url}/ws/vobiz/{campaign_id}/{call_sid}?token={ws_token}"
    logger.info(f"[{call_sid}] Issued Vobiz stream URL for campaign {campaign_id}")

    # keepCallAlive="true" makes <Stream> execute exclusively, so <Hangup/> runs only once
    # the stream disconnects — which is what ends the PSTN leg and stops the billing.
    # The default (false) runs later elements *concurrently* with the stream, so <Hangup/>
    # fired ~2s in and killed the caller while the agent talked on to a dead line.
    # https://vobiz.ai/docs/xml/stream
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=16000">{ws_url}</Stream>
    <Hangup/>
</Response>"""
    return Response(content=xml, media_type="application/xml")


@router.websocket("/ws/vobiz/{campaign_id}/{call_sid}", dependencies=[Depends(require_call_token_ws)])
async def vobiz_webhook(websocket: WebSocket, campaign_id: str, call_sid: str):
    await _handle_call(websocket, campaign_id, call_sid, client_type="vobiz")


@router.websocket("/ws/browser/{campaign_id}/{call_sid}", dependencies=[Depends(require_call_token_ws)])
async def browser_webhook(websocket: WebSocket, campaign_id: str, call_sid: str):
    await _handle_call(websocket, campaign_id, call_sid, client_type="browser")


async def _handle_call(websocket: WebSocket, campaign_id: str, call_sid: str, client_type: str) -> None:
    # The slot was taken before the dial, by the pump. Acquiring here is idempotent per
    # call_sid, so this refreshes that reservation rather than consuming a second one — and it
    # covers the paths that do not come from the pump: a manual dial, and the browser client.
    #
    # Refusing here is a last resort and no longer the main defence. It used to be the only
    # check, and it sat after Vobiz had dialled, billed us and rung a real person, who then
    # got the line closed on them with no Call row to show it. See app/core/call_slots.py.
    if not await call_slots.acquire(call_sid):
        logger.error(
            f"[{call_sid}] At carrier capacity ({settings.MAX_CONCURRENT_CALLS}) with the "
            f"call already connected — refusing the stream. The dial should not have been "
            f"placed; check that it came through the pump."
        )
        await websocket.close(code=1013)
        # Recorded rather than dropped silently. A billed call that nobody could serve is
        # exactly the thing that was invisible before.
        await _record_refused(campaign_id, call_sid)
        return

    await websocket.accept()
    _STREAMING_CALLS.add(call_sid)
    started_at = utc_now()
    # Deliberately not reporting the in-flight count here. Reading it is another round trip
    # to Redis on the path that opens a live call, and it was being spent on a log line. The
    # count is on the dashboard, where somebody is actually looking at it.
    logger.info(f"[{call_sid}] {client_type} stream open | campaign={campaign_id}")

    transcript = ""
    # Default to FAILED: any path that leaves this block without a clean voice session —
    # missing project, pipeline crash, exhausted LLM turns — is a call we did not conduct.
    status = CallStatus.FAILED
    # Whether the agent itself decided the call was over. end_call is the only way it ever
    # does, so anything else means the call stopped for a reason nobody chose. Bound out
    # here because the queue entry is settled in `finally`, where `result` may not exist.
    end_reason: Optional[str] = None
    try:
        async with AsyncSessionLocal() as db:
            db.add(Call(
                campaign_id=campaign_id,
                call_sid=call_sid,
                contact_id=contact_uuid(await recall_contact(call_sid)),
                phone_number=await recall_dialed_number(call_sid),
                status=CallStatus.IN_PROGRESS,
                started_at=started_at,
            ))
            await db.commit()
            project = await get_project_by_campaign(db, campaign_id)

        if not project:
            logger.warning(f"[{call_sid}] No project for campaign {campaign_id}; closing stream")
            await websocket.close(code=1008)
            return

        result = await run_voice_agent(
            websocket,
            build_campaign_context(project),
            call_sid,
            client_type=client_type,
            project_name=project["name"],
            # .get, not [...]: the project dict is cached in Redis for 24 hours, so an entry
            # written before this field existed will simply not have the key. Missing means
            # the greeting names the project, which is what it did before either way.
            developer_name=project.get("developer_name"),
            customer_name=await recall_customer_name(call_sid),
        )
        transcript = result.transcript
        # Alongside the status, because "COMPLETED" answers whether the call worked and this
        # answers who ended it — and those were indistinguishable when a call finished with no
        # reason logged at all.
        end_reason = result.end_reason
        if result.end_reason:
            logger.info(f"[{call_sid}] Ended by: {result.end_reason}")
        if result.error:
            logger.error(f"[{call_sid}] Voice session ended in error: {result.error}")
        elif result.answering_machine:
            # Not COMPLETED: recording a voicemail as a connect flatters the campaign's
            # answer rate in the dashboard, and the number still deserves a retry.
            status = CallStatus.MACHINE
        else:
            status = CallStatus.COMPLETED
    except WebSocketDisconnect:
        # The stream going away is how a normal call ends, not a system failure. It does not
        # say who ended it, so it is not recorded as the caller having done so.
        logger.info(f"[{call_sid}] Stream disconnected")
        end_reason = "the media stream closed"
        status = CallStatus.COMPLETED
    except Exception as e:
        logger.error(f"[{call_sid}] Error handling call: {e}")
    finally:
        _STREAMING_CALLS.discard(call_sid)
        # Before the DB write, and before anything that can raise: this frees the carrier
        # slot, and a slot that is not released blocks a third of the account's capacity
        # until it ages out.
        await call_slots.release(call_sid)
        await _finalize_call(call_sid, started_at, transcript, status)
        # The queue entry, so a number that rang out becomes eligible for a retry and one
        # that spoke to the agent is never dialled again. Kept out of _finalize_call because
        # that function owns the call history, which is a separate record.
        #
        # answered_words counts only what the prospect said. It counted the whole transcript,
        # agent lines included — and the agent always speaks, so the count was never zero and
        # the "connected but no conversation" branch in record_outcome could not run.
        spoken_words = len(prospect_text(transcript).split())
        closed_by_agent = end_reason == "end_call tool"
        # Loud, and on its own line, because it is countable: this is a prospect who was
        # talking to us when the call stopped. One of them on 2 Sep 2026 turned out to be the
        # carrier tearing the stream down mid-sentence, and nothing in the logs said so.
        if status is CallStatus.COMPLETED and spoken_words > 0 and not closed_by_agent:
            logger.warning(
                f"[{call_sid}] Call cut off mid-conversation | ended_by={end_reason!r} | "
                f"prospect said {spoken_words} words | the agent never closed it"
            )
        await record_outcome(
            await recall_contact(call_sid),
            status,
            answered_words=spoken_words,
            closed_by_agent=closed_by_agent,
        )


def contact_uuid(raw):
    """The contact id carried through Redis, as a UUID, or None.

    Tolerant on purpose: this sits in the path that opens a live call, and a malformed value
    must cost the link between a call and its queue entry, never the call itself.
    """
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        logger.warning(f"Ignoring an unusable contact id on a dial: {raw!r}")
        return None


async def _record_refused(campaign_id: str, call_sid: str) -> None:
    """Write down a call that was billed and could not be served.

    The old code returned before the Call row was created, so a caller who was dialled,
    charged for and hung up on left no trace at all — which is why nobody knew it was
    happening. Best effort: if this write fails there is nothing further to try.
    """
    try:
        async with AsyncSessionLocal() as db:
            db.add(Call(
                campaign_id=campaign_id,
                call_sid=call_sid,
                contact_id=contact_uuid(await recall_contact(call_sid)),
                phone_number=await recall_dialed_number(call_sid),
                status=CallStatus.FAILED,
                started_at=utc_now(),
                ended_at=utc_now(),
                duration_seconds=0,
            ))
            await db.commit()
    except Exception as e:
        logger.error(f"[{call_sid}] Could not record the refused call: {e}")
    await record_outcome(await recall_contact(call_sid), CallStatus.FAILED)


async def _finalize_call(
    call_sid: str, started_at: datetime, transcript: str, status: CallStatus
) -> None:
    ended_at = utc_now()
    duration = (ended_at - started_at).total_seconds()
    transcript_stored = False

    try:
        async with AsyncSessionLocal() as db:
            call_record = (await db.execute(select(Call).where(Call.call_sid == call_sid))).scalars().first()
            if call_record is None:
                logger.error(f"[{call_sid}] Call record missing; transcript cannot be persisted")
                return

            call_record.status = status
            call_record.ended_at = ended_at
            call_record.duration_seconds = duration

            if transcript:
                existing = await db.scalar(select(Transcript.id).where(Transcript.call_id == call_record.id))
                if existing is None:
                    db.add(Transcript(call_id=call_record.id, full_text=transcript))
                    transcript_stored = True

            await db.commit()
    except Exception as e:
        logger.error(f"[{call_sid}] Failed to finalise call record: {e}")
        return

    logger.info(
        f"[{call_sid}] Call finalised | status={status.value} | duration={duration:.1f}s"
    )
    # A failed session still yields a partial transcript worth extracting a lead from.
    # A voicemail does not: the transcript is somebody's outgoing greeting, and running it
    # through the extractor costs an OpenAI call to learn that the person was not home.
    if status is CallStatus.MACHINE:
        logger.info(f"[{call_sid}] Answering machine; skipping extraction")
        return
    if transcript_stored:
        await enqueue_extraction(call_sid)

async def callback_fields(request: Request) -> dict:
    """The carrier's callback body, whichever way it was encoded.

    Vobiz posts form-encoded, as its Plivo-shaped API does everywhere else. JSON is accepted
    too because a body we cannot read looks exactly like a call that never happened, and the
    cost of guessing wrong is a held slot and a contact stuck mid-dial.
    """
    try:
        form = await request.form()
        if form:
            return {k: v for k, v in form.items()}
    except Exception:
        pass
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def carrier_call_id(fields: dict) -> str:
    """The carrier's own id for this call, out of its callback.

    Ours is a UUID we mint; theirs is the one their dashboard and their support desk index
    by, and the two are unrelated. Without this, asking them about one specific call meant
    matching it up by phone number and wall-clock time.
    """
    lowered = {k.lower(): v for k, v in fields.items()}
    for key in ("calluuid", "call_uuid", "requestuuid", "request_uuid"):
        value = str(lowered.get(key) or "").strip()
        if value:
            return value
    return ""


def was_answered(fields: dict) -> bool:
    """Whether the carrier says a human picked up.

    An answer time is the one field this can rest on. Hangup causes differ between carriers
    and this integration has seen exactly one of them; an answer time either exists or does
    not. Duration is a second witness for a carrier that reports one and not the other —
    both are read case-insensitively because the callback's own casing is not ours to rely on.
    """
    lowered = {k.lower(): v for k, v in fields.items()}
    answer_time = str(lowered.get("answertime") or lowered.get("answer_time") or "").strip()
    if answer_time and answer_time.lower() not in ("none", "null", "0"):
        return True
    duration = str(lowered.get("duration") or lowered.get("billsec") or "").strip()
    try:
        return float(duration) > 0
    except ValueError:
        return False


def hangup_cause_of(fields: dict) -> str:
    lowered = {k.lower(): v for k, v in fields.items()}
    for key in ("hangupcause", "hangup_cause", "hangupcausename", "hangup_cause_name"):
        value = str(lowered.get(key) or "").strip()
        if value:
            return value
    return ""


@router.post("/vobiz/hangup/{campaign_id}/{call_sid}", dependencies=[Depends(require_call_token)])
async def vobiz_hangup(campaign_id: str, call_sid: str, request: Request):
    """The carrier telling us a call is over — including the ones that never began.

    This is the only report a ring-out produces. The media websocket opens on answer, so
    before this endpoint existed a call nobody picked up left the carrier slot held until it
    aged out twelve minutes later and the contact sitting in DIALING until the reaper found
    it. On a three-slot account three unanswered dials stopped the queue for twelve minutes.

    Two things happen here and their order matters. The slot is released first and
    unconditionally: the call is over either way, and freeing it is what lets the pump dial
    the next number. Only then is the contact's fate decided, and only for calls that were
    never answered — an answered call is finalised by the session that served it, which is
    the only side that knows whether the prospect said anything.

    Always answers 200. Vobiz retries a failed callback up to three times, and a retry storm
    on a body we could not parse would hold nothing open but would bury the log.
    """
    fields = await callback_fields(request)
    answered = was_answered(fields)
    cause = hangup_cause_of(fields)

    # Before anything that can fail. A slot that is not released blocks a third of the
    # account's capacity until it ages out, and nothing below is worth risking that for.
    await call_slots.release(call_sid)

    changed = False
    try:
        changed = await record_carrier_outcome(
            await recall_contact(call_sid), cause, answered
        )
    except Exception as e:
        # The reaper still returns the contact to the queue, so this is a delay, not a loss.
        logger.error(f"[{call_sid}] Could not record the carrier's hangup verdict: {e}")

    logger.info(
        f"[{call_sid}] Carrier hung up | answered={answered} | cause={cause or 'unreported'}"
        f" | vobiz_call_id={carrier_call_id(fields) or 'unreported'}"
        + (" | contact updated" if changed else "")
    )
    return Response(status_code=200)

