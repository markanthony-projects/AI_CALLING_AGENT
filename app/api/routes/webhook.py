from datetime import datetime

from fastapi import APIRouter, Depends, Request, Response, WebSocket, WebSocketDisconnect
from loguru import logger
from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.security import issue_call_token, require_call_token, require_call_token_ws
from app.models.db import Call, CallStatus, Transcript
from app.services.agent import run_voice_agent
from app.services.call_context import recall_customer_name, recall_dialed_number
from app.services.discovery import get_project_by_campaign
from app.services.extraction import enqueue_extraction
from app.utils.context_builder import build_campaign_context
from app.utils.timeutils import utc_now

router = APIRouter()

ACTIVE_CALLS = 0

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
    global ACTIVE_CALLS

    if ACTIVE_CALLS >= settings.MAX_CONCURRENT_CALLS:
        logger.warning(
            f"[{call_sid}] Concurrency limit reached "
            f"(max {settings.MAX_CONCURRENT_CALLS}); rejecting call"
        )
        await websocket.close(code=1013)
        return

    await websocket.accept()
    ACTIVE_CALLS += 1
    _STREAMING_CALLS.add(call_sid)
    started_at = utc_now()
    logger.info(f"[{call_sid}] {client_type} stream open | campaign={campaign_id} | active={ACTIVE_CALLS}")

    transcript = ""
    # Default to FAILED: any path that leaves this block without a clean voice session —
    # missing project, pipeline crash, exhausted LLM turns — is a call we did not conduct.
    status = CallStatus.FAILED
    try:
        async with AsyncSessionLocal() as db:
            db.add(Call(
                campaign_id=campaign_id,
                call_sid=call_sid,
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
            customer_name=await recall_customer_name(call_sid),
        )
        transcript = result.transcript
        if result.error:
            logger.error(f"[{call_sid}] Voice session ended in error: {result.error}")
        elif result.answering_machine:
            # Not COMPLETED: recording a voicemail as a connect flatters the campaign's
            # answer rate in the dashboard, and the number still deserves a retry.
            status = CallStatus.MACHINE
        else:
            status = CallStatus.COMPLETED
    except WebSocketDisconnect:
        # The caller hanging up is how a normal call ends, not a system failure.
        logger.info(f"[{call_sid}] Stream disconnected")
        status = CallStatus.COMPLETED
    except Exception as e:
        logger.error(f"[{call_sid}] Error handling call: {e}")
    finally:
        ACTIVE_CALLS -= 1
        _STREAMING_CALLS.discard(call_sid)
        await _finalize_call(call_sid, started_at, transcript, status)


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
        f"[{call_sid}] Call finalised | status={status.value} | duration={duration:.1f}s | active={ACTIVE_CALLS}"
    )
    # A failed session still yields a partial transcript worth extracting a lead from.
    # A voicemail does not: the transcript is somebody's outgoing greeting, and running it
    # through the extractor costs an OpenAI call to learn that the person was not home.
    if status is CallStatus.MACHINE:
        logger.info(f"[{call_sid}] Answering machine; skipping extraction")
        return
    if transcript_stored:
        await enqueue_extraction(call_sid)
