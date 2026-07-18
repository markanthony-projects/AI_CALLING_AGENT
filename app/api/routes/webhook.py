from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.api.dependencies import get_db
from app.services.agent import run_voice_agent
from app.services.discovery import get_project_by_campaign
from app.models.db import Call, CallStatus
from loguru import logger
from app.utils.context_builder import build_campaign_context
from datetime import datetime
from app.services.extraction import dispatch_extraction_task

router = APIRouter()

ACTIVE_CALLS = 0
MAX_CALLS = 8

@router.post("/exotel/answer/{campaign_id}/{call_sid}")
async def exotel_answer(campaign_id: str, call_sid: str, request: Request):
    host = request.headers.get("host", "localhost:8000")
    protocol = "wss" if "localhost" not in host else "ws"
    ws_url = f"{protocol}://{host}/ws/exotel/{campaign_id}/{call_sid}"
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""
    return Response(content=xml, media_type="application/xml")

@router.websocket("/ws/exotel/{campaign_id}/{call_sid}")
async def exotel_webhook(
    websocket: WebSocket,
    campaign_id: str,
    call_sid: str,
    db: AsyncSession = Depends(get_db)
):
    global ACTIVE_CALLS
    
    if ACTIVE_CALLS >= MAX_CALLS:
        logger.warning(f"Concurrency limit reached (MAX {MAX_CALLS}). Rejecting call {call_sid}")
        await websocket.close(code=1013)
        return
        
    await websocket.accept()
    ACTIVE_CALLS += 1
    started_at = datetime.utcnow()
    logger.info(f"Incoming Exotel WS | Campaign {campaign_id} | Call {call_sid} | Active: {ACTIVE_CALLS}")
    
    transcript = ""
    try:
        # Create Call Record
        try:
            call_record = Call(campaign_id=campaign_id, call_sid=call_sid, status=CallStatus.IN_PROGRESS, started_at=started_at)
            db.add(call_record)
            await db.commit()
        except Exception as e:
            logger.warning(f"Could not insert Call record (might exist): {e}")
            await db.rollback()

        project = await get_project_by_campaign(db, campaign_id)
        if not project:
            logger.warning(f"Project not found for Campaign {campaign_id}. Closing connection.")
            await websocket.close(code=1008)
            return
            
        campaign_context = build_campaign_context(project)
        logger.info(f"Successfully loaded DB context for Call {call_sid}")
        
        transcript = await run_voice_agent(websocket, campaign_context, call_sid, project_name=project["name"])
        
    except WebSocketDisconnect:
        logger.info(f"Call {call_sid} disconnected. Webhook exited gracefully.")
    except Exception as e:
        logger.error(f"Error handling call {call_sid}: {e}")
    finally:
        ACTIVE_CALLS -= 1
        ended_at = datetime.utcnow()
        duration = (ended_at - started_at).total_seconds()
        try:
            stmt = update(Call).where(Call.call_sid == call_sid).values(
                status=CallStatus.COMPLETED,
                ended_at=ended_at,
                duration_seconds=duration
            )
            await db.execute(stmt)
            await db.commit()
            logger.info(f"Call {call_sid} finalized. Duration: {duration}s. Active calls: {ACTIVE_CALLS}")
            
            # Dispatch extraction to Redis ONLY after successfully committing DB status
            if transcript:
                await dispatch_extraction_task(transcript, call_sid)
        except Exception as e:
            logger.error(f"Failed to finalize Call DB record or extraction for {call_sid}: {e}")

@router.websocket("/ws/browser/{campaign_id}/{call_sid}")
async def browser_webhook(
    websocket: WebSocket,
    campaign_id: str,
    call_sid: str,
    db: AsyncSession = Depends(get_db)
):
    global ACTIVE_CALLS
    
    if ACTIVE_CALLS >= MAX_CALLS:
        logger.warning(f"Concurrency limit reached (MAX {MAX_CALLS}). Rejecting call {call_sid}")
        await websocket.close(code=1013)
        return
        
    await websocket.accept()
    ACTIVE_CALLS += 1
    started_at = datetime.utcnow()
    logger.info(f"Incoming Browser WS | Campaign {campaign_id} | Call {call_sid} | Active: {ACTIVE_CALLS}")
    
    transcript = ""
    try:
        try:
            call_record = Call(campaign_id=campaign_id, call_sid=call_sid, status=CallStatus.IN_PROGRESS, started_at=started_at)
            db.add(call_record)
            await db.commit()
        except Exception as e:
            logger.warning(f"Could not insert Call record: {e}")
            await db.rollback()

        project = await get_project_by_campaign(db, campaign_id)
        if not project:
            await websocket.close(code=1008)
            return
            
        campaign_context = build_campaign_context(project)
        
        transcript = await run_voice_agent(websocket, campaign_context, call_sid, client_type="browser", project_name=project["name"])
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error handling browser call {call_sid}: {e}")
    finally:
        ACTIVE_CALLS -= 1
        ended_at = datetime.utcnow()
        duration = (ended_at - started_at).total_seconds()
        try:
            stmt = update(Call).where(Call.call_sid == call_sid).values(
                status=CallStatus.COMPLETED,
                ended_at=ended_at,
                duration_seconds=duration
            )
            await db.execute(stmt)
            await db.commit()
            logger.info(f"Browser call {call_sid} finalized.")
            
            # Dispatch extraction to Redis ONLY after successfully committing DB status
            if transcript:
                await dispatch_extraction_task(transcript, call_sid)
        except Exception as e:
            logger.error(f"Failed to finalize Browser Call DB record or extraction for {call_sid}: {e}")
