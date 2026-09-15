"""Yesterday's numbers, in one message, to wherever the team reads in the morning.

The call_metrics table answers "was yesterday slower?" — but only to somebody who opens the
dashboard and asks. A product whose USP is latency needs the number to arrive uninvited.
This builds one short summary of the previous IST day and posts it to a webhook: any
service that takes {"text": ...} — a Slack or Google Chat incoming webhook, a WhatsApp
bridge, a Discord hook — and logs the same line either way, so the summary exists even
when nothing is configured to receive it.

Medians of per-call p50/p95 rather than a pooled percentile across turns: the question is
"what did a typical call feel like", and one runaway call should move the worst column,
not the typical one. Counts are summed, because one bad call is exactly what a count is for.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.db import CallMetrics

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass(frozen=True)
class DaySummary:
    day: date
    calls: int
    p50_ms: Optional[int]
    p95_ms: Optional[int]
    worst_ms: Optional[int]
    first_word_ms: Optional[int]
    held_replies: int
    tts_reconnects: int
    llm_failures: int


def ist_day_bounds(day: date) -> tuple[datetime, datetime]:
    """The naive-UTC start and end of an IST calendar day, for a created_at stored naive UTC."""
    start_ist = datetime(day.year, day.month, day.day, tzinfo=_IST)
    start = start_ist.astimezone(timezone.utc).replace(tzinfo=None)
    return start, start + timedelta(days=1)


async def summarise_day(db: AsyncSession, day: date) -> DaySummary:
    start, end = ist_day_bounds(day)
    row = (
        await db.execute(
            select(
                func.count(CallMetrics.id),
                func.percentile_cont(0.5).within_group(CallMetrics.p50_turn_ms),
                func.percentile_cont(0.5).within_group(CallMetrics.p95_turn_ms),
                func.max(CallMetrics.max_turn_ms),
                func.percentile_cont(0.5).within_group(CallMetrics.first_word_ms),
                func.coalesce(func.sum(CallMetrics.held_replies), 0),
                func.coalesce(func.sum(CallMetrics.tts_reconnects), 0),
                func.coalesce(func.sum(CallMetrics.llm_failures), 0),
            ).where(CallMetrics.created_at >= start, CallMetrics.created_at < end)
        )
    ).one()
    calls, p50, p95, worst, first_word, held, reconnects, failures = row
    return DaySummary(
        day=day,
        calls=int(calls or 0),
        p50_ms=round(float(p50)) if p50 is not None else None,
        p95_ms=round(float(p95)) if p95 is not None else None,
        worst_ms=int(worst) if worst is not None else None,
        first_word_ms=round(float(first_word)) if first_word is not None else None,
        held_replies=int(held or 0),
        tts_reconnects=int(reconnects or 0),
        llm_failures=int(failures or 0),
    )


def _ms(value: Optional[int]) -> str:
    return f"{value:,} ms" if value is not None else "—"


def render(today: DaySummary, yesterday: Optional[DaySummary] = None) -> str:
    """One message a person reads in ten seconds: the number, and whether it moved."""
    if today.calls == 0:
        return f"Calls {today.day:%a %d %b}: none placed."

    def delta(now: Optional[int], before: Optional[int]) -> str:
        if now is None or before is None or yesterday is None or yesterday.calls == 0:
            return ""
        diff = now - before
        if abs(diff) < 50:
            return " (flat)"
        return f" ({'+' if diff > 0 else '−'}{abs(diff):,} vs prev day)"

    lines = [
        f"Calls {today.day:%a %d %b}: {today.calls} call{'s' if today.calls != 1 else ''}",
        f"Typical turn p50 {_ms(today.p50_ms)}{delta(today.p50_ms, yesterday.p50_ms if yesterday else None)}",
        f"Typical worst turn p95 {_ms(today.p95_ms)}{delta(today.p95_ms, yesterday.p95_ms if yesterday else None)}",
        f"Slowest single turn {_ms(today.worst_ms)}",
        f"First word {_ms(today.first_word_ms)} after the line opened",
    ]
    trouble = []
    if today.held_replies:
        trouble.append(f"{today.held_replies} half-sentence replies held back")
    if today.tts_reconnects:
        trouble.append(f"{today.tts_reconnects} voice reconnects")
    if today.llm_failures:
        trouble.append(f"{today.llm_failures} LLM failures")
    lines.append("Trouble: " + (", ".join(trouble) if trouble else "none"))
    return "\n".join(lines)


async def post(text: str) -> bool:
    """Deliver the summary. False when there is nowhere to send it or it did not arrive."""
    url = (settings.METRICS_SUMMARY_WEBHOOK_URL or "").strip()
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json={"text": text})
            if response.status_code >= 300:
                logger.error(f"Metrics summary webhook answered {response.status_code}: {response.text[:200]}")
                return False
        return True
    except Exception as e:  # noqa: BLE001 — a summary that cannot be sent must not fail the job
        logger.error(f"Metrics summary webhook failed: {e}")
        return False


async def send_yesterdays_summary(db: AsyncSession, now_ist: Optional[datetime] = None) -> str:
    """Build, log and post the summary for the previous IST day. Returns the text."""
    now_ist = now_ist or datetime.now(_IST)
    today = now_ist.date()
    text = render(
        await summarise_day(db, today - timedelta(days=1)),
        await summarise_day(db, today - timedelta(days=2)),
    )
    logger.info("METRICS SUMMARY\n" + text)
    if await post(text):
        logger.info("Metrics summary delivered to the webhook")
    return text
