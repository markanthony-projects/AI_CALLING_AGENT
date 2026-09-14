"""call_metrics: the numbers a call is judged by, in a table instead of a rotating log

Every column here was already a log line — LATENCY summary, FIRST WORD, held replies, TTS
reconnects, LLM failures — and the log rotates after a couple of days. Asked for the week's
p50 on 12 Sep 2026, the answer was grep. A product whose USP is latency has to answer that
from a table, and the dashboard has to be able to draw it.

One row per finished call, written once at finalisation, best effort. Cascades with the
call: a deleted call has no numbers worth keeping on their own.

Revision ID: a4c8e2f61b93
Revises: f3a91d7c2e58
Create Date: 2026-09-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a4c8e2f61b93"
down_revision: Union[str, Sequence[str], None] = "f3a91d7c2e58"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "call_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "call_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("calls.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("turns", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("p50_turn_ms", sa.Integer(), nullable=True),
        sa.Column("p95_turn_ms", sa.Integer(), nullable=True),
        sa.Column("max_turn_ms", sa.Integer(), nullable=True),
        sa.Column("first_word_ms", sa.Integer(), nullable=True),
        sa.Column("held_replies", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tts_reconnects", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tts_revivals", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("llm_failures", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("end_reason", sa.String(), nullable=True),
        sa.Column("stt", sa.String(), nullable=False),
        sa.Column("llm", sa.String(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
    )
    op.create_index("ix_call_metrics_call_id", "call_metrics", ["call_id"], unique=True)
    op.create_index("ix_call_metrics_created_at", "call_metrics", ["created_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_call_metrics_created_at", table_name="call_metrics")
    op.drop_index("ix_call_metrics_call_id", table_name="call_metrics")
    op.drop_table("call_metrics")
