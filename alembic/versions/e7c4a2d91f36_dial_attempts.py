"""dial_attempts: a row for every dial the pump asks the carrier to place

A reported night-time call could not be confirmed or ruled out from the data. Call rows
exist only for answered calls, the contact's last_attempt_at is overwritten by the next
retry, and the worker's logs are lost on every rebuild. This table is the record all three
lacked: one row per dial, written when the request goes to the carrier, completed once by
the carrier's hangup callback, never updated again and never deleted.

No backfill — nothing before this migration can be reconstructed, and a partial ledger
that looks complete is worse than one that starts on a known date.

Revision ID: e7c4a2d91f36
Revises: d5b81f0c3a72
Create Date: 2026-09-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e7c4a2d91f36"
down_revision: Union[str, Sequence[str], None] = "d5b81f0c3a72"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "dial_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("call_sid", sa.String(), nullable=False),
        sa.Column(
            "contact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("contacts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("campaign_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("phone_number", sa.String(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("dialed_at", sa.DateTime(), nullable=False),
        sa.Column("carrier_accepted", sa.Boolean(), nullable=False),
        sa.Column("answered", sa.Boolean(), nullable=True),
        sa.Column("hangup_cause", sa.String(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_dial_attempts_call_sid", "dial_attempts", ["call_sid"], unique=True)
    op.create_index("ix_dial_attempts_contact_id", "dial_attempts", ["contact_id"])
    op.create_index("ix_dial_attempts_campaign_id", "dial_attempts", ["campaign_id"])
    op.create_index("ix_dial_attempts_phone_number", "dial_attempts", ["phone_number"])
    op.create_index("ix_dial_attempts_dialed_at", "dial_attempts", ["dialed_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_dial_attempts_dialed_at", table_name="dial_attempts")
    op.drop_index("ix_dial_attempts_phone_number", table_name="dial_attempts")
    op.drop_index("ix_dial_attempts_campaign_id", table_name="dial_attempts")
    op.drop_index("ix_dial_attempts_contact_id", table_name="dial_attempts")
    op.drop_index("ix_dial_attempts_call_sid", table_name="dial_attempts")
    op.drop_table("dial_attempts")
