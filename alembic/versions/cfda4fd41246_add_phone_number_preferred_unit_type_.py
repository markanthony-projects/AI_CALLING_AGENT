"""add phone_number, preferred_unit_type, lead created_at, call index

Adds the contact details the system was discarding. A booked site visit was unreachable
because the dialled number was never stored anywhere, only logged.

created_at is added nullable and backfilled from each lead's originating call before the
NOT NULL constraint goes on: adding a NOT NULL column outright would fail against the
existing rows. The NUMERIC columns gain bounded precision — an unbounded NUMERIC had been
storing durations as 54.51915999999999939973349682986736.

Revision ID: cfda4fd41246
Revises: 4d1cfd6979ea
Create Date: 2026-07-29 17:50:32.293525
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "cfda4fd41246"
down_revision: Union[str, Sequence[str], None] = "4d1cfd6979ea"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UTC_NOW = sa.text("(now() at time zone 'utc')")


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("calls", sa.Column("phone_number", sa.String(), nullable=True))
    op.create_index(op.f("ix_calls_phone_number"), "calls", ["phone_number"], unique=False)
    # calls are read by campaign on every dashboard query; this index was declared in
    # scripts/init_db.sql but never actually existed in the database.
    op.create_index(op.f("ix_calls_campaign_id"), "calls", ["campaign_id"], unique=False)

    op.add_column("leads", sa.Column("phone_number", sa.String(), nullable=True))
    op.create_index(op.f("ix_leads_phone_number"), "leads", ["phone_number"], unique=False)
    op.add_column("leads", sa.Column("preferred_unit_type", sa.String(), nullable=True))

    # 1. add nullable  2. backfill from the originating call  3. enforce NOT NULL
    op.add_column("leads", sa.Column("created_at", sa.DateTime(), nullable=True))
    op.execute(
        """
        UPDATE leads l
        SET created_at = COALESCE(
            (SELECT MIN(c.started_at) FROM calls c WHERE c.lead_id = l.id),
            (now() at time zone 'utc')
        )
        WHERE l.created_at IS NULL
        """
    )
    op.alter_column("leads", "created_at", nullable=False, server_default=UTC_NOW)

    op.alter_column(
        "calls", "duration_seconds",
        existing_type=sa.Numeric(), type_=sa.Numeric(10, 2), existing_nullable=True,
    )
    op.alter_column(
        "leads", "budget",
        existing_type=sa.Numeric(), type_=sa.Numeric(14, 2), existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "leads", "budget",
        existing_type=sa.Numeric(14, 2), type_=sa.Numeric(), existing_nullable=True,
    )
    op.alter_column(
        "calls", "duration_seconds",
        existing_type=sa.Numeric(10, 2), type_=sa.Numeric(), existing_nullable=True,
    )
    op.drop_column("leads", "created_at")
    op.drop_column("leads", "preferred_unit_type")
    op.drop_index(op.f("ix_leads_phone_number"), table_name="leads")
    op.drop_column("leads", "phone_number")
    op.drop_index(op.f("ix_calls_campaign_id"), table_name="calls")
    op.drop_index(op.f("ix_calls_phone_number"), table_name="calls")
    op.drop_column("calls", "phone_number")
