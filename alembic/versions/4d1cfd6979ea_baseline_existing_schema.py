"""baseline existing schema

Creates the schema as it stood before Alembic took ownership of it — i.e. exactly what
SQLAlchemy's create_all used to build. Later revisions layer their changes on top.

This revision was originally empty, because it was written against a database whose tables
create_all had already made. That is fine for that one database and broken everywhere else:
on a fresh volume there was nothing to alter, so the next revision died on
`ALTER TABLE calls ADD COLUMN` and a first deployment could never start.

The upgrade is therefore skipped when the tables are already present, so an existing
database can still be baselined without tripping over "relation already exists".

Revision ID: 4d1cfd6979ea
Revises:
Create Date: 2026-07-29 17:49:53.588796

"""
from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '4d1cfd6979ea'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("transcripts", "calls", "leads", "campaigns", "projects")


def _schema_exists() -> bool:
    return sa.inspect(op.get_bind()).has_table("calls")


def upgrade() -> None:
    """Upgrade schema."""
    if _schema_exists():
        # Pre-Alembic database being brought under version control: record the revision
        # and change nothing.
        return

    # Vector(1536) needs the extension in place. scripts/init_db.sql does this on a fresh
    # Postgres volume, but a managed database or a re-pointed DATABASE_URL will not have
    # run it, and CREATE EXTENSION needs privileges the app user may lack — so attempt it
    # here and let the CREATE TABLE below produce the real error if it truly is missing.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("campaign_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("locality", sa.String(), nullable=False),
        sa.Column("min_price", sa.Numeric(), nullable=True),
        sa.Column("max_price", sa.Numeric(), nullable=True),
        sa.Column("config_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("amenities", sa.dialects.postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("nearby_facilities", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("possession_status", sa.String(), nullable=True),
        sa.Column("usps", sa.dialects.postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("rera_id", sa.String(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_projects_campaign_id"), "projects", ["campaign_id"], unique=False)

    op.create_table(
        "campaigns",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("ACTIVE", "PAUSED", "COMPLETED", name="campaignstatus"),
            nullable=True,
        ),
        sa.Column("start_date", sa.DateTime(), nullable=True),
        sa.Column("end_date", sa.DateTime(), nullable=True),
        sa.Column("total_leads_dialed", sa.Numeric(), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "leads",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("customer_name", sa.String(), nullable=True),
        sa.Column("preferred_location", sa.String(), nullable=True),
        sa.Column("budget", sa.Numeric(), nullable=True),
        sa.Column("timeline", sa.String(), nullable=True),
        sa.Column("callback_time", sa.DateTime(), nullable=True),
        sa.Column("site_visit_time", sa.DateTime(), nullable=True),
        sa.Column("status", sa.Enum("HOT", "WARM", "COLD", name="leadstatus"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "calls",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("campaign_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("lead_id", sa.UUID(as_uuid=True), nullable=True),
        sa.Column("call_sid", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("IN_PROGRESS", "COMPLETED", "FAILED", name="callstatus"),
            nullable=True,
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Numeric(), nullable=True),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"]),
        sa.ForeignKeyConstraint(["lead_id"], ["leads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_calls_call_sid"), "calls", ["call_sid"], unique=True)
    # No index on calls.campaign_id here — the next revision creates it.

    op.create_table(
        "transcripts",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("call_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("full_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["call_id"], ["calls.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("call_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    for table in _TABLES:
        op.drop_table(table)
    for enum_name in ("callstatus", "leadstatus", "campaignstatus"):
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
