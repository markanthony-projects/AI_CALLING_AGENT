"""add dashboard_users

Revision ID: a7c31f9be204
Revises: db0682dd0be0
Create Date: 2026-07-30 23:10:00.000000

The dashboard runs in a browser and therefore cannot hold API_KEY, which dials and so
spends money on presentation. It signs in as a named user instead, and this table is where
those users live. Passwords are never stored — only an scrypt digest.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a7c31f9be204"
down_revision: Union[str, Sequence[str], None] = "db0682dd0be0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# create_type=False on the column, created explicitly below: letting the column create it
# means a downgrade/upgrade cycle fails on "type already exists".
DASHBOARDROLE = postgresql.ENUM("VIEWER", "ADMIN", name="dashboardrole", create_type=False)


def upgrade() -> None:
    DASHBOARDROLE.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "dashboard_users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("role", DASHBOARDROLE, server_default="VIEWER", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(now() at time zone 'utc')"),
            nullable=False,
        ),
    )
    # Unique, not merely indexed: login looks a user up by email, and two rows for one
    # address would make which password works depend on row order.
    op.create_index("ix_dashboard_users_email", "dashboard_users", ["email"], unique=True)

    # The dashboard's list and detail views sort and filter on these. Without them every
    # page load sequentially scans calls, which grows without bound.
    op.create_index("ix_calls_started_at", "calls", ["started_at"])
    op.create_index("ix_leads_created_at", "leads", ["created_at"])
    op.create_index("ix_leads_status", "leads", ["status"])
    op.create_index("ix_calls_status", "calls", ["status"])


def downgrade() -> None:
    op.drop_index("ix_calls_status", table_name="calls")
    op.drop_index("ix_leads_status", table_name="leads")
    op.drop_index("ix_leads_created_at", table_name="leads")
    op.drop_index("ix_calls_started_at", table_name="calls")
    op.drop_index("ix_dashboard_users_email", table_name="dashboard_users")
    op.drop_table("dashboard_users")
    DASHBOARDROLE.drop(op.get_bind(), checkfirst=True)
