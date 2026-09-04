"""add developer_name to projects

The agent introduces itself with whatever is in projects.name, so on a live call it said:

    "I am Priya calling you from Abhee Codename New Dimension."

That is the project. A real person calls from the company that is building it, and names
the project when they get to describing it. The two were the same field, so there was
nowhere to put the difference.

Nullable, and no backfill. Every existing project keeps a NULL here and the greeting falls
back to the project name exactly as it does today — a schema change must not alter what a
live call says before anybody has typed the new value in.

Revision ID: d5b81f0c3a72
Revises: c9a3f21e7b40
Create Date: 2026-09-04
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d5b81f0c3a72"
down_revision: Union[str, Sequence[str], None] = "c9a3f21e7b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("projects", sa.Column("developer_name", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("projects", "developer_name")
