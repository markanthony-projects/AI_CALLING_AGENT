"""add agent_name to projects

The agent introduces itself as a module constant — "I am Priya" — so every campaign on the
deployment is the same person. A builder running two campaigns has two teams, and the name
on the call should say which one.

Nullable, and no backfill. Every existing project keeps a NULL here and the greeting uses
the default exactly as it does today; a schema change must not alter what a live call says
before anybody has typed the new value in.

One per campaign rather than one per call, on purpose. dial_attempts records how often a
number is dialled more than once, and a prospect who meets Priya on Tuesday and Ananya on
Thursday has been called by a company that does not know who called them.

Revision ID: f3a91d7c2e58
Revises: e7c4a2d91f36
Create Date: 2026-09-12
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "f3a91d7c2e58"
down_revision: Union[str, Sequence[str], None] = "e7c4a2d91f36"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("projects", sa.Column("agent_name", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("projects", "agent_name")
