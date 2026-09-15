"""leads.whatsapp_number: where the brochure goes when it is not the number we dialled

The WhatsApp close now confirms the number — "on this number?" — and a prospect who gives a
different one has it read back and recorded here. Null means the number we called, which is
the common case and what sales already has in phone_number.

Revision ID: b7d3e9a24c51
Revises: a4c8e2f61b93
Create Date: 2026-09-15
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "b7d3e9a24c51"
down_revision: Union[str, Sequence[str], None] = "a4c8e2f61b93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("leads", sa.Column("whatsapp_number", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("leads", "whatsapp_number")
