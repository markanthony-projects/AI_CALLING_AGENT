"""leads.status not null defaulting to warm

Revision ID: db0682dd0be0
Revises: cfda4fd41246
Create Date: 2026-07-30 12:38:38.715082

A null status hides a lead from every filter the sales team works from. Thirteen of the
twenty-four leads on record had one, including a prospect who named a 1.5 Crore budget and
agreed to a site visit.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'db0682dd0be0'
down_revision: Union[str, Sequence[str], None] = 'cfda4fd41246'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LEADSTATUS = postgresql.ENUM('HOT', 'WARM', 'COLD', name='leadstatus', create_type=False)


def upgrade() -> None:
    # Autogenerate proposed SET NOT NULL on its own, which fails against rows already holding
    # null. Backfill first. None of those rows has a site visit, so WARM is the floor the
    # application now applies rather than a claim about how interested those prospects were.
    op.execute("UPDATE leads SET status = 'WARM' WHERE status IS NULL")
    op.alter_column(
        "leads",
        "status",
        existing_type=LEADSTATUS,
        server_default=sa.text("'WARM'"),
        nullable=False,
    )


def downgrade() -> None:
    # Drops the constraint only. Which rows were null before the backfill is not recoverable,
    # and inventing that distinction again would be worse than leaving them WARM.
    op.alter_column(
        "leads",
        "status",
        existing_type=LEADSTATUS,
        server_default=None,
        nullable=True,
    )
