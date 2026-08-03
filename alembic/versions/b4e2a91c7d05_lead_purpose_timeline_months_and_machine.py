"""Record why they are buying, when, and whether a machine answered.

Three things live calls proved were missing:

  leads.purpose          "Are you buying for yourself or for investment?" is asked on nearly
                         every call and was answered plainly every time. There was nowhere
                         to put the answer, so it was thrown away — while the same lead
                         carried a budget and a locality that came out of the agent's own
                         pitch.

  leads.timeline_months  timeline held 'Maybe around in 2 months.' verbatim. A rep can read
                         it; nothing can sort or filter on it, so it could not be used to
                         decide who to call first. The words stay, the number joins them.

  calls.status MACHINE   A dial landed on voicemail and was recorded COMPLETED, which
                         inflates the campaign's answer rate in the direction that flatters
                         it, and sent someone's outgoing greeting to the extractor as though
                         it were a conversation.

Revision ID: b4e2a91c7d05
Revises: a7c31f9be204
"""

import sqlalchemy as sa
from alembic import op

revision = "b4e2a91c7d05"
down_revision = "a7c31f9be204"
branch_labels = None
depends_on = None

_PURPOSE = sa.Enum("SELF_USE", "INVESTMENT", name="purpose")


def upgrade() -> None:
    _PURPOSE.create(op.get_bind(), checkfirst=True)

    op.add_column("leads", sa.Column("purpose", _PURPOSE, nullable=True))
    op.add_column("leads", sa.Column("timeline_months", sa.Integer(), nullable=True))
    op.create_index("ix_leads_purpose", "leads", ["purpose"])
    op.create_index("ix_leads_timeline_months", "leads", ["timeline_months"])

    # Postgres cannot add an enum value inside a transaction that later uses it, and
    # Alembic's autogenerate does not detect enum changes at all — hence the raw statement.
    # IF NOT EXISTS keeps this re-runnable against a database that has been patched by hand.
    op.execute("ALTER TYPE callstatus ADD VALUE IF NOT EXISTS 'MACHINE'")


def downgrade() -> None:
    op.drop_index("ix_leads_timeline_months", table_name="leads")
    op.drop_index("ix_leads_purpose", table_name="leads")
    op.drop_column("leads", "timeline_months")
    op.drop_column("leads", "purpose")
    _PURPOSE.drop(op.get_bind(), checkfirst=True)

    # callstatus deliberately keeps MACHINE. Postgres cannot remove a value from an enum,
    # and rebuilding the type would mean rewriting every row of calls to decide what a
    # MACHINE call should become — a data question a schema downgrade must not answer on
    # its own. An unused value costs nothing.
