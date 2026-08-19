"""Give dialling a queue, and give "don't call me" somewhere to live.

Numbers used to arrive in the body of a dial request, get copied into Redis so the answer
webhook could find them, and then exist only as a phone_number on a Call row. Three things
followed from that:

  the queue          Nothing could answer "who is left to call". Re-uploading a lead list
                     dialled everyone on it a second time, and a number that rang out was
                     simply lost.

  the pacing         Worse, the dial endpoint fired every number in the request at once and
                     the concurrency cap was checked when the media websocket opened — after
                     Vobiz had dialled, billed us, and rung a real person. Twenty numbers on
                     a three-slot account meant seventeen people were called, charged for,
                     and hung up on, and because the check returned before the Call row was
                     written there was no record it had happened. contacts is what the pump
                     draws from, so a dial is placed only when a slot is free.

  the suppression    ContactStatus.DND cannot outlive the campaign that learned it. Someone
                     who asks not to be called has told the company, not one lead list, and
                     the next project's import would have undone it.

calls.contact_id closes the loop so an outcome can be written back to the queue entry rather
than matched on a phone number that may appear in several campaigns.

Revision ID: c9a3f21e7b40
Revises: b4e2a91c7d05
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c9a3f21e7b40"
down_revision = "b4e2a91c7d05"
branch_labels = None
depends_on = None

CONTACT_STATUS = (
    "PENDING",
    "DIALING",
    "COMPLETED",
    "NO_ANSWER",
    "FAILED",
    "EXHAUSTED",
    "DND",
    "INVALID",
    "SKIPPED",
)


def upgrade() -> None:
    contact_status = postgresql.ENUM(*CONTACT_STATUS, name="contactstatus")
    # create_type=False on the columns below, so the type is created exactly once here.
    contact_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "contacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("campaign_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("phone_number", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(*CONTACT_STATUS, name="contactstatus", create_type=False),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_attempt_at", sa.DateTime(), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=True),
        sa.Column("last_outcome", sa.String(), nullable=True),
        sa.Column("source_row", sa.Integer(), nullable=True),
        sa.Column("import_batch_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(now() at time zone 'utc')"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        # Deleting a campaign takes its queue with it. Orphan rows would still be visible to
        # the pump's campaign join and would never be dialled or cleaned up.
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        # What makes a re-import idempotent rather than a second round of calls.
        sa.UniqueConstraint("campaign_id", "phone_number", name="uq_contacts_campaign_phone"),
    )
    op.create_index("ix_contacts_phone_number", "contacts", ["phone_number"])
    op.create_index("ix_contacts_import_batch_id", "contacts", ["import_batch_id"])
    # The pump's exact query: eligible rows for one campaign, soonest first.
    op.create_index("ix_contacts_pump", "contacts", ["campaign_id", "status", "next_attempt_at"])

    op.create_table(
        "suppressions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("phone_number", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column("added_by", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(now() at time zone 'utc')"),
            nullable=False,
        ),
        # Unique so adding the same number twice is a no-op, not a duplicate to reconcile.
        sa.UniqueConstraint("phone_number", name="uq_suppressions_phone"),
    )
    op.create_index("ix_suppressions_created_at", "suppressions", ["created_at"])

    # SET NULL rather than CASCADE: a contact may be deleted, and losing the call record —
    # with its transcript and its lead — because its queue entry was tidied up would be a
    # far worse trade than a call with no contact behind it.
    op.add_column(
        "calls", sa.Column("contact_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_index("ix_calls_contact_id", "calls", ["contact_id"])
    op.create_foreign_key(
        "fk_calls_contact_id", "calls", "contacts", ["contact_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    op.drop_constraint("fk_calls_contact_id", "calls", type_="foreignkey")
    op.drop_index("ix_calls_contact_id", table_name="calls")
    op.drop_column("calls", "contact_id")

    op.drop_index("ix_suppressions_created_at", table_name="suppressions")
    op.drop_table("suppressions")

    op.drop_index("ix_contacts_pump", table_name="contacts")
    op.drop_index("ix_contacts_import_batch_id", table_name="contacts")
    op.drop_index("ix_contacts_phone_number", table_name="contacts")
    op.drop_table("contacts")

    postgresql.ENUM(name="contactstatus").drop(op.get_bind(), checkfirst=True)
