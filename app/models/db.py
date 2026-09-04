from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey, Index, Integer, Numeric, String, Text,
    UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid
import enum
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.utils.timeutils import utc_now

class LeadStatus(str, enum.Enum):
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"

class Purpose(str, enum.Enum):
    """Why they are buying.

    Asked on nearly every call — "for yourself, or for investment?" — answered plainly, and
    until now thrown away, because there was nowhere to put it. It decides which project a
    colleague pitches next and how they pitch it.
    """

    SELF_USE = "SELF_USE"
    INVESTMENT = "INVESTMENT"

class CampaignStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

class CallStatus(str, enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    # An answering machine picked up. Distinct from COMPLETED, which a voicemail was being
    # recorded as: it inflated the connect rate on the dashboard, and it sent a recording of
    # someone else's outgoing message to the extractor as though it were a conversation.
    # Distinct from FAILED too — nothing went wrong, the person simply was not there.
    MACHINE = "MACHINE"

class ContactStatus(str, enum.Enum):
    """Where a number is in the dialling queue.

    The contacts table IS the queue. Before it existed, numbers arrived in the body of a dial
    request, were copied into Redis so the answer webhook could find them, and then existed
    only as a phone_number on a Call row. Nothing could answer "who is left to call", nothing
    could retry a number that did not answer, and re-uploading a list dialled everyone again.
    """

    # Waiting to be dialled. next_attempt_at decides when it becomes eligible.
    PENDING = "PENDING"
    # A slot has been taken and Vobiz has been asked to dial. Distinct from PENDING so the
    # pump cannot hand the same number to two workers, and distinct from IN_PROGRESS on the
    # Call because this is set before the carrier has connected anything.
    DIALING = "DIALING"
    # The conversation happened. Whether it produced a good lead is the Lead's business.
    COMPLETED = "COMPLETED"
    # Rang out, or a machine picked up. Eligible for retry until the attempts run out.
    NO_ANSWER = "NO_ANSWER"
    # The dial itself failed, or the pipeline broke. Also retried: the cause is usually ours.
    FAILED = "FAILED"
    # Retries are spent. A terminal state that is deliberately not FAILED — the difference
    # between "we could not reach them" and "we stopped trying" is what an operator needs.
    EXHAUSTED = "EXHAUSTED"
    # They asked not to be called, or the number is on the suppression list. Never dialled
    # again by anything, and never counted as a failure.
    DND = "DND"
    # The spreadsheet cell was not a phone number we can dial. Kept rather than dropped so
    # the operator can see which rows did not make it and fix the source.
    INVALID = "INVALID"
    # Removed from the run by an operator without being a DND. Not dialled, not retried.
    SKIPPED = "SKIPPED"


# Statuses the pump will pick up again once next_attempt_at has passed.
RETRIABLE_CONTACT_STATUSES = (ContactStatus.NO_ANSWER, ContactStatus.FAILED)

# Statuses nothing will ever dial again.
TERMINAL_CONTACT_STATUSES = (
    ContactStatus.COMPLETED,
    ContactStatus.EXHAUSTED,
    ContactStatus.DND,
    ContactStatus.INVALID,
    ContactStatus.SKIPPED,
)


class DashboardRole(str, enum.Enum):
    # Sales works leads and reads calls. Only an admin may spend money by dialing or
    # change a campaign's state, so the destructive surface has a named holder.
    VIEWER = "VIEWER"
    ADMIN = "ADMIN"


class DashboardUser(Base):
    """A person who signs in to the dashboard.

    Separate from API_KEY on purpose: that key is a bearer secret that dials, and handing
    it to a browser would put a spend-capable credential in devtools, in the URL bar's
    history, and in every extension on the page.
    """

    __tablename__ = "dashboard_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    # scrypt digest, never the password. Format is documented in app.core.passwords.
    password_hash = Column(String, nullable=False)
    role = Column(
        Enum(DashboardRole),
        default=DashboardRole.VIEWER,
        server_default=text("'VIEWER'"),
        nullable=False,
    )
    # Deactivating beats deleting: leads and campaigns are not owned by a user, but an
    # audit trail that references a vanished row reads as corruption.
    is_active = Column(Boolean, default=True, server_default=text("true"), nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime,
        default=utc_now,
        server_default=text("(now() at time zone 'utc')"),
        nullable=False,
    )


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(String, index=True, nullable=True) # Deprecated, keeping for backwards compat
    name = Column(String, nullable=False)
    # Who the agent says it is calling from. Separate from name because the agent used to
    # introduce itself with the project — "I am Priya calling you from Abhee Codename New
    # Dimension" — and a real person calls from the company, naming the project when they
    # describe it. Nullable: without one the greeting falls back to name, as before.
    developer_name = Column(String, nullable=True)
    city = Column(String, nullable=False)
    locality = Column(String, nullable=False)
    min_price = Column(Numeric)
    max_price = Column(Numeric)
    config_json = Column(JSONB)
    amenities = Column(ARRAY(String))
    nearby_facilities = Column(JSONB)
    possession_status = Column(String)
    usps = Column(ARRAY(String))
    rera_id = Column(String)
    embedding = Column(Vector(1536))
    
    campaigns = relationship("Campaign", back_populates="project")

class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.ACTIVE)
    start_date = Column(DateTime, default=utc_now)
    end_date = Column(DateTime)
    total_leads_dialed = Column(Numeric, default=0)
    
    project = relationship("Project", back_populates="campaigns")
    calls = relationship("Call", back_populates="campaign")
    contacts = relationship(
        "Contact", back_populates="campaign", cascade="all, delete-orphan", passive_deletes=True
    )

class Lead(Base):
    __tablename__ = "leads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # The number we dialled, carried over from the Call. Without it a booked site visit
    # cannot be confirmed, a lead cannot be de-duplicated, and DND cannot be honoured.
    phone_number = Column(String, index=True)
    customer_name = Column(String)
    preferred_location = Column(String)
    # Which configuration the prospect wants, normalised against the project's config_json
    # so "2bhk"/"2 BHK"/"two bedroom" all land on the same value a rep can filter by.
    preferred_unit_type = Column(String)
    budget = Column(Numeric(14, 2))
    # Asked on nearly every call and, until now, discarded — there was nowhere to put it.
    # It decides which project a colleague pitches next and how.
    purpose = Column(Enum(Purpose), index=True)
    # The prospect's own words, kept because a rep reads them before dialling.
    timeline = Column(String)
    # ...and the same thing as a number, because 'Maybe around in 2 months.' cannot be
    # sorted, filtered or used to decide who to call first.
    timeline_months = Column(Integer, index=True)
    callback_time = Column(DateTime)
    site_visit_time = Column(DateTime)
    # A null status hides the lead from every filter sales works from, so the column carries
    # the floor rather than trusting each writer to supply one.
    status = Column(
        Enum(LeadStatus),
        default=LeadStatus.WARM,
        server_default=text("'WARM'"),
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime,
        default=utc_now,
        server_default=text("(now() at time zone 'utc')"),
        nullable=False,
        index=True,
    )

    calls = relationship("Call", back_populates="lead")

class Call(Base):
    __tablename__ = "calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), index=True, nullable=False)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=True)
    # Which queue entry produced this call. Nullable: calls placed by hand, and every call
    # made before the queue existed, have no contact behind them.
    contact_id = Column(
        UUID(as_uuid=True), ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    call_sid = Column(String, unique=True, index=True, nullable=False)
    # E.164 number actually dialled. Recorded here so a call can be traced to a person even
    # when extraction produces no lead.
    phone_number = Column(String, index=True)
    # Indexed for the dashboard's call list, which filters on status and sorts on time.
    status = Column(Enum(CallStatus), default=CallStatus.IN_PROGRESS, index=True)
    started_at = Column(DateTime, default=utc_now, index=True)
    ended_at = Column(DateTime, nullable=True)
    # Bounded precision: an unbounded NUMERIC stored 54.51915999999999939973349682986736
    duration_seconds = Column(Numeric(10, 2), nullable=True)
    
    campaign = relationship("Campaign", back_populates="calls")
    lead = relationship("Lead", back_populates="calls")
    transcript = relationship("Transcript", back_populates="call", uselist=False)

class Contact(Base):
    """One number to dial on one campaign, and everything the queue needs to know about it.

    Rows are created by a spreadsheet import and consumed by the dial pump in
    app/worker.py, which takes only as many as the carrier has free slots for. That is the
    whole reason this table exists: dialling used to fire every number in the request at
    once, and the concurrency cap was checked after the money had already been spent.
    """

    __tablename__ = "contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(
        UUID(as_uuid=True), ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False
    )
    # E.164, normalised on import. Indexed on its own as well as in the composite below,
    # because the suppression check and the "has this person been called before" question
    # both look across campaigns.
    phone_number = Column(String, nullable=False, index=True)
    # From the spreadsheet. Optional: without it the agent asks for the name on the call.
    name = Column(String)

    status = Column(
        Enum(ContactStatus),
        default=ContactStatus.PENDING,
        server_default=text("'PENDING'"),
        nullable=False,
    )
    # Dials placed, not conversations had. Compared against MAX_DIAL_ATTEMPTS.
    attempts = Column(Integer, default=0, server_default=text("0"), nullable=False)
    last_attempt_at = Column(DateTime)
    # When this becomes eligible again. Null means "as soon as the pump gets to it", which is
    # what a fresh import wants; a retry sets it forward.
    next_attempt_at = Column(DateTime)
    # Why the last attempt ended the way it did, in words an operator can act on.
    last_outcome = Column(String)

    # Which spreadsheet row this came from. Kept so a rejected row can be found and fixed in
    # the source file rather than hunted for by phone number.
    source_row = Column(Integer)
    # Groups everything from one upload, so a mistaken import can be found and undone.
    import_batch_id = Column(UUID(as_uuid=True), index=True)

    created_at = Column(
        DateTime, default=utc_now, server_default=text("(now() at time zone 'utc')"),
        nullable=False,
    )
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    campaign = relationship("Campaign", back_populates="contacts")

    __table_args__ = (
        # One number appears once per campaign. Re-uploading the same sheet must not dial
        # everybody a second time, and this is what makes the import idempotent.
        UniqueConstraint("campaign_id", "phone_number", name="uq_contacts_campaign_phone"),
        # The pump's query: eligible contacts for a campaign, soonest first. Composite
        # because it filters on both and orders on the third; three separate indexes would
        # leave the sort to a heap scan on a table that grows with every lead list.
        Index("ix_contacts_pump", "campaign_id", "status", "next_attempt_at"),
    )


class Suppression(Base):
    """A number that must never be dialled again, by any campaign.

    Separate from ContactStatus.DND because it has to outlive the campaign that learned it.
    Someone who says "don't call me" has told the company, not one lead list, and the next
    project's import must not undo that. It is also the only record that would answer a
    regulator asking whether the request was honoured.
    """

    __tablename__ = "suppressions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Unique, so adding the same number twice is a no-op rather than a duplicate to reconcile.
    phone_number = Column(String, nullable=False, unique=True, index=True)
    # Free text: "asked not to be called", "wrong number", "employee". Read by a human who is
    # deciding whether an entry was a mistake.
    reason = Column(String)
    # Who added it. An empty value means the agent recorded it from the call itself.
    added_by = Column(String)
    created_at = Column(
        DateTime, default=utc_now, server_default=text("(now() at time zone 'utc')"),
        nullable=False, index=True,
    )


class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), unique=True, nullable=False)
    full_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    
    call = relationship("Call", back_populates="transcript")
