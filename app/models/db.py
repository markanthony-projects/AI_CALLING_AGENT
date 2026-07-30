from sqlalchemy import Column, String, Text, DateTime, Numeric, Enum, ForeignKey, text
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

class CampaignStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

class CallStatus(str, enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(String, index=True, nullable=True) # Deprecated, keeping for backwards compat
    name = Column(String, nullable=False)
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
    timeline = Column(String)
    callback_time = Column(DateTime)
    site_visit_time = Column(DateTime)
    # A null status hides the lead from every filter sales works from, so the column carries
    # the floor rather than trusting each writer to supply one.
    status = Column(
        Enum(LeadStatus),
        default=LeadStatus.WARM,
        server_default=text("'WARM'"),
        nullable=False,
    )
    created_at = Column(
        DateTime,
        default=utc_now,
        server_default=text("(now() at time zone 'utc')"),
        nullable=False,
    )

    calls = relationship("Call", back_populates="lead")

class Call(Base):
    __tablename__ = "calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), index=True, nullable=False)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=True)
    call_sid = Column(String, unique=True, index=True, nullable=False)
    # E.164 number actually dialled. Recorded here so a call can be traced to a person even
    # when extraction produces no lead.
    phone_number = Column(String, index=True)
    status = Column(Enum(CallStatus), default=CallStatus.IN_PROGRESS)
    started_at = Column(DateTime, default=utc_now)
    ended_at = Column(DateTime, nullable=True)
    # Bounded precision: an unbounded NUMERIC stored 54.51915999999999939973349682986736
    duration_seconds = Column(Numeric(10, 2), nullable=True)
    
    campaign = relationship("Campaign", back_populates="calls")
    lead = relationship("Lead", back_populates="calls")
    transcript = relationship("Transcript", back_populates="call", uselist=False)

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), unique=True, nullable=False)
    full_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    
    call = relationship("Call", back_populates="transcript")
