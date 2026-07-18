from sqlalchemy import Column, String, Text, DateTime, Numeric, Enum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
import enum
from pgvector.sqlalchemy import Vector
from app.core.database import Base

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
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    total_leads_dialed = Column(Numeric, default=0)
    
    project = relationship("Project", back_populates="campaigns")
    calls = relationship("Call", back_populates="campaign")

class Lead(Base):
    __tablename__ = "leads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_name = Column(String)
    preferred_location = Column(String)
    budget = Column(Numeric)
    timeline = Column(String)
    callback_time = Column(DateTime)
    site_visit_time = Column(DateTime)
    status = Column(Enum(LeadStatus))
    
    calls = relationship("Call", back_populates="lead")

class Call(Base):
    __tablename__ = "calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    lead_id = Column(UUID(as_uuid=True), ForeignKey("leads.id"), nullable=True)
    call_sid = Column(String, unique=True, index=True, nullable=False)
    status = Column(Enum(CallStatus), default=CallStatus.IN_PROGRESS)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Numeric, nullable=True)
    
    campaign = relationship("Campaign", back_populates="calls")
    lead = relationship("Lead", back_populates="calls")
    transcript = relationship("Transcript", back_populates="call", uselist=False)

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), unique=True, nullable=False)
    full_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    call = relationship("Call", back_populates="transcript")
