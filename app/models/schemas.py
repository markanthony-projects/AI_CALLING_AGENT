from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from app.models.db import LeadStatus

class LeadExtraction(BaseModel):
    is_prospect: bool = Field(default=False, description="True if the caller is a genuine prospect interested in real estate. False if spam, wrong number, or completely uninterested.")
    customer_name: Optional[str] = Field(None, description="Extracted customer name")
    preferred_location: Optional[str] = Field(None, description="Preferred locality or city")
    budget: Optional[float] = Field(None, description="Customer budget numerically")
    timeline: Optional[str] = Field(None, description="Purchase timeline")
    callback_time: Optional[datetime] = Field(None, description="Scheduled callback date and time")
    site_visit_time: Optional[datetime] = Field(None, description="Scheduled site visit date and time")
    status: Optional[LeadStatus] = Field(None, description="Lead status: HOT, WARM, or COLD")

class ProjectFilterParams(BaseModel):
    city: Optional[str] = None
    locality: Optional[str] = None
    min_budget: Optional[float] = None
    max_budget: Optional[float] = None
