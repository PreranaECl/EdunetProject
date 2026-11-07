"""
schemas.py — Pydantic models for Emergency Assist AI.

Defines validated data structures exchanged between agents and services.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Emergency Assessment Agent output schema
# ---------------------------------------------------------------------------

class EmergencyAssessment(BaseModel):
    """Structured output from the Emergency Assessment Agent."""

    agent: str = Field(default="Emergency Assessment Agent")
    emergency_category: str
    severity_level: Literal["Critical", "High", "Moderate", "Low", "Unknown"]
    emergency_summary: str
    observed_indicators: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    recommended_service: List[str] = Field(default_factory=list)
    immediate_emergency_contact_required: bool = False
    image_quality: Optional[str] = None          # "acceptable" | "poor" | "not_provided"
    information_gaps: List[str] = Field(default_factory=list)
    requires_human_assistance: bool = True
    status: str = Field(default="preliminary_assessment")

    @field_validator("severity_level", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        mapping = {
            "critical": "Critical",
            "high": "High",
            "moderate": "Moderate",
            "low": "Low",
            "unknown": "Unknown",
        }
        return mapping.get(str(v).lower(), "Unknown")


# ---------------------------------------------------------------------------
# Emergency Response Agent output schema
# ---------------------------------------------------------------------------

class EmergencyContact(BaseModel):
    country: str = "India"
    number: str = "112"


class EmergencyResponse(BaseModel):
    """Structured output from the Emergency Response Agent."""

    agent: str = Field(default="Emergency Response Agent")
    emergency_summary: str
    severity_level: Literal["Critical", "High", "Moderate", "Low", "Unknown"]
    immediate_actions: List[str] = Field(default_factory=list)
    recommended_services: List[str] = Field(default_factory=list)
    emergency_contact: EmergencyContact = Field(default_factory=EmergencyContact)
    safety_warnings: List[str] = Field(default_factory=list)
    human_assistance_required: bool = True
    disclaimer: str = Field(
        default=(
            "AI-generated preliminary guidance. Follow official emergency responder "
            "instructions. Not a substitute for professional emergency services."
        )
    )

    @field_validator("severity_level", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        mapping = {
            "critical": "Critical",
            "high": "High",
            "moderate": "Moderate",
            "low": "Low",
            "unknown": "Unknown",
        }
        return mapping.get(str(v).lower(), "Unknown")


# ---------------------------------------------------------------------------
# Help Center (Google Maps Places result) schema
# ---------------------------------------------------------------------------

class HelpCenter(BaseModel):
    """A single nearby emergency help center returned by Google Maps."""

    name: str
    category: str
    address: str = ""
    distance_km: Optional[float] = None
    phone: Optional[str] = None
    place_id: Optional[str] = None
    maps_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class HelpCenterSearchResult(BaseModel):
    """Wrapper for all help center search results."""

    search_location: dict                        # {"latitude": ..., "longitude": ...}
    help_centers: List[HelpCenter] = Field(default_factory=list)
    data_source: str = "Google Places API"
    status: str = "success"
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Performance metrics schema
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agent state machine
# ---------------------------------------------------------------------------

from enum import Enum

class AgentState(str, Enum):
    """Explicit state machine for each agent execution."""
    IDLE      = "idle"
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    BLOCKED   = "blocked"
    RETRYING  = "retrying"


class AgentExecutionRecord(BaseModel):
    """Single entry in the agent execution log."""

    component: str
    state: AgentState = AgentState.IDLE
    status: str = "idle"                         # human-readable mirror of state
    execution_time_s: float = 0.0
    validation: str = "N/A"                      # "Passed" | "Failed" | "N/A"
    notes: str = ""
    # Structured token fields (no more string parsing)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LatencyPoint(BaseModel):
    """A single latency data point for trending charts."""
    timestamp: str       # ISO format
    assessment_s: float = 0.0
    response_s: float = 0.0
    maps_s: float = 0.0
    total_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    emergency_type: str = ""


class SessionMetrics(BaseModel):
    """Aggregate performance metrics for the current Streamlit session."""

    total_requests: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    total_response_time_s: float = 0.0
    assessment_latency_s: float = 0.0
    response_latency_s: float = 0.0
    maps_latency_s: float = 0.0
    guardrail_interventions: int = 0
    gemini_api_calls: int = 0
    maps_api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    # Rate limiting
    requests_this_minute: int = 0
    last_request_time: float = 0.0
    # Per-workflow history for trending
    latency_history: List[LatencyPoint] = Field(default_factory=list)
    execution_log: List[AgentExecutionRecord] = Field(default_factory=list)
