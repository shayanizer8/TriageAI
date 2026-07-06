"""
Pydantic request/response models for the FastAPI API layer.
"""
from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Scheduling API
# ---------------------------------------------------------------------------

class DoctorResponse(BaseModel):
    id: uuid.UUID
    name: str
    specialty: str
    department: str

    model_config = {"from_attributes": True}


class SlotResponse(BaseModel):
    id: uuid.UUID
    doctor_id: uuid.UUID
    datetime: datetime
    is_booked: bool

    model_config = {"from_attributes": True}


class BookAppointmentRequest(BaseModel):
    slot_id: uuid.UUID
    patient_id: uuid.UUID
    call_id: Optional[uuid.UUID] = None


class AppointmentResponse(BaseModel):
    id: uuid.UUID
    patient_id: uuid.UUID
    slot_id: uuid.UUID
    call_id: Optional[uuid.UUID]
    created_at: datetime
    doctor: DoctorResponse
    slot: SlotResponse

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Triage RAG
# ---------------------------------------------------------------------------

class TriageQueryRequest(BaseModel):
    symptom_text: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class ICDMatchResponse(BaseModel):
    icd_code: str
    condition_name: str
    symptom_keywords: list[str]
    urgency_hint: int
    similarity_score: float


class TriageQueryResponse(BaseModel):
    query: str
    matches: list[ICDMatchResponse]


# ---------------------------------------------------------------------------
# Webhook payloads (LiveKit)
# ---------------------------------------------------------------------------

class LiveKitWebhookEvent(BaseModel):
    """Simplified LiveKit webhook payload."""
    event: str
    room: Optional[dict] = None
    participant: Optional[dict] = None
    created_at: Optional[int] = None


class CallEndedPayload(BaseModel):
    room_id: str
    call_id: Optional[str] = None
    ended_at: Optional[datetime] = None
