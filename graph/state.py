"""
TriageState — the single shared state dict that flows through the
LangGraph StateGraph and is cached in Redis during a live call.

Urgency scale (P1–P5):
  1 = Immediate / Life-threatening (ER/999)
  2 = Emergency (ED within 1 hour)
  3 = Urgent (same-day appointment)
  4 = Semi-urgent (within 48h)
  5 = Routine (next available GP slot)
"""
from typing import Annotated, Optional
import operator
from typing_extensions import TypedDict


class AppointmentDetails(TypedDict, total=False):
    doctor_name: str
    specialty: str
    department: str
    datetime: str          # ISO-8601 string
    appointment_id: str
    slot_id: str


class ICDMatch(TypedDict, total=False):
    icd_code: str
    condition_name: str
    symptom_keywords: list[str]
    urgency_hint: int      # 1-5 hint from the KB
    similarity_score: float


class TriageState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Call / session metadata
    # ------------------------------------------------------------------
    room_id: str               # LiveKit room name (unique per call)
    call_id: Optional[str]     # Postgres Call.id (UUID as str)
    started_at: str            # ISO-8601 timestamp

    # ------------------------------------------------------------------
    # Patient info — collected progressively by Intake Agent
    # ------------------------------------------------------------------
    patient_id: Optional[str]
    patient_name: Optional[str]
    patient_dob: Optional[str]    # "YYYY-MM-DD"
    patient_phone: Optional[str]
    patient_email: Optional[str]

    # ------------------------------------------------------------------
    # Transcript — append-only via LangGraph reducer
    # Each entry: "Patient: <text>" | "Agent: <text>"
    # ------------------------------------------------------------------
    transcript: Annotated[list[str], operator.add]

    # ------------------------------------------------------------------
    # Medical analysis — updated on every utterance
    # ------------------------------------------------------------------
    symptoms: list[str]           # extracted symptom keywords
    urgency_score: int            # 1–5; default 5 (assume routine)
    urgency_label: str            # "EMERGENCY" | "URGENT" | "ROUTINE"
    icd_matches: list[ICDMatch]   # top RAG results from Pinecone
    onset_trigger: Optional[str]  # what triggered the complaint (injury, activity, spontaneous)
    current_treatment: Optional[str]  # medications/remedies the patient has already tried
    known_allergies: Optional[str]    # patient-reported medication allergies

    # ------------------------------------------------------------------
    # Routing & booking
    # ------------------------------------------------------------------
    required_specialty: Optional[str]
    candidate_slots: list[dict]       # proposed slots before patient confirms
    appointment_id: Optional[str]
    appointment_details: Optional[AppointmentDetails]

    # ------------------------------------------------------------------
    # Confirmation & follow-up
    # ------------------------------------------------------------------
    confirmation_text: Optional[str]   # verbal message for agent to speak
    followup_result: Optional[dict]    # result from FollowupAgent

    # ------------------------------------------------------------------
    # Flow control flags
    # ------------------------------------------------------------------
    call_ended: bool
    followup_sent: bool
    path: str                     # "PENDING" | "EMERGENCY" | "ROUTINE"

    # ------------------------------------------------------------------
    # Error tracking
    # ------------------------------------------------------------------
    error: Optional[str]


def default_state(room_id: str) -> TriageState:
    """
    Return a fresh TriageState with safe defaults.
    Call this when a new LiveKit room is created.
    """
    from datetime import datetime, timezone
    import uuid
    return TriageState(
        room_id=room_id,
        call_id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc).isoformat(),
        patient_id=str(uuid.uuid4()),
        patient_name=None,
        patient_dob=None,
        patient_phone=None,
        patient_email=None,
        transcript=[],
        symptoms=[],
        urgency_score=5,           # optimistic default — assume routine
        urgency_label="ROUTINE",
        icd_matches=[],
        onset_trigger=None,
        current_treatment=None,
        known_allergies=None,
        required_specialty=None,
        candidate_slots=[],
        appointment_id=None,
        appointment_details=None,
        confirmation_text=None,
        followup_result=None,
        call_ended=False,
        followup_sent=False,
        path="PENDING",
        error=None,
    )
