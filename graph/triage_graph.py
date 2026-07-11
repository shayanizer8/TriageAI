"""
LangGraph StateGraph — post-intake routing and confirmation phase.

The graph is invoked AFTER the Intake Agent finishes the voice conversation.
By that point:
  - TriageState has urgency_score, symptoms, icd_matches (from parallel Symptom Analyzer)
  - patient_id is set

The graph then handles:
  supervisor_node → route_after_supervisor → [emergency | router] → confirmation → followup
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from graph.state import TriageState
from graph.supervisor import supervisor_node, route_after_supervisor
from agents.specialist_router import SpecialistRouter
from agents.followup_agent import FollowupAgent
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Node: Emergency
# ---------------------------------------------------------------------------
async def emergency_node(state: TriageState) -> dict:
    """
    Handles P1/P2 emergency path.
    In production: would trigger a nurse alert, SIP transfer, or 911 API call.
    For demo: logs the emergency and sets a descriptive appointment_details.
    """
    logger.critical(
        "EMERGENCY PATH | room=%s score=%d symptoms=%s",
        state.get("room_id"),
        state.get("urgency_score"),
        state.get("symptoms"),
    )
    return {
        "path": "EMERGENCY",
        "appointment_details": {
            "doctor_name": "Emergency Duty Physician",
            "specialty": "Emergency Medicine",
            "department": "Emergency Department",
            "datetime": "Immediately",
            "appointment_id": "EMERGENCY",
        },
    }


# ---------------------------------------------------------------------------
# Node: Specialist Router
# ---------------------------------------------------------------------------
async def router_node(state: TriageState) -> dict:
    """
    Invokes the SpecialistRouter to find candidate appointment slots.
    Does NOT book — the entrypoint handles interactive confirmation with the patient.
    """
    router = SpecialistRouter(state)
    candidates = await router.find_candidate_slots(max_slots=2)

    if candidates:
        return {
            "candidate_slots": candidates,
            "path": "ROUTINE",
        }
    else:
        logger.error("Router found no slots — falling back for room: %s", state.get("room_id"))
        return {
            "candidate_slots": [],
            "appointment_details": {
                "doctor_name": "On-call GP",
                "specialty": state.get("required_specialty", "General Practice"),
                "department": "Outpatient Clinic",
                "datetime": "Next available slot — you will be contacted",
                "appointment_id": "PENDING",
            },
        }


def format_spoken_datetime(dt_str: str) -> str:
    """Format ISO 8601 datetime strings into friendly, spoken-word format."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        day = dt.day
        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        time_str = dt.strftime("%I:%M %p").lstrip("0")
        return f"{dt.strftime('%A, %B')} {day}{suffix} at {time_str}"
    except Exception:
        return dt_str


# ---------------------------------------------------------------------------
# Node: Confirmation
# ---------------------------------------------------------------------------
async def confirmation_node(state: TriageState) -> dict:
    """
    Prepares the verbal confirmation text for the Intake Agent to speak.
    (The actual speech is handled in entrypoint.py after graph completion.)
    """
    appointment = state.get("appointment_details") or {}
    doctor = appointment.get("doctor_name", "your specialist")
    specialty = appointment.get("specialty", "")
    when = format_spoken_datetime(appointment.get("datetime", "shortly"))
    path = state.get("path", "ROUTINE")

    if path == "EMERGENCY":
        confirmation_text = (
            "Based on your symptoms, I am flagging this as an emergency. "
            "Please go to the Emergency Department immediately. "
        )
    else:
        confirmation_text = (
            f"I have booked you an appointment with {doctor} "
            f"{'in ' + specialty + ' ' if specialty else ''}for {when}. "
            "You will receive a text message and email shortly with all the details. "
            "Is there anything else you need before I let you go?"
        )

    logger.info("Confirmation prepared for room: %s", state.get("room_id"))
    return {"confirmation_text": confirmation_text}


# ---------------------------------------------------------------------------
# Node: Follow-up
# ---------------------------------------------------------------------------
async def followup_node(state: TriageState) -> dict:
    """
    Triggers the FollowupAgent to send SMS + email post-call.
    """
    agent = FollowupAgent(state)
    result = await agent.run()
    return {
        "followup_sent": True,
        "followup_result": result,
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_triage_graph():
    """
    Build and compile the LangGraph StateGraph for post-intake routing.

    Graph structure:
      START → supervisor → [emergency | router] → confirmation → END
    """
    graph = StateGraph(TriageState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("router", router_node)
    graph.add_node("confirmation", confirmation_node)

    # Edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "emergency": "emergency",
            "router": "router",
        },
    )
    graph.add_edge("emergency", "confirmation")
    graph.add_edge("router", "confirmation")
    graph.add_edge("confirmation", END)

    return graph.compile()


# Singleton — compiled once at import time
triage_graph = build_triage_graph()
