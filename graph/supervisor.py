"""
Supervisor node — rule-based, zero LLM calls.

Watches urgency_score in TriageState and deterministically routes
to the emergency or routine path.

Design principle: life-or-death routing must NOT go through an LLM.
A threshold check is the only safe implementation.
"""
from graph.state import TriageState
from config.settings import get_settings

settings = get_settings()


def supervisor_node(state: TriageState) -> dict:
    """
    LangGraph node — determines call path from urgency score.

    Returns only the keys that change; LangGraph merges with existing state.
    """
    score: int = state.get("urgency_score", 5)
    threshold: int = settings.emergency_threshold  # default 2

    if score <= threshold:
        return {
            "path": "EMERGENCY",
            "urgency_label": "EMERGENCY",
        }
    elif score == 3:
        return {
            "path": "ROUTINE",
            "urgency_label": "URGENT",
        }
    else:
        return {
            "path": "ROUTINE",
            "urgency_label": "ROUTINE",
        }


# ---------------------------------------------------------------------------
# Conditional edge function used in LangGraph routing
# ---------------------------------------------------------------------------

def route_after_supervisor(state: TriageState) -> str:
    """
    LangGraph conditional edge.
    Returns the name of the next node to execute after supervisor_node.
    """
    if state.get("path") == "EMERGENCY":
        return "emergency"
    return "router"


# ---------------------------------------------------------------------------
# Inline urgency check — used by background asyncio supervisor task
# (not LangGraph) to interrupt the live call before phase transitions.
# ---------------------------------------------------------------------------

def is_emergency(state: TriageState) -> bool:
    """True if the current urgency score qualifies as an emergency."""
    return state.get("urgency_score", 5) <= settings.emergency_threshold
