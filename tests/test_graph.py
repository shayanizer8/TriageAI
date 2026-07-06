"""
Tests for the LangGraph state machine.
Verifies correct routing (EMERGENCY vs ROUTINE) and state transitions.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from graph.state import default_state, TriageState
from graph.supervisor import supervisor_node, route_after_supervisor, is_emergency


# ---------------------------------------------------------------------------
# Supervisor node tests
# ---------------------------------------------------------------------------
class TestSupervisorNode:
    def test_score_1_routes_emergency(self):
        state = default_state("test-room")
        state["urgency_score"] = 1
        result = supervisor_node(state)
        assert result["path"] == "EMERGENCY"
        assert result["urgency_label"] == "EMERGENCY"

    def test_score_2_routes_emergency(self):
        state = default_state("test-room")
        state["urgency_score"] = 2
        result = supervisor_node(state)
        assert result["path"] == "EMERGENCY"

    def test_score_3_routes_routine_urgent(self):
        state = default_state("test-room")
        state["urgency_score"] = 3
        result = supervisor_node(state)
        assert result["path"] == "ROUTINE"
        assert result["urgency_label"] == "URGENT"

    def test_score_4_routes_routine(self):
        state = default_state("test-room")
        state["urgency_score"] = 4
        result = supervisor_node(state)
        assert result["path"] == "ROUTINE"
        assert result["urgency_label"] == "ROUTINE"

    def test_score_5_routes_routine(self):
        state = default_state("test-room")
        state["urgency_score"] = 5
        result = supervisor_node(state)
        assert result["path"] == "ROUTINE"

    def test_is_emergency_true_for_low_score(self):
        state = default_state("test-room")
        state["urgency_score"] = 1
        assert is_emergency(state) is True

    def test_is_emergency_false_for_routine(self):
        state = default_state("test-room")
        state["urgency_score"] = 4
        assert is_emergency(state) is False


class TestConditionalEdge:
    def test_emergency_path(self):
        state = default_state("test-room")
        state["path"] = "EMERGENCY"
        assert route_after_supervisor(state) == "emergency"

    def test_routine_path(self):
        state = default_state("test-room")
        state["path"] = "ROUTINE"
        assert route_after_supervisor(state) == "router"

    def test_pending_defaults_to_router(self):
        state = default_state("test-room")
        state["path"] = "PENDING"
        assert route_after_supervisor(state) == "router"


# ---------------------------------------------------------------------------
# LangGraph integration test (mocked external calls)
# ---------------------------------------------------------------------------
class TestTriageGraph:
    @pytest.mark.asyncio
    async def test_routine_graph_execution(self):
        """Graph should complete without errors for a routine case."""
        state = default_state("test-room-graph")
        state["urgency_score"] = 4
        state["urgency_label"] = "ROUTINE"
        state["required_specialty"] = "General Practice"
        state["patient_id"] = "e89c3ab2-cd6e-4c7c-aa6c-34d678be32fb"
        state["call_id"] = "a89c3ab2-cd6e-4c7c-aa6c-34d678be32fa"

        with patch("agents.specialist_router.SpecialistRouter.route_and_book", new_callable=AsyncMock) as mock_route, \
             patch("agents.followup_agent.FollowupAgent.run", new_callable=AsyncMock) as mock_run:

            mock_route.return_value = {
                "doctor_name": "Dr. Anna Thompson",
                "specialty": "General Practice",
                "message": "Appointment confirmed."
            }
            # Also mock the state update normally done by tool call
            state["appointment_details"] = {
                "doctor_name": "Dr. Anna Thompson",
                "specialty": "General Practice",
                "department": "Outpatient Clinic",
                "datetime": "2026-07-01T10:00:00Z",
                "appointment_id": "test-appointment-id",
            }
            mock_run.return_value = {"sms_sent": True, "email_sent": True}

            from graph.triage_graph import triage_graph
            result = await triage_graph.ainvoke(state)

            # Supervisor should have run
            assert result.get("path") in ("ROUTINE", "EMERGENCY")

    @pytest.mark.asyncio
    async def test_emergency_graph_skips_router(self):
        """Emergency path should call emergency_node, not router_node."""
        state = default_state("test-room-emergency")
        state["urgency_score"] = 1
        state["urgency_label"] = "EMERGENCY"

        with patch("agents.specialist_router.SpecialistRouter.route_and_book", new_callable=AsyncMock) as mock_route, \
             patch("agents.followup_agent.FollowupAgent.run", new_callable=AsyncMock) as mock_run:

            mock_run.return_value = {"sms_sent": True, "email_sent": True}

            from graph.triage_graph import triage_graph
            result = await triage_graph.ainvoke(state)

            # Router should NOT have been called on emergency path
            mock_route.assert_not_called()
            assert result.get("path") == "EMERGENCY"
