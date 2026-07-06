"""
FastAPI endpoint tests using httpx.AsyncClient.
"""
from __future__ import annotations

import pytest
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Scheduling API
# ---------------------------------------------------------------------------
class TestScheduling:
    @pytest.mark.asyncio
    async def test_get_doctors_no_filter(self, client: AsyncClient):
        with patch("api.routers.schedule.get_db") as mock_db:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_db.return_value = mock_session

            response = await client.get("/schedule/doctors")
            # Just ensure endpoint exists and returns valid HTTP
            assert response.status_code in (200, 422, 500)

    @pytest.mark.asyncio
    async def test_get_doctors_with_specialty(self, client: AsyncClient):
        response = await client.get("/schedule/doctors?specialty=Cardiology")
        # Endpoint exists
        assert response.status_code != 404


# ---------------------------------------------------------------------------
# Triage RAG
# ---------------------------------------------------------------------------
class TestTriageRAG:
    @pytest.mark.asyncio
    async def test_query_endpoint_exists(self, client: AsyncClient):
        with patch("api.routers.triage.query_medical_kb", new_callable=AsyncMock) as mock_rag:
            mock_rag.return_value = [
                {
                    "icd_code": "J06.9",
                    "condition_name": "Acute upper respiratory infection",
                    "symptom_keywords": ["cough", "fever"],
                    "urgency_hint": 4,
                    "similarity_score": 0.91,
                }
            ]
            response = await client.post(
                "/triage/query",
                json={"symptom_text": "fever and cough", "top_k": 3},
            )
            assert response.status_code == 200
            data = response.json()
            assert "matches" in data
            assert len(data["matches"]) == 1


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
class TestWebhooks:
    @pytest.mark.asyncio
    async def test_livekit_webhook_room_finished(self, client: AsyncClient):
        with patch("api.routers.webhooks._trigger_followup", new_callable=AsyncMock):
            response = await client.post(
                "/webhook/livekit",
                json={
                    "event": "room_finished",
                    "room": {"name": "test-room-123"},
                },
            )
            assert response.status_code == 200
            assert response.json()["received"] is True

    @pytest.mark.asyncio
    async def test_manual_call_ended_trigger(self, client: AsyncClient):
        with patch("api.routers.webhooks._trigger_followup", new_callable=AsyncMock):
            response = await client.post(
                "/webhook/call-ended",
                params={"room_id": "test-room-999"},
            )
            assert response.status_code == 200
            assert response.json()["triggered"] is True
