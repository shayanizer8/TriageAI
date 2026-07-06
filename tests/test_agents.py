"""
Unit tests for the four agents.
Uses pytest-mock to mock external API calls (Cerebras AsyncOpenAI, Pinecone, LiveKit).
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.state import default_state


# ---------------------------------------------------------------------------
# Symptom Analyzer tests
# ---------------------------------------------------------------------------
class TestSymptomAnalyzer:
    @pytest.mark.asyncio
    async def test_emergency_chest_pain(self):
        """Chest pain should yield urgency_score <= 2."""
        state = default_state("test-room-001")
        state["transcript"] = ["Patient: I have severe chest pain and can't breathe"]

        with patch("agents.symptom_analyzer.query_medical_kb", new_callable=AsyncMock) as mock_rag, \
             patch("agents.symptom_analyzer.AsyncOpenAI") as mock_openai:

            mock_rag.return_value = [
                {
                    "icd_code": "I21.9",
                    "condition_name": "Acute myocardial infarction",
                    "symptom_keywords": ["chest pain", "dyspnea"],
                    "urgency_hint": 1,
                    "similarity_score": 0.96,
                }
            ]

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "urgency_score": 1,
                    "symptoms": ["chest pain", "shortness of breath"],
                    "required_specialty": "Emergency Medicine",
                    "reasoning": "Acute MI suspected",
                    "icd_codes_used": ["I21.9"],
                })))
            ]
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            from agents.symptom_analyzer import SymptomAnalyzer
            analyzer = SymptomAnalyzer(state)
            await analyzer.analyze_utterance("I have severe chest pain and can't breathe")

            assert state["urgency_score"] <= 2
            assert state["urgency_label"] == "EMERGENCY"
            assert "chest pain" in state["symptoms"]

    @pytest.mark.asyncio
    async def test_routine_cold_symptoms(self):
        """Common cold symptoms should yield urgency_score >= 4."""
        state = default_state("test-room-002")
        state["transcript"] = ["Patient: I have a runny nose and sore throat for 2 days"]

        with patch("agents.symptom_analyzer.query_medical_kb", new_callable=AsyncMock) as mock_rag, \
             patch("agents.symptom_analyzer.AsyncOpenAI") as mock_openai:

            mock_rag.return_value = [
                {
                    "icd_code": "J06.9",
                    "condition_name": "Acute upper respiratory infection",
                    "symptom_keywords": ["runny nose", "sore throat"],
                    "urgency_hint": 5,
                    "similarity_score": 0.89,
                }
            ]

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "urgency_score": 5,
                    "symptoms": ["runny nose", "sore throat"],
                    "required_specialty": "General Practice",
                    "reasoning": "Common cold",
                    "icd_codes_used": ["J06.9"],
                })))
            ]
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

            from agents.symptom_analyzer import SymptomAnalyzer
            analyzer = SymptomAnalyzer(state)
            await analyzer.analyze_utterance("I have a runny nose and sore throat for 2 days")

            assert state["urgency_score"] >= 4
            assert state["urgency_label"] == "ROUTINE"


# ---------------------------------------------------------------------------
# Follow-up Agent tests
# ---------------------------------------------------------------------------
class TestFollowupAgent:
    @pytest.mark.asyncio
    async def test_followup_sends_sms_and_email(self):
        """FollowupAgent.run() should attempt both SMS and email."""
        state = default_state("test-room-003")
        state["patient_name"] = "John Doe"
        state["patient_phone"] = "+12125551234"
        state["patient_email"] = "john@example.com"
        state["symptoms"] = ["headache", "fever"]
        state["appointment_details"] = {
            "doctor_name": "Dr. Anna Thompson",
            "specialty": "General Practice",
            "department": "Outpatient Clinic",
            "datetime": "2026-07-01T10:00:00Z",
        }

        with patch("agents.followup_agent.generate_summary", new_callable=AsyncMock) as mock_summary, \
             patch("agents.followup_agent.send_sms", new_callable=AsyncMock) as mock_sms, \
             patch("agents.followup_agent.send_email", new_callable=AsyncMock) as mock_email:

            mock_summary.return_value = "Your appointment is confirmed."
            mock_sms.return_value = True
            mock_email.return_value = True

            from agents.followup_agent import FollowupAgent
            agent = FollowupAgent(state)
            result = await agent.run()

            assert result["sms_sent"] is True
            assert result["email_sent"] is True
            assert state["followup_sent"] is True
            mock_sms.assert_called_once()
            mock_email.assert_called_once()


# ---------------------------------------------------------------------------
# Patient Info Extraction tests
# ---------------------------------------------------------------------------
class TestPatientExtraction:
    @pytest.mark.asyncio
    async def test_extract_patient_info_success(self):
        """extract_patient_info should extract structured details from transcript."""
        transcript = [
            "Patient: My name is Arthur Conan Doyle.",
            "Patient: I was born on May 22, 1859.",
            "Patient: Contact me at 555-987-6543 and email arthur@sherlock.com."
        ]
        
        with patch("entrypoint.AsyncOpenAI") as mock_openai:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "patient_name": "Arthur Conan Doyle",
                    "patient_dob": "1859-05-22",
                    "patient_phone": "555-987-6543",
                    "patient_email": "arthur@sherlock.com"
                })))
            ]
            mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
            
            from entrypoint import extract_patient_info
            result = await extract_patient_info(transcript)
            
            assert result is not None
            assert result["patient_name"] == "Arthur Conan Doyle"
            assert result["patient_dob"] == "1859-05-22"
            assert result["patient_phone"] == "555-987-6543"
            assert result["patient_email"] == "arthur@sherlock.com"
