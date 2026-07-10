"""
Specialist Router Agent.

Runs after intake is complete (not during the call — avoids noise).
Takes the urgency score, symptoms, and required specialty from TriageState,
then uses Cerebras's tool-calling API to call the Mock HIS API and book a slot.

Returns the booked appointment details back into TriageState.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
import openai
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graph.state import TriageState
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(openai.RateLimitError),
    reraise=True,
)
async def _router_chat_completion(client: AsyncOpenAI, **kwargs) -> Any:
    return await client.chat.completions.create(**kwargs)

# ---------------------------------------------------------------------------
# HIS API base URL (FastAPI mock running locally or in Docker)
# ---------------------------------------------------------------------------
HIS_BASE_URL = "http://localhost:8000/schedule"

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI tool format)
# ---------------------------------------------------------------------------
ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_doctors",
            "description": (
                "Search the hospital's doctor database by medical specialty. "
                "Returns a list of available doctors in that department."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty, e.g. 'Cardiology', 'General Practice', 'Neurology'",
                    }
                },
                "required": ["specialty"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": "Get open appointment slots for a specific doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "UUID of the doctor",
                    }
                },
                "required": ["doctor_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book the earliest available slot for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {
                        "type": "string",
                        "description": "UUID of the slot to book",
                    },
                    "patient_id": {
                        "type": "string",
                        "description": "UUID of the patient",
                    },
                    "call_id": {
                        "type": "string",
                        "description": "UUID of the current call session",
                    },
                },
                "required": ["slot_id", "patient_id"],
            },
        }
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """You are a hospital scheduling coordinator AI.

You have tools to search for doctors, check availability, and book appointments.
Your job: find the most appropriate doctor for the patient based on their urgency and specialty,
then book the earliest available slot.

Rules:
- Always book the EARLIEST available slot.
- If the urgency score is 1-2, look for emergency/acute care.
- If the specialty is not found, fall back to 'General Practice'.
- After booking, confirm the appointment details.
- Be efficient: use the minimum number of tool calls needed.
"""


async def _call_his_api(method: str, path: str, json_body: dict | None = None) -> dict:
    """Call the Mock HIS API. Raises on HTTP errors."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = f"{HIS_BASE_URL}{path}"
        if method == "GET":
            resp = await client.get(url, params=json_body)
        else:
            resp = await client.post(url, json=json_body)
        resp.raise_for_status()
        return resp.json()


async def _execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if tool_name == "get_available_doctors":
            result = await _call_his_api("GET", "/doctors", tool_input)
        elif tool_name == "get_available_slots":
            result = await _call_his_api("GET", "/slots", tool_input)
        elif tool_name == "book_appointment":
            result = await _call_his_api("POST", "/book", tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except httpx.HTTPError as exc:
        result = {"error": str(exc)}

    return json.dumps(result)


class SpecialistRouter:
    """
    Uses Mistral with tool calling to query the Mock HIS API
    and book the best available appointment for the patient.
    """

    def __init__(self, state: TriageState) -> None:
        self.state = state
        self._client = AsyncOpenAI(
            api_key=settings.mistral_api_key,
            base_url="https://api.mistral.ai/v1",
        )
        # Normalize the required specialty stored in state to match the DB exactly
        if self.state.get("required_specialty"):
            self.state["required_specialty"] = self._normalize_specialty(self.state["required_specialty"])

    @staticmethod
    def _normalize_specialty(specialty: str) -> str:
        """Map common specialty spellings, typos, and synonyms to the exact DB specialty names."""
        if not specialty:
            return "General Practice"
        norm = specialty.strip().lower()
        if "ortho" in norm:
            return "Orthopaedics"
        if "pediatr" in norm or "paediatr" in norm:
            return "Paediatrics"
        if "cardio" in norm:
            return "Cardiology"
        if "emergency" in norm:
            return "Emergency Medicine"
        if "neuro" in norm:
            return "Neurology"
        if "pulmon" in norm:
            return "Pulmonology"
        if "gastro" in norm:
            return "Gastroenterology"
        if "dermat" in norm:
            return "Dermatology"
        if "psych" in norm:
            return "Psychiatry"
        if "general practice" in norm or "gp" in norm or "general medicine" in norm or "family" in norm:
            return "General Practice"
        return specialty.title()

    async def find_candidate_slots(self, max_slots: int = 2) -> list[dict]:
        """
        Find candidate appointment slots without booking.
        Returns a list of dicts: [{doctor_name, specialty, slot_id, datetime}, ...]
        """
        specialty = self.state.get("required_specialty", "General Practice")
        candidates: list[dict] = []

        try:
            doctors_result = await _call_his_api("GET", "/doctors", {"specialty": specialty})
            doctors = doctors_result if isinstance(doctors_result, list) else []

            if not doctors:
                # Fallback to General Practice
                doctors_result = await _call_his_api("GET", "/doctors", {"specialty": "General Practice"})
                doctors = doctors_result if isinstance(doctors_result, list) else []

            for doc in doctors:
                if len(candidates) >= max_slots:
                    break
                doc_id = doc.get("id")
                doc_name = doc.get("name", "Doctor")
                doc_specialty = doc.get("specialty", "General Practice")
                try:
                    slots_result = await _call_his_api("GET", "/slots", {"doctor_id": str(doc_id)})
                    slots = slots_result if isinstance(slots_result, list) else []
                    for slot in slots:
                        if len(candidates) >= max_slots:
                            break
                        candidates.append({
                            "doctor_name": doc_name,
                            "specialty": doc_specialty,
                            "slot_id": str(slot.get("id")),
                            "datetime": slot.get("datetime", ""),
                        })
                except Exception:
                    continue  # doctor has no available slots

        except Exception as exc:
            logger.error("Failed to find candidate slots: %s", exc)

        return candidates

    async def book_slot(self, slot_id: str) -> dict | None:
        """Book a specific slot by ID. Returns appointment details or None."""
        patient_id = self.state.get("patient_id", "")
        call_id = self.state.get("call_id", "")
        try:
            result = await _call_his_api("POST", "/book", {
                "slot_id": slot_id,
                "patient_id": patient_id,
                "call_id": call_id,
            })
            if "error" not in result:
                appointment_details = {
                    "doctor_name": result.get("doctor", {}).get("name", "your specialist"),
                    "specialty": result.get("doctor", {}).get("specialty", "General Practice"),
                    "department": result.get("doctor", {}).get("department", "Outpatient Clinic"),
                    "datetime": result.get("slot", {}).get("datetime", "the scheduled time"),
                    "appointment_id": result.get("id"),
                    "slot_id": result.get("slot_id"),
                }
                self.state["appointment_details"] = appointment_details
                return appointment_details
        except Exception as exc:
            logger.error("Failed to book slot %s: %s", slot_id, exc)
        return None

    async def route_and_book(self) -> dict | None:
        """
        Main entry point. Runs the tool-calling loop until an appointment is booked.
        Returns the appointment details dict, or None on failure.
        """
        specialty = self.state.get("required_specialty", "General Practice")
        symptoms = ", ".join(self.state.get("symptoms", []))
        urgency = self.state.get("urgency_score", 5)
        patient_id = self.state.get("patient_id", "")
        call_id = self.state.get("call_id", "")

        user_message = (
            f"Patient urgency score: {urgency}/5 (1=most urgent).\n"
            f"Required specialty: {specialty}.\n"
            f"Symptoms: {symptoms}.\n"
            f"Patient ID: {patient_id}.\n"
            f"Call ID: {call_id}.\n\n"
            "Please find and book the earliest appropriate appointment."
        )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for _ in range(6):
            response = await _router_chat_completion(
                self._client,
                model=settings.router_model,
                messages=messages,
                tools=ROUTER_TOOLS,
                temperature=0.0,
                parallel_tool_calls=False,
            )

            response_message = response.choices[0].message
            # Append response to message history
            messages.append(response_message)

            if not response_message.tool_calls:
                # Routing complete — extract appointment details from final message
                return self._extract_appointment(response_message)

            # Execute all tool calls in this turn
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except Exception:
                    function_args = {}
                
                result_str = await _execute_tool(function_name, function_args)
                
                if function_name == "book_appointment":
                    try:
                        res_dict = json.loads(result_str)
                        if "error" not in res_dict:
                            self.state["appointment_details"] = {
                                "doctor_name": res_dict.get("doctor", {}).get("name", "your specialist"),
                                "specialty": res_dict.get("doctor", {}).get("specialty", "General Practice"),
                                "department": res_dict.get("doctor", {}).get("department", "Outpatient Clinic"),
                                "datetime": res_dict.get("slot", {}).get("datetime", "the scheduled time"),
                                "appointment_id": res_dict.get("id"),
                                "slot_id": res_dict.get("slot_id"),
                            }
                    except Exception as e:
                        logger.error("Failed to parse book_appointment result for state: %s", e)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_str,
                })
                logger.debug("Tool: %s | Result: %s", function_name, result_str[:200])

        logger.error("Router failed to book appointment for room: %s", self.state["room_id"])
        return None

    def _extract_appointment(self, response_message: Any) -> dict | None:
        """Parse appointment details from the final assistant message."""
        text_content = response_message.content
        if text_content:
            # The LLM should have already booked via tool; state is in Redis/DB
            # Return a summary dict for the confirmation node
            appt = self.state.get("appointment_details")
            appt_dict = appt if isinstance(appt, dict) else {}
            return {
                "doctor_name": appt_dict.get("doctor_name", "your specialist"),
                "specialty": self.state.get("required_specialty", "General Practice"),
                "message": text_content,
            }
        return None
