"""
LiveKit Agent Entrypoint — the main worker process.

Run with:
    python entrypoint.py dev   (development, auto-reconnects)
    python entrypoint.py start (production)

Each inbound SIP call creates a new LiveKit Room.
LiveKit dispatches a job to this worker, which calls `entrypoint()`.

Concurrency model for a single call:
  ┌─ VoicePipelineAgent (IntakeAgent) — voice loop ──────────────────────┐
  │                                                                       │
  ├─ asyncio.Task: analyzer_loop                                          │
  │    Symptom Analyzer fires on every patient utterance                  │
  │    Updates urgency_score in shared TriageState                        │
  │                                                                       │
  ├─ asyncio.Task: supervisor_loop                                        │
  │    Polls urgency_score every 2 seconds                                │
  │    If score ≤ threshold → interrupts IntakeAgent for emergency        │
  │                                                                       │
  └─ LangGraph graph (after intake complete)                              │
       supervisor → [emergency | router] → confirmation → followup        │
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli
from openai import AsyncOpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agents.intake_agent import IntakeAgent
from agents.symptom_analyzer import SymptomAnalyzer
from graph.state import TriageState, default_state
from graph.supervisor import is_emergency
from graph.triage_graph import triage_graph
from config.settings import get_settings

# Configure LangSmith tracing
settings = get_settings()
os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, settings.log_level))


# ---------------------------------------------------------------------------
# Background: Supervisor polling task
# ---------------------------------------------------------------------------
async def supervisor_loop(
    intake_agent: IntakeAgent,
    state: TriageState,
    stop_event: asyncio.Event,
    poll_interval: float = 2.0,
) -> None:
    """
    Polls urgency_score every `poll_interval` seconds.
    Triggers emergency interrupt if score breaches the threshold.
    """
    while not stop_event.is_set():
        if is_emergency(state):
            logger.warning(
                "Supervisor: EMERGENCY detected | room=%s score=%d",
                state["room_id"],
                state["urgency_score"],
            )
            await intake_agent.interrupt_for_emergency()
            break
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Persist call state to Postgres
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(openai.RateLimitError),
    reraise=True,
)
async def extract_patient_info(transcript: list[str]) -> dict | None:
    """
    Extract patient name, DOB, phone, and email from the conversation transcript
    using Mistral (mistral-small-latest).
    """
    if not transcript:
        return None
    try:
        import json

        client = AsyncOpenAI(
            api_key=settings.mistral_api_key,
            base_url="https://api.mistral.ai/v1",
        )
        
        system_prompt = (
            "You are a clinical data extraction assistant.\n"
            "Analyze the conversation transcript between a Patient and an AI Agent, and extract the patient's personal details.\n\n"
            "Format your response as a JSON object with the following keys:\n"
            "- patient_name: Patient's full name (or null if not found)\n"
            "- patient_dob: Patient's date of birth. Convert it to 'YYYY-MM-DD' format (e.g. '1985-01-15'). If not found or incomplete, output null.\n"
            "- patient_phone: Patient's phone number (or null if not found)\n"
            "- patient_email: Patient's email address (or null if not found)\n\n"
            "Output ONLY the raw JSON object. Do not include markdown code block formatting or any conversational text."
        )
        user_msg = "\n".join(transcript)
        
        response = await client.chat.completions.create(
            model=settings.intake_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0
        )
        
        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                raw_content = "\n".join(lines[1:-1]).strip()

        return json.loads(raw_content)
    except Exception as e:
        logger.warning("Failed to extract patient info from transcript: %s", e)
        return None


async def save_call_to_db(state: TriageState) -> str | None:
    """
    Write a Call row to Postgres and return the call UUID.
    Also creates a default Patient record so foreign key checks succeed.
    Gracefully handles DB unavailability (demo mode).
    """
    try:
        from db.database import AsyncSessionLocal
        from db.models import Call, Patient
        import uuid as _uuid
        from datetime import date

        call_id = str(_uuid.uuid4())
        patient_id = str(_uuid.uuid4())
        async with AsyncSessionLocal() as session:
            # 1. Create a default Patient record so foreign key checks succeed
            patient = Patient(
                id=_uuid.UUID(patient_id),
                name="Unknown Patient",
                dob=date(1900, 1, 1),
                phone=f"+1{str(_uuid.uuid4().int)[:10]}",
                email="unknown@example.com",
            )
            session.add(patient)

            # 2. Create the Call record linked to the patient
            call = Call(
                id=_uuid.UUID(call_id),
                room_id=state["room_id"],
                started_at=datetime.now(timezone.utc),
                patient_id=_uuid.UUID(patient_id),
            )
            session.add(call)
            await session.commit()

        state["call_id"] = call_id
        state["patient_id"] = patient_id
        return call_id
    except Exception as exc:
        logger.warning("DB unavailable — call not persisted: %s", exc)
        return None


async def update_call_in_db(state: TriageState) -> None:
    """Update the Call row with transcript, urgency, and ICD codes after call ends."""
    call_id = state.get("call_id")
    if not call_id:
        return
    try:
        from db.database import AsyncSessionLocal
        from db.models import Call, Patient
        from sqlalchemy import select
        import uuid as _uuid

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Call).where(Call.id == _uuid.UUID(call_id))
            )
            call = result.scalar_one_or_none()
            if call:
                call.ended_at = datetime.now(timezone.utc)
                call.transcript = "\n".join(state.get("transcript", []))
                call.urgency_score = state.get("urgency_score")
                call.urgency_label = state.get("urgency_label")
                call.icd_codes = [m["icd_code"] for m in state.get("icd_matches", []) if m.get("icd_code")]
                
                # Update patient details if available
                if call.patient_id:
                    try:
                        res_p = await session.execute(
                            select(Patient).where(Patient.id == call.patient_id)
                        )
                        patient = res_p.scalar_one_or_none()
                        if patient:
                            if state.get("patient_name"):
                                patient.name = state["patient_name"]
                            if state.get("patient_phone"):
                                phone_taken = False
                                if state.get("patient_phone") != patient.phone:
                                    res_check = await session.execute(
                                        select(Patient).where(Patient.phone == state["patient_phone"])
                                    )
                                    if res_check.scalar_one_or_none():
                                        phone_taken = True
                                        logger.warning("Phone %s is already registered to another patient record.", state["patient_phone"])
                                
                                if not phone_taken:
                                    patient.phone = state["patient_phone"]
                            if state.get("patient_email"):
                                patient.email = state["patient_email"]
                            if state.get("patient_dob"):
                                try:
                                    patient.dob = datetime.strptime(state["patient_dob"], "%Y-%m-%d").date()
                                except Exception as e:
                                    logger.warning("Failed to parse patient DOB: %s", e)
                    except Exception as p_exc:
                        logger.warning("Failed to update patient details: %s", p_exc)
                            
                await session.commit()
    except Exception as exc:
        logger.warning("Failed to update call in DB: %s", exc)


async def filler_message_timer(intake_agent: IntakeAgent, stop_event: asyncio.Event) -> None:
    """Waits for 10 seconds and triggers a filler speech if booking is still in progress."""
    try:
        await asyncio.sleep(10.0)
        if not stop_event.is_set():
            logger.info("Intake complete but booking still in progress after 10s — speaking filler message")
            await intake_agent.say(
                "Please stay with me. I am currently finding the best available appointment slot for you."
            )
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Main LiveKit entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext) -> None:
    """
    Called by LiveKit for each inbound call (one coroutine per room).
    """
    room_name = ctx.room.name
    logger.info("New call received | room=%s", room_name)

    # 1. Initialise shared state
    state: TriageState = default_state(room_name)

    # 2. Persist call to DB (non-blocking — demo can work without DB)
    await save_call_to_db(state)

    # 3. Connect to the room
    await ctx.connect()

    # 4. Initialise agents
    intake_agent = IntakeAgent(ctx, state)
    symptom_analyzer = SymptomAnalyzer(state)

    # 5. Wire the Symptom Analyzer to fire on every patient utterance
    intake_agent.on_transcript(symptom_analyzer.analyze_utterance)

    # 6. Create stop event for the supervisor loop
    stop_event = asyncio.Event()

    # 7. Start parallel background tasks
    supervisor_task = asyncio.create_task(
        supervisor_loop(intake_agent, state, stop_event)
    )

    # 8. Start the voice agent (Intake Agent)
    await intake_agent.start()

    try:
        # 9. Wait for intake to complete (patient has given all info)
        await intake_agent.wait_for_completion()
        logger.info("Intake complete | room=%s urgency=%d", room_name, state.get("urgency_score", 5))

        # Start background filler message timer
        filler_stop_event = asyncio.Event()
        filler_task = asyncio.create_task(
            filler_message_timer(intake_agent, filler_stop_event)
        )

        try:
            # 10. Stop supervisor loop
            stop_event.set()
            supervisor_task.cancel()

            # 10b. Extract patient details from transcript to populate state before routing/booking
            try:
                transcript_list = state.get("transcript", [])
                logger.info("Full transcript for room=%s before extraction: %s", room_name, transcript_list)
                extracted = await extract_patient_info(transcript_list)
                if extracted:
                    if extracted.get("patient_name"):
                        state["patient_name"] = extracted["patient_name"]
                    if extracted.get("patient_dob"):
                        state["patient_dob"] = extracted["patient_dob"]
                    if extracted.get("patient_phone"):
                        state["patient_phone"] = extracted["patient_phone"]
                    if extracted.get("patient_email"):
                        state["patient_email"] = extracted["patient_email"]
                    logger.info("Extracted patient info: %s", extracted)
            except Exception as exc:
                logger.error("Failed to extract patient info: %s", exc)

            # 11. Run LangGraph routing graph
            try:
                final_state = await triage_graph.ainvoke(state)
                state.update(final_state)
            except Exception as exc:
                logger.error("LangGraph error: %s", exc)
                state["error"] = str(exc)
        finally:
            # Always cancel and stop the filler timer task when graph execution completes (success or fail)
            filler_stop_event.set()
            filler_task.cancel()

        # 12. Speak the confirmation to the patient
        confirmation_text = state.get("confirmation_text", "")  # type: ignore[attr-defined]
        logger.info(
            "Confirmation text for room=%s: %r (length=%d)",
            room_name, confirmation_text[:80] if confirmation_text else "<empty>",
            len(confirmation_text) if confirmation_text else 0,
        )
        if confirmation_text:
            await intake_agent.say(confirmation_text)
            # Sleep for a duration proportional to the word count, with a minimum of 10 seconds
            word_count = len(confirmation_text.split())
            sleep_time = max(10.0, word_count / 2.0)
            logger.info("Sleeping %.1fs for TTS playout (%d words) | room=%s", sleep_time, word_count, room_name)
            await asyncio.sleep(sleep_time)
        else:
            logger.warning("No confirmation text — skipping speech | room=%s", room_name)

    finally:
        # 13. End the call and guarantee DB persistence (shielded from cancellation)
        state["call_ended"] = True
        
        # Stop supervisor loop if it hasn't been stopped
        if not stop_event.is_set():
            stop_event.set()
            supervisor_task.cancel()

        # If patient info was not extracted yet, try one last time
        if not state.get("patient_name") and state.get("transcript"):
            try:
                transcript_list = state["transcript"]
                logger.info("Full transcript for room=%s in finally before extraction: %s", room_name, transcript_list)
                extracted = await extract_patient_info(transcript_list)
                if extracted:
                    if extracted.get("patient_name"):
                        state["patient_name"] = extracted["patient_name"]
                    if extracted.get("patient_dob"):
                        state["patient_dob"] = extracted["patient_dob"]
                    if extracted.get("patient_phone"):
                        state["patient_phone"] = extracted["patient_phone"]
                    if extracted.get("patient_email"):
                        state["patient_email"] = extracted["patient_email"]
            except Exception as exc:
                logger.error("Failed to extract patient info in finally: %s", exc)

        try:
            await asyncio.shield(update_call_in_db(state))
        except Exception as exc:
            logger.error("Failed to update call in DB (cancellation shielded): %s", exc)
        logger.info("Call ended | room=%s", room_name)


# ---------------------------------------------------------------------------
# Worker entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )
