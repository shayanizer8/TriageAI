"""
Intake Agent.

Responsibilities:
  1. Greet the patient
  2. Collect: name, date of birth, phone, email, chief complaint, symptoms
  3. Emit transcript events so the Symptom Analyzer can read in real-time
  4. Accept an "interrupt" signal from the Supervisor to switch to emergency path
  5. Deliver the appointment confirmation at call end

This agent wraps LiveKit's voice Agent and wires up STT (Groq),
TTS (Cartesia), through the Agents SDK plugins.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Awaitable

from livekit import agents, rtc
from livekit.agents import JobContext
from livekit.agents.voice import Agent, AgentSession
from livekit.agents import llm as agents_llm
from livekit.plugins import openai, groq, cartesia, silero

from graph.state import TriageState
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# System prompt for the Intake Agent
# ---------------------------------------------------------------------------
INTAKE_SYSTEM_PROMPT = f"""You are a warm, professional medical triage assistant for {settings.hospital_name}.
Your role is to collect information from the patient during a phone call.

Your tone: calm, empathetic, clear. Never use medical jargon the patient won't understand.

CONVERSATION ORDER — You MUST complete each step in sequence (Step 1 to Step 12). Do NOT skip to a later step or jump to appointment booking until all preceding steps are fully answered:

Step 1. CHIEF COMPLAINT: Ask "What brings you in today?" (the patient's main reason for calling).
Step 2. ONSET/TRIGGER: Ask "Did something specific trigger this — like an injury, activity, or event — or did it come on by itself?" (Ask this for EVERY complaint).
Step 3. DURATION: Ask "How long has this been going on?"
Step 4. SEVERITY: Ask "On a scale of 1 to 10, how would you rate it?"
Step 5. COMPLAINT-SPECIFIC QUESTIONS: Ask the relevant sub-questions based on the complaint:
    - PAIN: Ask exactly where the pain is and whether it spreads or radiates.
    - SKIN: Ask if it's spreading, itchy, or changing.
    - RESPIRATORY: Ask about difficulty breathing, cough type, and fever.
    - GI: Ask about nausea, vomiting, and appetite changes.
    - NEUROLOGICAL: Ask about dizziness, vision changes, or numbness.
Step 6. AMBIGUOUS / RED-FLAG PROBING: If the patient describes a boundary symptom (e.g. general tightness, numbness, headache, breathing discomfort), ask clarifying questions to rule out emergency flags (e.g. for tightness, ask if it's in the chest or throat; for headache, ask if it started like a sudden thunderclap).
Step 7. CURRENT TREATMENT: Ask "Have you tried anything for it so far — any medication, rest, or home remedies?"
Step 8. ALLERGIES: Ask "Do you have any known medication allergies?"
Step 9. PATIENT NAME: Ask the patient for their full name.
Step 10. PATIENT DOB: Ask the patient for their date of birth.
Step 11. PATIENT PHONE: Ask the patient for their phone number.
Step 12. PATIENT EMAIL: Ask the patient for their email address. After they state it, spell it back character by character with hyphens (e.g., "So that's J-O-H-N at G-M-A-I-L dot com — is that right?") and confirm it.
Step 13. BOOKING TRIGGER: ONLY when Steps 1 through 12 are fully completed, say EXACTLY: "Thank you. Let me check available appointments for you now." Do NOT say this phrase or check slots until Step 12 (Email confirmation) has been completed.

RULES:
- Ask ONE question at a time. Do not overwhelm the patient.
- Do NOT speak the step numbers, step names, or step progress updates (e.g., do NOT say "Step 1", "Step 2", "Step 1 complete", or "Phase 3"). Converse naturally and ask the questions as a human assistant would.
- If the patient sounds distressed, acknowledge it: "I can hear this is worrying you, you're doing great."
- If the patient says something like "chest pain", "can't breathe", "stroke", "unconscious" — immediately
  say: "This sounds very serious. Please hang up and call 9-1-1 immediately."
  Then stop asking questions.
- You MUST explicitly ask for and collect all four demographics in Steps 9, 10, 11, and 12. Do NOT skip any of these details under any circumstances.
- Keep responses SHORT (under 30 words). This is a phone call, not a chat.
"""


class IntakeAgent:
    """
    Wraps LiveKit's voice Agent for the medical intake flow.

    Usage:
        agent = IntakeAgent(ctx, state)
        await agent.start()
        await agent.wait_for_completion()
    """

    def __init__(self, ctx: JobContext, state: TriageState) -> None:
        self.ctx = ctx
        self.state = state
        self._completion_event = asyncio.Event()
        self._interrupted = False

        # Callbacks registered by the orchestrator
        self._on_transcript_callbacks: list[Callable[[str], Awaitable[None]]] = [] # a list that will hold the background function symptom analyzer.

        # Build the LiveKit voice Agent (v1.6.x API)
        self._agent = Agent(
            instructions=INTAKE_SYSTEM_PROMPT,
            vad=silero.VAD.load(
                activation_threshold=0.6,
                min_speech_duration=0.3,
                min_silence_duration=0.3,
            ),
            stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en",
                api_key=settings.groq_api_key,
            ),
            llm=openai.LLM(
                model=settings.intake_model,
                api_key=settings.mistral_api_key,
                base_url="https://api.mistral.ai/v1",
            ),
            tts=cartesia.TTS(
                model="sonic-3",
                api_key=settings.cartesia_api_key,
            ),
            allow_interruptions=True,
            min_endpointing_delay=1.0,
        )

        # Session will be created when start() is called
        self._session: AgentSession | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start the voice agent in the LiveKit room."""
        self._session = AgentSession()
        self._wire_events()
        await self._session.start(self._agent, room=self.ctx.room)
        await asyncio.sleep(1.0)  # Brief pause for audio to stabilise
        self._session.say(
            f"Hello, thank you for calling {settings.hospital_name}. "
            "I'm your AI triage assistant. Can you tell me what's brought you in today?"
        )
        logger.info("Intake Agent started in room: %s", self.state["room_id"])

    async def set_silent_standby(self) -> None:
        """Put the agent in silent standby mode (instructions + manual turn detection)."""
        logger.info("Setting IntakeAgent to silent standby mode")
        # if self._session:
        #     try:
        #         self._session.update_options(turn_detection="manual")
        #     except Exception as e:
        #         logger.warning("Failed to set turn_detection to manual: %s", e)
        await self._agent.update_instructions(
            "You are in silent standby mode. Under no circumstances should you speak, respond, or generate any text. "
            "Remain completely silent."
        )

    async def wait_for_completion(self) -> None:
        """Block until intake is complete (or interrupted)."""
        await self._completion_event.wait()

    @staticmethod
    def _sanitize_for_tts(text: str) -> str:
        """Expand abbreviations that TTS engines mispronounce."""
        import re
        # "Dr." or "Dr " at word boundary -> "Doctor "
        text = re.sub(r'\bDr\.\s*', 'Doctor ', text)
        text = re.sub(r'\bDr\s', 'Doctor ', text)
        return text

    async def say(self, text: str) -> None: # for confirmation/emergency messages.
        """Make the agent speak (used for confirmation/emergency messages)."""
        if self._session:
            try:
                self._session.say(self._sanitize_for_tts(text), allow_interruptions=False)
            except Exception as e:
                logger.warning("Failed to speak message (session may be closing/draining): %s", e)

    async def interrupt_for_emergency(self) -> None:
        """
        Called by the Supervisor when urgency_score <= emergency_threshold.
        Immediately overrides the conversation with an emergency message.
        """
        if self._interrupted:
            return
        self._interrupted = True
        logger.warning("EMERGENCY INTERRUPT triggered for room: %s", self.state["room_id"])
        if self._session:
            self._session.say(
                "I need to stop you there — what you're describing sounds like a medical emergency. "
                "Please hang up and call 9-1-1 immediately.",
                allow_interruptions=False,
            )
        self._completion_event.set()

    def on_transcript(self, callback: Callable[[str], Awaitable[None]]) -> None: 
        """Register a callback invoked on each finalised patient utterance."""
        self._on_transcript_callbacks.append(callback) # everytime the intake agent hears a sentence from the patient, it will call this function and pass the sentence to symptom analyzer.

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _wire_events(self) -> None:
        """Hook into LiveKit session events."""
        if not self._session:
            return

        @self._session.on("conversation_item_added")
        def _on_conversation_item(ev) -> None:
            item = ev.item if hasattr(ev, "item") else ev
            role = getattr(item, "role", None)
            text_content = ""
            if hasattr(item, "text_content"):
                text_content = item.text_content
            elif hasattr(item, "content"):
                text_content = str(item.content)

            if not text_content:
                return

            if role == "user":
                # Append to shared state transcript
                self.state["transcript"] = self.state.get("transcript", []) + [f"Patient: {text_content}"]

                # Notify all registered callbacks (e.g., Symptom Analyzer)
                for cb in self._on_transcript_callbacks:
                    asyncio.create_task(cb(text_content))
            elif role == "assistant":
                self.state["transcript"] = (
                    self.state.get("transcript", []) + [f"Agent: {text_content}"]
                )
                # Check if agent signalled end of intake (strict match)
                cleaned_text = text_content.lower().replace(",", "").replace("!", "").replace(".", "").strip()
                if "thank you let me check available appointments for you now" in cleaned_text:
                    self._completion_event.set()
