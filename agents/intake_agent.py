"""
Intake Agent — Claude Haiku 3.5, voice I/O via LiveKit.

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
Your goal: gather the following information step-by-step in natural conversation:

1. Patient's full name
2. Date of birth (for identity verification)
3. Phone number (confirm it from caller ID if possible)
4. Email address (for follow-up summary)
5. Chief complaint — "what brings you in today?"
6. Symptom details — how long, severity 1-10, any relevant history

RULES:
- Ask ONE question at a time. Do not overwhelm the patient.
- If the patient sounds distressed, acknowledge it: "I can hear this is worrying you, you're doing great."
- If the patient says something like "chest pain", "can't breathe", "stroke", "unconscious" — immediately
  say: "This sounds very serious. Please hold while I connect you to emergency services." 
  Then stop asking questions.
- Keep responses SHORT (under 30 words). This is a phone call, not a chat.
- When all information is collected, say EXACTLY: "Thank you. Let me find you the right specialist now."
  This signals the end of intake to the system.

Begin by greeting the patient and asking for their name.
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
            vad=silero.VAD.load(),
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
            "I'm your AI triage assistant. Can I start with your full name please?"
        )
        logger.info("Intake Agent started in room: %s", self.state["room_id"])

    async def wait_for_completion(self) -> None:
        """Block until intake is complete (or interrupted)."""
        await self._completion_event.wait()

    async def say(self, text: str) -> None: # for confirmation/emergency messages.
        """Make the agent speak (used for confirmation/emergency messages)."""
        if self._session:
            self._session.say(text, allow_interruptions=False)

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
                "Please call 9-1-1 immediately, or stay on the line and I will connect you now.",
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
                # Check if agent signalled end of intake
                if "find you the right specialist" in text_content.lower():
                    self._completion_event.set()
