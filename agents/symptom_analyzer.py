"""
Symptom Analyzer Agent — Cerebras, silent (no voice I/O).

Runs in parallel with the Intake Agent. On every patient utterance:
  1. Queries Pinecone for matching ICD-10 conditions
  2. Sends transcript + RAG results to LLM for urgency scoring
  3. Updates urgency_score + symptoms + icd_matches in shared TriageState
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graph.state import TriageState
from rag.pinecone_client import query_medical_kb
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
async def _analyzer_chat_completion(client: AsyncOpenAI, **kwargs) -> Any:
    return await client.chat.completions.create(**kwargs)

# ---------------------------------------------------------------------------
# System prompt — deliberately safety-biased (err toward emergency)
# ---------------------------------------------------------------------------
ANALYZER_SYSTEM_PROMPT = """You are a senior emergency medicine physician AI assistant performing real-time triage.

Your job: analyse the patient's transcript and the retrieved ICD-10 conditions, then output a structured
urgency assessment.

URGENCY SCALE (always err toward higher urgency when uncertain):
  1 = IMMEDIATE — life-threatening (cardiac arrest, stroke, severe anaphylaxis, respiratory failure)
  2 = EMERGENCY — needs ED within 1 hour (chest pain, high fever with confusion, active bleeding)
  3 = URGENT    — needs same-day appointment (moderate pain, fever, infection symptoms)
  4 = SEMI-URGENT — within 48 hours (minor injury, rash, earache)
  5 = ROUTINE   — next available GP (prescription refill, check-up, mild cold)

RULES:
- When in doubt, go UP (lower number = more urgent). Never downplay symptoms.
- "Chest pain" alone = score 2 minimum.
- "Can't breathe" = score 1.
- Pediatric patients (under 12) should be scored at least one level higher.
- Extract specific symptoms as a flat list of medical keyword strings.
- Clarify Ambiguous Symptoms (Triage Probing):
  If the patient describes an ambiguous or vague symptom that could be an emergency OR routine depending on details (e.g. general "tightness", generic "headache", generic "chest discomfort", general "numbness" without other signs), do NOT assign a score of 1 or 2 yet. Instead, assign a score of 3 (URGENT) to allow the Intake Agent to ask clarifying questions. Only upgrade to a score of 1 or 2 (EMERGENCY/IMMEDIATE) once the patient explicitly confirms critical red-flag details (e.g., the tightness is in the chest/heart region, the pain radiates to the arm/jaw, the headache was a sudden "thunderclap", or there is active severe shortness of breath).
- Psychiatric / Mental Health Complaints:
  For psychiatric, psychological, or mental health complaints (e.g. general anxiety, panic, depression, stress, loneliness), do NOT assign an emergency score of 1 or 2 (EMERGENCY/IMMEDIATE) unless the patient explicitly expresses active intent of self-harm, suicidal ideation, or severe psychotic detachment. High ratings of subjective emotional tension (e.g. 9/10 tension) are URGENT (Score 3) and should be routed to Psychiatry, not redirected to 911/ED.
- Determine the required medical specialty. You MUST choose one of the following exact specialties:
  * Cardiology
  * Emergency Medicine
  * General Practice
  * Neurology
  * Pulmonology
  * Gastroenterology
  * Orthopaedics
  * Dermatology
  * Psychiatry
  * Paediatrics

You MUST respond with ONLY valid JSON in this exact format:
{
  "urgency_score": <int 1-5>,
  "symptoms": ["<symptom>", ...],
  "required_specialty": "<specialty name>",
  "reasoning": "<one sentence max>",
  "icd_codes_used": ["<ICD code>", ...],
  "indicators": {
    "cardiac_red_flags": <true/false>,
    "respiratory_distress": <true/false>,
    "stroke_neurological_deficits": <true/false>,
    "active_uncontrolled_bleeding": <true/false>,
    "active_self_harm_or_suicidal_intent": <true/false>
  }
}

INDICATORS RULES:
- cardiac_red_flags: set to true ONLY for acute cardiac symptoms (e.g., chest pain/pressure/squeezing radiating to arm/jaw, or suspected acute heart attack/angina).
- respiratory_distress: set to true ONLY if the patient is experiencing severe difficulty breathing, acute respiratory failure, throat swelling/anaphylaxis, or stridor.
- stroke_neurological_deficits: set to true ONLY for acute signs of stroke (e.g., one-sided facial drooping, one-sided limb weakness, sudden slurred speech, or acute loss of balance).
- active_uncontrolled_bleeding: set to true ONLY for severe, life-threatening hemorrhage (e.g., arterial bleeding, severe active hematemesis, or traumatic amputation).
- active_self_harm_or_suicidal_intent: set to true ONLY if the patient expresses active suicidal ideation, intent to self-harm, or severe psychotic detachment.
"""


def _build_analysis_prompt(
    transcript: list[str],
    icd_matches: list[dict],
) -> str:
    """Build the user message for the analysis call."""
    transcript_text = "\n".join(transcript[-10:])  # last 10 utterances
    icd_text = json.dumps(icd_matches, indent=2) if icd_matches else "No matches yet."

    return f"""PATIENT TRANSCRIPT (most recent utterances):
{transcript_text}

RETRIEVED ICD-10 CONDITIONS (from medical KB):
{icd_text}

Please analyse and return your urgency assessment as JSON."""


def _parse_analysis(raw: str) -> dict[str, Any]:
    """Parse the JSON response from the LLM. Falls back gracefully."""
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to parse analyzer response: %s", raw[:200])
        # Safe fallback — assume routine to avoid false emergencies
        return {
            "urgency_score": 4,
            "symptoms": [],
            "required_specialty": "General Practice",
            "reasoning": "Parse error — defaulting to semi-urgent.",
            "icd_codes_used": [],
        }


class SymptomAnalyzer:
    """
    Silent background agent that scores urgency on each patient utterance.

    Usage:
        analyzer = SymptomAnalyzer(state)
        # Register as a transcript callback on IntakeAgent:
        intake_agent.on_transcript(analyzer.analyze_utterance)
    """

    def __init__(self, state: TriageState) -> None:
        self.state = state
        self._client = AsyncOpenAI(
            api_key=settings.mistral_api_key,
            base_url="https://api.mistral.ai/v1",
        )
        # Semaphore — only one analysis at a time to avoid race conditions
        self._lock = asyncio.Semaphore(1)
        # Time-based rate limit debounce
        import time
        self._last_run_time = 0.0

    async def analyze_utterance(self, utterance: str) -> None:
        """
        Called on every finalised patient utterance.
        Non-blocking: acquires lock, runs analysis, updates state.
        """
        async with self._lock:
            await self._run_analysis()

    async def _run_analysis(self) -> None:
        """Core analysis loop: RAG → LLM → state update."""
        transcript = self.state.get("transcript", [])
        if not transcript:
            return

        # Skip analysis if we have transitioned to Phase 3 (Demographics)
        # We can detect this if the last agent utterance asked for name, DOB, phone, or email.
        agent_lines = [t for t in transcript if t.startswith("Agent:")]
        if agent_lines:
            last_agent_text = agent_lines[-1].lower()
            if any(k in last_agent_text for k in ["full name", "date of birth", "dob", "phone number", "email address"]):
                logger.debug("SymptomAnalyzer: skipping because we are in the demographics phase (Phase 3)")
                return

        # 1. Query Pinecone for relevant ICD-10 conditions
        # Use the last patient utterance as the query
        patient_lines = [t for t in transcript if t.startswith("Patient:")]
        if not patient_lines:
            return

        latest_patient_text = patient_lines[-1].replace("Patient: ", "").strip()

        # Debounce 1: Skip if the utterance is very short (e.g. "yes", "okay", "no", DOB or name)
        # since it doesn't contain medical symptoms to analyze.
        if len(latest_patient_text.split()) < 3:
            logger.debug("SymptomAnalyzer: skipping short utterance: %s", latest_patient_text)
            return

        # Debounce 2: Rate limit calls to at most once every 10 seconds.
        import time
        now = time.time()
        if now - self._last_run_time < 10.0:
            logger.debug("SymptomAnalyzer: skipping due to 10s rate-limit debounce")
            return
        self._last_run_time = now

        try:
            icd_matches = await query_medical_kb(latest_patient_text, top_k=5)
        except Exception as exc:
            logger.warning("Pinecone query failed: %s", exc)
            icd_matches = []

        # 2. Call LLM for urgency scoring
        try:
            response = await _analyzer_chat_completion(
                self._client,
                model=settings.analyzer_model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_analysis_prompt(transcript, icd_matches),
                    },
                ],
            )
            raw_text = response.choices[0].message.content
        except Exception as exc:
            logger.error("Cerebras API error in SymptomAnalyzer: %s", exc)
            return

        # 3. Parse, validate, and compute clinical urgency deterministically
        analysis = _parse_analysis(raw_text)
        
        # Implement the Decoupled Triage Engine Logic:
        indicators = analysis.get("indicators")
        if indicators is None:
            # Backward-compatibility fallback: use raw LLM score (e.g. for existing mocks/tests)
            score = int(analysis.get("urgency_score", 4))
        else:
            has_emergency_flag = (
                indicators.get("cardiac_red_flags", False) or
                indicators.get("respiratory_distress", False) or
                indicators.get("stroke_neurological_deficits", False) or
                indicators.get("active_uncontrolled_bleeding", False) or
                indicators.get("active_self_harm_or_suicidal_intent", False)
            )

            if has_emergency_flag:
                # Overrides to Clinical Emergency
                if indicators.get("respiratory_distress") or indicators.get("active_self_harm_or_suicidal_intent"):
                    score = 1  # Immediate
                else:
                    score = 2  # Emergency
            else:
                # If no emergency flags are active, force score to be at least 3 (Urgent / Routine)
                raw_score = int(analysis.get("urgency_score", 4))
                score = max(3, raw_score)

        score = max(1, min(5, score))  # clamp to [1, 5]

        # 4. Update shared state
        self.state["urgency_score"] = score
        self.state["urgency_label"] = (
            "EMERGENCY" if score <= settings.emergency_threshold
            else "URGENT" if score == 3
            else "ROUTINE"
        )
        self.state["symptoms"] = analysis.get("symptoms", [])
        self.state["icd_matches"] = icd_matches
        new_specialty = analysis.get("required_specialty")
        current_specialty = self.state.get("required_specialty")
        if new_specialty:
            if current_specialty and current_specialty != "General Practice" and new_specialty == "General Practice":
                # Preserve the more specific specialty already found in previous turns
                pass
            else:
                self.state["required_specialty"] = new_specialty

        logger.info(
            "Analyzer update | room=%s score=%d label=%s specialty=%s",
            self.state["room_id"],
            score,
            self.state["urgency_label"],
            self.state.get("required_specialty"),
        )
