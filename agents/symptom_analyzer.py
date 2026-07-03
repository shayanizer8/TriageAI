"""
Symptom Analyzer Agent — Cerebras, silent (no voice I/O).

Runs in parallel with the Intake Agent. On every patient utterance:
  1. Queries Pinecone for matching ICD-10 conditions
  2. Sends transcript + RAG results to LLM for urgency scoring
  3. Updates urgency_score + symptoms + icd_matches in shared TriageState
  4. Persists state to Redis after each update
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
- Determine the required medical specialty.

You MUST respond with ONLY valid JSON in this exact format:
{
  "urgency_score": <int 1-5>,
  "symptoms": ["<symptom>", ...],
  "required_specialty": "<specialty name>",
  "reasoning": "<one sentence max>",
  "icd_codes_used": ["<ICD code>", ...]
}
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

        # Debounce 2: Rate limit calls to Cerebras to at most once every 10 seconds.
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

        # 3. Parse and validate
        analysis = _parse_analysis(raw_text)
        score = int(analysis.get("urgency_score", 5))
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
        self.state["required_specialty"] = analysis.get(
            "required_specialty", self.state.get("required_specialty")
        )

        logger.info(
            "Analyzer update | room=%s score=%d label=%s specialty=%s",
            self.state["room_id"],
            score,
            self.state["urgency_label"],
            self.state.get("required_specialty"),
        )
