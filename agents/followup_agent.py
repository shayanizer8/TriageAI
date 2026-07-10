from __future__ import annotations

import logging
from datetime import datetime
import asyncio
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from openai import AsyncOpenAI
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graph.state import TriageState
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
FOLLOWUP_SYSTEM_PROMPT = f"""You are a medical notification system at {settings.hospital_name}.

Generate the body content of a follow-up email for a patient who completed a triage call.

Strict formatting rules:
1. Do NOT include a subject line (e.g. do NOT write "Subject: ...").
2. Do NOT include a greeting (e.g. do NOT write "Dear Ahmed Shazad" or "Dear Patient"). A greeting is added programmatically.
3. Do NOT include any placeholder signatures (e.g. do NOT write "[Your Name]" or similar sign-offs).
4. Start directly with the confirmation details.

Include:
  - Appointment confirmation (doctor, specialty, date/time)
  - What they should bring (ID, insurance card, medication list)
  - A brief summary of their reported symptoms (1 sentence)
  - Emergency reminder: "If your condition worsens before your appointment, call 911 or go to your nearest ED."
  - Contact: {settings.hospital_name}, {settings.hospital_address}

Keep it under 150 words. Warm and reassuring tone. No medical jargon.
"""


def _build_summary_prompt(state: TriageState) -> str:
    appointment = state.get("appointment_details") or {}
    symptoms = ", ".join(state.get("symptoms", [])) or "reported symptoms"
    doctor = appointment.get("doctor_name", "your assigned specialist")
    specialty = appointment.get("specialty", "")
    appt_datetime = appointment.get("datetime", "the scheduled time")
    patient_name = state.get("patient_name", "Patient")
    onset = state.get("onset_trigger", "")
    treatment = state.get("current_treatment", "")
    allergies = state.get("known_allergies", "")

    prompt = (
        f"Patient name: {patient_name}\n"
        f"Reported symptoms: {symptoms}\n"
        f"Doctor: {doctor} ({specialty})\n"
        f"Appointment: {appt_datetime}\n"
    )
    if onset:
        prompt += f"Onset/trigger: {onset}\n"
    if treatment:
        prompt += f"Current treatment tried: {treatment}\n"
    if allergies:
        prompt += f"Known allergies: {allergies}\n"
    prompt += "\nPlease write the follow-up message."
    return prompt


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(openai.RateLimitError),
    reraise=True,
)
async def generate_summary(state: TriageState) -> str:
    client = AsyncOpenAI(
        api_key=settings.mistral_api_key,
        base_url="https://api.mistral.ai/v1",
    )
    response = await client.chat.completions.create(
        model=settings.followup_model,
        max_tokens=400,
        messages=[
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": _build_summary_prompt(state)},
        ],
    )
    return response.choices[0].message.content





def _send_email_sync(to_email: str, patient_name: str, message: str) -> bool:
    if not settings.gmail_address or not settings.gmail_app_passwprd:
        logger.warning("Gmail SMTP credentials not configured — skipping email send")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = settings.gmail_address
        msg["To"] = to_email
        msg["Subject"] = f"Your appointment at {settings.hospital_name} — confirmation"
        msg.attach(MIMEText(_format_html_email(patient_name, message), "html"))
        
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10.0) as server:
            server.starttls()
            server.login(settings.gmail_address, settings.gmail_app_passwprd)
            server.sendmail(settings.gmail_address, to_email, msg.as_string())
        logger.info("Email sent via Gmail SMTP to=%s", to_email)
        return True
    except Exception as exc:
        logger.error("Gmail SMTP email error: %s", exc)
        return False


async def send_email(to_email: str, patient_name: str, message: str) -> bool:
    """Send email via Gmail SMTP. Returns True on success."""
    return await asyncio.to_thread(_send_email_sync, to_email, patient_name, message)


def _format_html_email(patient_name: str, message: str) -> str:
    """Minimal HTML email template."""
    paragraphs = "".join(f"<p>{line}</p>" for line in message.split("\n") if line.strip())
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; padding: 20px;">
        <div style="background: #1a6fa8; padding: 20px; border-radius: 8px 8px 0 0;">
            <h2 style="color: white; margin: 0;">{settings.hospital_name}</h2>
            <p style="color: #d0e8f7; margin: 4px 0 0;">Triage Follow-up Summary</p>
        </div>
        <div style="border: 1px solid #ddd; border-top: none; padding: 24px; border-radius: 0 0 8px 8px;">
            <p>Dear {patient_name},</p>
            {paragraphs}
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="font-size: 12px; color: #888;">
                {settings.hospital_name} &bull; {settings.hospital_address}<br>
                This is an automated message. Do not reply to this email.
            </p>
        </div>
    </body>
    </html>
    """


class FollowupAgent:
    """
    Orchestrates the full post-call follow-up sequence:
      generate summary → send SMS → send email
    """

    def __init__(self, state: TriageState) -> None:
        self.state = state

    async def run(self) -> dict:
        """
        Execute the follow-up sequence.
        Returns a result dict with send status for SMS and email.
        """
        result = {"sms_sent": False, "email_sent": False, "summary": ""}

        # 1. Generate summary
        try:
            summary = await generate_summary(self.state)
            result["summary"] = summary
            logger.info("Follow-up summary generated for room: %s", self.state["room_id"])
        except Exception as exc:
            logger.error("Summary generation failed: %s", exc)
            summary = (
                f"Dear {self.state.get('patient_name', 'Patient')}, "
                "your appointment has been booked. "
                f"Please contact {settings.hospital_name} for details."
            )
            result["summary"] = summary



        # 3. Send email (if email available)
        email = self.state.get("patient_email")
        patient_name = self.state.get("patient_name", "Patient")
        if email:
            result["email_sent"] = await send_email(email, patient_name, summary)

        # 4. Mark follow-up as sent in state
        self.state["followup_sent"] = True

        logger.info(
            "Follow-up complete | room=%s sms=%s email=%s",
            self.state["room_id"],
            result["sms_sent"],
            result["email_sent"],
        )
        return result
