"""
TriageAI configuration — loaded once via lru_cache.
All values come from environment variables (or .env file).
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LiveKit ---
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # --- STT / TTS ---
    groq_api_key: str = ""
    cartesia_api_key: str = ""

    # --- LLM (Cerebras / Mistral) ---
    cerebras_api_key: str = ""
    mistral_api_key: str = ""

    # --- Vector DB ---
    pinecone_api_key: str = ""
    pinecone_index_name: str = "triage-medical-kb"
    pinecone_environment: str = "us-east-1"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://triage:triage_password@localhost:5432/triageai"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"

    # --- Gmail SMTP ---
    gmail_address: str = ""
    gmail_app_passwprd: str = ""

    # --- LangSmith ---
    langchain_api_key: str = ""
    langchain_project: str = "triageai"
    langchain_tracing_v2: str = "true"

    # --- App ---
    environment: str = "development"
    log_level: str = "INFO"
    hospital_name: str = "General Hospital"
    hospital_address: str = "123 Medical Drive"

    # --- Agent Models ---
    intake_model: str = "mistral-small-latest"
    analyzer_model: str = "mistral-large-latest"
    router_model: str = "mistral-large-latest"
    followup_model: str = "mistral-small-latest"
    embedding_model: str = "mistral-embed"

    # --- Urgency Thresholds ---
    # Urgency score range: 1 (ER / life-threatening) → 5 (routine GP)
    # Scores at or below this value → EMERGENCY path
    emergency_threshold: int = 2

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton. Call get_settings() everywhere."""
    return Settings()
