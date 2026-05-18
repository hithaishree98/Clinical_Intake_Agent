from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Operational tuning knobs — all magic numbers live here, nowhere else.
# ---------------------------------------------------------------------------

class IntakeConfig(BaseModel):
    # ── Quality gate thresholds ───────────────────────────────────────────
    ed_quality_threshold: float = 0.75
    clinic_quality_threshold: float = 0.60
    max_quality_retries: int = 2

    # ── Conversation history window ───────────────────────────────────────
    # Only the most recent N messages are sent to the LLM; older turns stay in DB.
    messages_window_size: int = 30

    # ── Session lifecycle ─────────────────────────────────────────────────
    session_ttl_hours: int = 4
    checkpoint_retention_days: int = 30

    # ── Session turns guard ───────────────────────────────────────────────
    # Hard cap per session — prevents runaway LLM cost.
    max_session_turns: int = 30
    max_identity_attempts: int = 3

    # ── Report generation ─────────────────────────────────────────────────
    max_report_attempts: int = 3
    use_llm_report_narrative: bool = True

    # ── Clinical list extraction ─────────────────────────────────────────
    # False falls back to regex parsers — useful for cost comparison or emergency rollback.
    use_llm_list_extractor: bool = True

    # ── Cost controls ─────────────────────────────────────────────────────
    # Gemini Flash pricing (per million tokens).
    gemini_input_cost_per_million: float = 0.075
    gemini_output_cost_per_million: float = 0.30
    # Repair rate alert: fires when >5% of calls need a second LLM pass.
    repair_rate_alert_threshold: float = 0.05
    # Hard cost cap per session (~8× typical $0.0012).
    max_cost_usd_per_session: float = 0.05

    # ── Webhook dead-letter ───────────────────────────────────────────────
    dead_letter_retry_after_hours: int = 24
    # Cap lifetime retries to prevent permanent retry storms.
    dead_letter_max_lifetime_attempts: int = 6

    # ── Voice transcription (Groq Whisper) ────────────────────────────────
    groq_stt_model: str = "whisper-large-v3-turbo"
    transcribe_max_bytes: int = 2_000_000   # 2 MB; typical 30s clip is 100-400 KB
    transcribe_max_seconds: int = 30


class Settings(BaseSettings):
    app_db_path: str = str(BASE_DIR / "data" / "app.db")
    checkpoint_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")

    gemini_api_key: str = "dev-placeholder-key"
    gemini_flash_model: str = "gemini-2.5-flash-lite"

    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 10.0

    debug_mode: bool = False

    jwt_secret: str = "CHANGE_ME_jwt_secret"
    clinician_password: str = "CHANGE_ME_password"
    # 1 h default: a stolen token expires within one shift.
    clinician_token_lifetime_seconds: int = 3600

    # Must be set explicitly in production; empty default is intentional.
    cors_allowed_origins: list[str] = []
    require_consent: bool = True

    slack_webhook_url: str = ""
    completion_webhook_url: str = ""
    completion_webhook_secret: str = ""
    webhook_max_attempts: int = 3

    llm_timeout_seconds: float = 15.0

    background_workers: int = 8
    background_shutdown_timeout: float = 30.0

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 60.0

    fhir_server_url: str = ""
    fhir_server_bearer_token: str = ""

    # Default: HAPI FHIR public test server. Production: set to Epic/Cerner base URL.
    ehr_fhir_url: str = "https://hapi.fhir.org/baseR4"
    ehr_fhir_bearer_token: str = ""

    # Empty disables /transcribe (503) and hides the mic button.
    groq_api_key: str = ""
    voice_enabled: bool = True

    rxnorm_enabled: bool = True
    rxnorm_timeout_seconds: float = 3.0
    rxnorm_cache_ttl_days: int = 30

    intake: IntakeConfig = IntakeConfig()

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_KEY: ClassVar[str] = "your_gemini_api_key_here"

    @field_validator("gemini_api_key")
    @classmethod
    def api_key_must_be_set(cls, v: str) -> str:
        if not v:
            return "dev-placeholder-key"
        if v.strip() == cls.API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        return v

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_set(cls, v: str) -> str:
        if not v or v.strip() == "CHANGE_ME_jwt_secret":
            raise ValueError(
                "JWT_SECRET is not set. Add a strong random value to your .env file: JWT_SECRET=..."
            )
        return v

    @field_validator("clinician_password")
    @classmethod
    def clinician_password_must_be_set(cls, v: str) -> str:
        if not v or v.strip() == "CHANGE_ME_password":
            raise ValueError(
                "CLINICIAN_PASSWORD is not set. Add it to your .env file: CLINICIAN_PASSWORD=..."
            )
        return v


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
