from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Operational tuning knobs — all magic numbers live here, nowhere else.
# ---------------------------------------------------------------------------

class IntakeConfig(BaseModel):
    """
    Tunable parameters for the intake state machine.

    Keeping these in one place means:
      - No scattered literals across nodes.py / agentic.py.
      - Easy A/B testing: swap values without grepping the codebase.
      - Ops can override via environment variables when packaged in Settings.
    """

    # ── Quality gate thresholds (Feature 3) ──────────────────────────────
    # ED patients must supply richer OPQRST detail before advancing because
    # incomplete data in an emergency setting poses a patient-safety risk.
    ed_quality_threshold: float = 0.75
    clinic_quality_threshold: float = 0.60

    # Maximum times the quality gate will ask a gap-fill question before
    # giving up and accepting what was captured.
    max_quality_retries: int = 2

    # ── Conversation history window ───────────────────────────────────────
    # Only the most recent N messages are included in LLM prompts.
    # Prevents token explosion on long intakes (>30 turns) and keeps costs
    # predictable.  Older messages remain in the DB for audit; they are just
    # not sent to the LLM.
    messages_window_size: int = 30

    # ── Session lifecycle ─────────────────────────────────────────────────
    session_ttl_hours: int = 4
    checkpoint_retention_days: int = 30

    # ── Session turns guard ───────────────────────────────────────────────
    # Hard cap on user messages per session. Prevents infinite loops and
    # runaway LLM costs. Patient is directed to the front desk when hit.
    max_session_turns: int = 30
    # Hard cap on identity collection attempts before giving up and directing
    # the patient to the front desk.
    max_identity_attempts: int = 6

    # ── Report generation ─────────────────────────────────────────────────
    max_report_attempts: int = 3
    # Use the deterministic template for clinician notes by default.
    # The LLM adds natural-language flow to the HPI narrative but the output
    # format is fully prescribed — the template produces an equivalent note at
    # zero LLM cost (~32% of per-session spend).  Set to True only when prose
    # quality is a hard requirement for the deployment context.
    use_llm_report_narrative: bool = False

    # ── Cost controls ─────────────────────────────────────────────────────
    # Gemini Flash pricing (per million tokens).  Update here when Anthropic
    # revises the pricing tier — both llm.py and patient.py read from these
    # so there is exactly one place to change.
    gemini_input_cost_per_million: float = 0.075
    gemini_output_cost_per_million: float = 0.30
    # Alert threshold for the repair rate over the last hour.
    # repair_used fires when the primary LLM call returns invalid JSON and a
    # second call is made — doubling cost for that call.  A rate above this
    # threshold is logged as a warning in /analytics.
    repair_rate_alert_threshold: float = 0.05
    # Hard cost cap per session. Once accumulated LLM spend exceeds this,
    # the session routes to done with a front-desk message rather than making
    # further LLM calls.  Default is ~8× a normal session (~$0.0012 typical).
    max_cost_usd_per_session: float = 0.01

    # ── Webhook dead-letter ───────────────────────────────────────────────
    # Exhausted deliveries are eligible for re-queue after this many hours.
    dead_letter_retry_after_hours: int = 24
    # Max total lifetime retries for a dead-lettered delivery (prevents
    # permanent retry storms against a broken downstream endpoint).
    dead_letter_max_lifetime_attempts: int = 6


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

    cors_allowed_origins: list[str] = ["*"]
    require_consent: bool = True

    slack_webhook_url: str = ""
    completion_webhook_url: str = ""
    completion_webhook_secret: str = ""
    webhook_max_attempts: int = 3

    llm_timeout_seconds: float = 15.0

    fhir_server_url: str = ""
    fhir_server_bearer_token: str = ""

    # Nested operational config.  Individual sub-fields can be overridden via
    # env vars using the Pydantic nested-model convention when needed.
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
