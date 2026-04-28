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
    max_identity_attempts: int = 3

    # ── Report generation ─────────────────────────────────────────────────
    max_report_attempts: int = 3
    # Use the deterministic template for clinician notes by default.
    # The LLM adds natural-language flow to the HPI narrative but the output
    # format is fully prescribed — the template produces an equivalent note at
    # zero LLM cost (~32% of per-session spend).  Set to True only when prose
    # quality is a hard requirement for the deployment context.
    use_llm_report_narrative: bool = True

    # ── Clinical list extraction ─────────────────────────────────────────
    # When True (default), allergies / PMH / recent_results are extracted
    # via run_json_step + ListExtractOut so multi-clause sentences like
    # "yes I'm allergic to peanuts and pollen" parse into clean entries.
    # When False, we fall back to the legacy regex parsers
    # (extract_allergies_simple / extract_list_simple) — useful for cost
    # comparison, A/B rollout, or as an emergency switch if the LLM list
    # prompt regresses in production.
    use_llm_list_extractor: bool = True

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
    max_cost_usd_per_session: float = 0.05

    # ── Webhook dead-letter ───────────────────────────────────────────────
    # Exhausted deliveries are eligible for re-queue after this many hours.
    dead_letter_retry_after_hours: int = 24
    # Max total lifetime retries for a dead-lettered delivery (prevents
    # permanent retry storms against a broken downstream endpoint).
    dead_letter_max_lifetime_attempts: int = 6

    # ── Voice transcription (Groq Whisper) ────────────────────────────────
    # Voice is a thin shell around /chat: the audio→text leg lives at
    # /transcribe, and the resulting text flows through the existing /chat
    # pipeline, so all chat-side guardrails (idempotency, cost cap, prompt
    # injection check, PHI masking, max turns) apply automatically.
    groq_stt_model: str = "whisper-large-v3-turbo"
    # 2 MB cap on a single audio upload — generous for a 30s webm clip
    # (typical sizes are 100-400 KB), tight enough to bound abuse.
    transcribe_max_bytes: int = 2_000_000
    # Hard duration ceiling.  The client auto-stops at this; the server
    # rejects payloads larger than transcribe_max_bytes regardless.
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
    # Clinician JWT lifetime.  Default 1h: a stolen token is valid only for
    # the rest of one shift handover, not a full day.  Override via
    # CLINICIAN_TOKEN_LIFETIME_SECONDS if your workflow requires longer.
    clinician_token_lifetime_seconds: int = 3600

    # Empty by default — production deployments must set CORS_ALLOWED_ORIGINS
    # explicitly.  The bundled frontend is served same-origin from /static, so
    # local dev does not need a permissive default.  A wildcard default is
    # incompatible with cookies/credentials and a poor first impression for any
    # security review.
    cors_allowed_origins: list[str] = []
    require_consent: bool = True

    slack_webhook_url: str = ""
    completion_webhook_url: str = ""
    completion_webhook_secret: str = ""
    webhook_max_attempts: int = 3

    llm_timeout_seconds: float = 15.0

    # Background task pool — sizes the executor that runs report generation,
    # webhook dispatch, and FHIR pushes.  Lifespan shutdown drains the pool,
    # so work in flight at SIGTERM completes (or hits its own timeout) instead
    # of being killed mid-write.
    background_workers: int = 8
    background_shutdown_timeout: float = 30.0

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 60.0

    fhir_server_url: str = ""
    fhir_server_bearer_token: str = ""

    # EHR FHIR endpoint for patient identity lookup at intake start.
    # Default: HAPI FHIR public test server (no auth, works for demos).
    # Production: set to Epic/Cerner base URL + ehr_fhir_bearer_token.
    ehr_fhir_url: str = "https://hapi.fhir.org/baseR4"
    ehr_fhir_bearer_token: str = ""

    # Voice transcription provider.  Empty key disables /transcribe (returns
    # 503) and the patient UI hides the mic button.  Kept at the top level
    # rather than in IntakeConfig because it is a credential, not a tuning knob.
    groq_api_key: str = ""
    voice_enabled: bool = True

    # RxNorm terminology service — production-grade drug-name canonicalisation
    # via NLM's free public API.  When disabled, normalize_drug_name returns
    # the patient phrase unchanged (acceptable for offline / network-isolated
    # dev environments; tests monkey-patch the lookup module so they never
    # depend on the network).
    rxnorm_enabled: bool = True
    rxnorm_timeout_seconds: float = 3.0
    rxnorm_cache_ttl_days: int = 30

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
