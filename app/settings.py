from pathlib import Path
from typing import ClassVar
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────
    app_db_path:        str = str(BASE_DIR / "data" / "app.db")
    checkpoint_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")

    # ── Gemini ────────────────────────────────────────────────────────
    gemini_api_key:     str
    gemini_flash_model: str = "gemini-2.0-flash"

    # ── Retry / reliability ───────────────────────────────────────────
    max_retries:       int   = 3
    base_retry_delay:  float = 1.0
    max_retry_delay:   float = 10.0

    # ── App behaviour ─────────────────────────────────────────────────
    debug_mode:         bool = False
    max_upload_size_mb: int  = 10

    # ── Auth ──────────────────────────────────────────────────────────
    jwt_secret:          str = "CHANGE_ME_jwt_secret"
    clinician_password:  str = "CHANGE_ME_password"

    # ── Consent gate ─────────────────────────────────────────────────
    # When True, patients must explicitly consent before intake begins.
    # Disable only for internal testing / demo environments.
    require_consent: bool = True

    # ── Notifications: ntfy.sh (free, open-source push notifications) ─
    # Get a topic at https://ntfy.sh — no account needed.
    # Subscribe via browser or the ntfy mobile app (iOS/Android).
    # Emergency alerts use priority=urgent (bypasses DND on mobile app).
    # Leave blank to disable.
    ntfy_topic: str = ""

    # ── Notifications: Discord webhook ────────────────────────────────
    # In Discord: Edit Channel → Integrations → Webhooks → New Webhook → Copy URL
    # Emergency alerts appear as red embeds; completions as green.
    # Leave blank to disable.
    discord_webhook_url: str = ""

    # ── Notifications: Slack ──────────────────────────────────────────
    # Existing integration — https://api.slack.com/messaging/webhooks
    slack_webhook_url: str = ""

    # ── Notifications: Generic FHIR webhook ──────────────────────────
    # POST of the FHIR R4 Bundle, signed with HMAC-SHA256.
    # For local testing: set to https://webhook.site/your-uuid (free, no signup).
    # Leave blank to disable.
    completion_webhook_url:    str = ""
    completion_webhook_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_KEY:ClassVar[str] = "api key here"

    @field_validator("gemini_api_key")
    @classmethod
    def api_key_must_be_set(cls, v: str) -> str:
        if not v or v.strip() == cls.API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        return v


settings = Settings()