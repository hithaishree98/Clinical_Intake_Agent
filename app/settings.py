from pathlib import Path
from typing import ClassVar
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_db_path: str = str(BASE_DIR / "data" / "app.db")
    checkpoint_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")

    gemini_api_key: str
    gemini_flash_model: str = "gemini-2.0-flash"

    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 10.0

    debug_mode: bool = False
    max_upload_size_mb: int = 10

    jwt_secret: str = "Replace"
    clinician_password: str = "Replace"

    # Optional outbound webhook fired when an intake report is ready.
    # If set, the FHIR Bundle is POSTed here with an HMAC-SHA256 signature.
    completion_webhook_url: str = ""
    completion_webhook_secret: str = ""

    # Slack incoming webhook URL for emergency alerts and intake completion.
    # Get one at: https://api.slack.com/messaging/webhooks — free, no library needed.
    # Leave blank to disable.
    slack_webhook_url: str = ""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_KEY: ClassVar[str] = "api key here"

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
