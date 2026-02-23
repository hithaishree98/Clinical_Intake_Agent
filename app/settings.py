from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_db_path: str = str(BASE_DIR / "data" / "app.db")
    checkpoint_db_path: str = str(BASE_DIR / "data" / "checkpoints.db")

    gemini_api_key: str
    gemini_flash_model: str = "gemini-3-flash-preview"

    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 10.0

    debug_mode: bool = False
    max_upload_size_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()