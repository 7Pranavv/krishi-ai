"""Application settings.

Everything that differs between a laptop and production lives here and is read
from the environment. No secret has a usable default: JWT_SECRET is required in
production and the app refuses to start without it.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- environment -------------------------------------------------
    environment: str = Field("development", alias="ENVIRONMENT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ---- http --------------------------------------------------------
    host: str = Field("127.0.0.1", alias="HOST")
    port: int = Field(8000, alias="PORT")
    # Comma-separated browser origins allowed to call the API. The bundled
    # frontend is same-origin, so this is only needed when hosting it apart.
    cors_origins: str = Field("", alias="CORS_ORIGINS")

    # ---- database ----------------------------------------------------
    # SQLite by default so a fresh clone runs with no external services.
    # Point at postgresql+psycopg://user:pass@host/db for production.
    database_url: str = Field(f"sqlite:///{REPO_DIR / 'krishi.db'}", alias="DATABASE_URL")

    # ---- auth --------------------------------------------------------
    jwt_secret: str = Field("", alias="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    access_token_days: int = Field(30, alias="ACCESS_TOKEN_DAYS")

    # ---- ml service --------------------------------------------------
    ml_service_url: str = Field("http://127.0.0.1:8100", alias="ML_SERVICE_URL")
    ml_timeout_seconds: float = Field(30.0, alias="ML_TIMEOUT_SECONDS")
    # Disease inference is much slower than the tabular models.
    ml_upload_timeout_seconds: float = Field(90.0, alias="ML_UPLOAD_TIMEOUT_SECONDS")
    max_upload_bytes: int = Field(8 * 1024 * 1024, alias="MAX_UPLOAD_BYTES")

    # ---- market prices -----------------------------------------------
    # Free key from https://data.gov.in/apis . Without it the market pages
    # report themselves unavailable; nothing else is affected.
    data_gov_api_key: str = Field("", alias="DATA_GOV_API_KEY")
    market_api_url: str = Field(
        "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
        alias="MARKET_API_URL",
    )

    # ---- ai assistant ------------------------------------------------
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    chat_history_turns: int = Field(10, alias="CHAT_HISTORY_TURNS")
    chat_max_chars: int = Field(2000, alias="CHAT_MAX_CHARS")

    @field_validator("environment")
    @classmethod
    def _known_env(cls, value: str) -> str:
        return value.lower()

    @field_validator("database_url")
    @classmethod
    def _absolute_sqlite_path(cls, value: str) -> str:
        """Anchor a relative SQLite path to the repo root, not the cwd.

        `sqlite:///./krishi.db` otherwise resolves against whatever directory
        uvicorn was started from, so launching from backend/ and from the repo
        root silently use two different databases - one of them missing all the
        seeded data. Absolute paths and non-SQLite URLs pass through untouched.
        """
        prefix = "sqlite:///"
        if not value.startswith(prefix):
            return value
        raw = value[len(prefix) :]
        if not raw or raw.startswith("/") or Path(raw).is_absolute():
            return value  # absolute path, or the :memory: form
        return f"{prefix}{(REPO_DIR / raw).resolve()}"

    @property
    def is_production(self) -> bool:
        return self.environment in {"production", "prod"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def ai_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def market_enabled(self) -> bool:
        return bool(self.data_gov_api_key)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if not settings.jwt_secret:
        if settings.is_production:
            raise RuntimeError(
                "JWT_SECRET must be set in production. "
                "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\""
            )
        # Development convenience: a per-process key. Restarting invalidates
        # existing tokens, which is the correct trade-off for a dev default.
        settings.jwt_secret = secrets.token_urlsafe(48)
    return settings
