"""Central configuration. Reads from environment / .env, never hard-code secrets."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SEC EDGAR requires a User-Agent with a real name + email, or it returns 403.
    sec_user_agent: str = "Sightline Dev you@example.com"

    qdrant_url: str = "http://localhost:6333"
    data_dir: Path = Path("./data")

    # LLM / VLM provider (OpenAI-compatible base URL works with most providers).
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "your-chat-model"
    vlm_model: str = "your-vision-model"
    router_model: str = "your-small-model"
    # Minimum seconds between LLM requests (free tiers cap requests/minute).
    # 0 = no throttle. `:free`-suffixed models always get a floor of 4s regardless.
    llm_min_interval_s: float = 0.0

    # Observability (optional). Blank = tracing is a no-op.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    monthly_budget_usd: float = 10.0

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings()
