"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    agent_llm_provider: str = "scripted"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    web_search_provider: str = "ddgs"
    tavily_api_key: str | None = None
    max_llm_calls: int = Field(default=10, ge=1)
    max_task_cost_usd: float = Field(default=0.20, gt=0)
    local_model_price_per_1k_tokens: float = Field(default=0.01, gt=0)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_completion_tokens: int = Field(default=400, ge=64, le=2000)
    tool_timeout_seconds: float = Field(default=8.0, gt=0)
    max_agent_steps: int = Field(default=10, ge=1)
