"""
Centralized configuration via Pydantic Settings.

Reads from .env file and environment variables.
All secrets and tunables live here — never scattered across modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings. Immutable after creation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ───────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LLM Providers ────────────────────────────────────────
    groq_api_key: str = ""

    # ── LLM Router ───────────────────────────────────────────
    fast_model: str = "llama-3.1-8b-instant"
    strong_model: str = "llama-3.3-70b-versatile"

    # ── Embeddings ───────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_dimension: int = 384  # MiniLM-L6-v2 output dim

    # ── Qdrant ───────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    qdrant_memory_collection: str = "memory"

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600  # 1 hour default

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ────────────────────────────────────────────
    retrieval_top_k: int = 5
    rerank_top_k: int = 3

    @property
    def is_qdrant_cloud(self) -> bool:
        return self.qdrant_api_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings factory — cached after first call."""
    return Settings()
