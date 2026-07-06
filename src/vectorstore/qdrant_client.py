"""
Qdrant connection manager.

SRP: only manages connection lifecycle.
Supports both local Docker and Qdrant Cloud transparently.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from qdrant_client import QdrantClient

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


def create_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client configured from settings.

    Automatically detects local vs cloud based on whether
    an API key is present.
    """
    settings = get_settings()

    kwargs: dict = {"url": settings.qdrant_url, "timeout": 30}
    if settings.is_qdrant_cloud:
        kwargs["api_key"] = settings.qdrant_api_key
        logger.info("qdrant_connecting", mode="cloud", url=settings.qdrant_url)
    else:
        logger.info("qdrant_connecting", mode="local", url=settings.qdrant_url)

    client = QdrantClient(**kwargs)
    logger.info("qdrant_connected")
    return client


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Singleton Qdrant client — one connection for the app lifecycle."""
    return create_qdrant_client()
