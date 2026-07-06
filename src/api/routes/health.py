"""
Health check endpoint — always the first route you build.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_cache_service
from src.config.settings import get_settings
from src.vectorstore.qdrant_client import get_qdrant_client

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Application health check.

    Returns service status for Qdrant, Redis, and the API itself.
    """
    settings = get_settings()

    # Check Qdrant
    qdrant_ok = False
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    # Check Redis
    cache = get_cache_service()
    redis_ok = await cache.health_check()

    return {
        "status": "healthy" if qdrant_ok else "degraded",
        "environment": settings.app_env,
        "services": {
            "qdrant": "connected" if qdrant_ok else "unavailable",
            "redis": "connected" if redis_ok else "unavailable",
            "api": "running",
        },
    }
