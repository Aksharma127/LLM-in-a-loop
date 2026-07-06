"""
Redis response cache.

Same query → same answer → no repeat LLM call.
Also stores conversation buffers for stateless workers.

SRP: only handles caching operations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


class CacheService:
    """
    Redis-backed caching for LLM responses and session state.

    Gracefully degrades if Redis is unavailable — cache misses
    just result in a fresh LLM call.
    """

    def __init__(self, redis_url: str | None = None):
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._ttl = settings.cache_ttl_seconds
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis | None:
        """Lazy-connect to Redis, return None if unavailable."""
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    self._redis_url,
                    decode_responses=True,
                )
                await self._client.ping()
                logger.info("redis_connected", url=self._redis_url)
            except Exception as e:
                logger.warning("redis_unavailable", error=str(e))
                self._client = None
        return self._client

    # ── Response cache ────────────────────────────────────────

    async def get_cached_response(self, query: str) -> str | None:
        """Look up a cached LLM response for a query."""
        client = await self._get_client()
        if client is None:
            return None

        key = f"response:{self._query_hash(query)}"
        try:
            cached = await client.get(key)
            if cached:
                logger.debug("cache_hit", query_prefix=query[:50])
            return cached
        except Exception:
            return None

    async def cache_response(self, query: str, response: str) -> None:
        """Cache an LLM response."""
        client = await self._get_client()
        if client is None:
            return

        key = f"response:{self._query_hash(query)}"
        try:
            await client.setex(key, self._ttl, response)
            logger.debug("response_cached", query_prefix=query[:50])
        except Exception as e:
            logger.warning("cache_write_failed", error=str(e))

    # ── Session storage ───────────────────────────────────────

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist a conversation buffer to Redis."""
        client = await self._get_client()
        if client is None:
            return

        key = f"session:{session_id}"
        try:
            await client.setex(key, self._ttl * 24, json.dumps(data))
        except Exception as e:
            logger.warning("session_save_failed", error=str(e))

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a conversation buffer from Redis."""
        client = await self._get_client()
        if client is None:
            return None

        key = f"session:{session_id}"
        try:
            data = await client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    # ── Health ────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Return True if Redis is reachable."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            return await client.ping()
        except Exception:
            return False

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic hash for cache key."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
