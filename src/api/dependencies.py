"""
Dependency injection and application lifespan management.

Wires up all services once at startup, tears them down at shutdown.
All route handlers get their dependencies from here — never
construct services inline.

Dependency Inversion: routes depend on abstractions provided here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI

from src.agents.graph import compile_agent_graph
from src.config.settings import get_settings
from src.ingestion.embedder import get_embedding_service
from src.llm.router import LLMRouter
from src.memory.cache import CacheService
from src.vectorstore.collections import ensure_collection
from src.vectorstore.qdrant_client import get_qdrant_client
from src.vectorstore.search import SearchService

logger = structlog.get_logger(__name__)

# ── Service registry (set during lifespan) ───────────────────

_services: dict[str, Any] = {}


def get_search_service() -> SearchService:
    return _services["search_service"]


def get_agent_graph():
    return _services["agent_graph"]


def get_cache_service() -> CacheService:
    return _services["cache_service"]


def get_llm_router() -> LLMRouter:
    return _services["llm_router"]


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan — initialize all services at startup.

    This is where the dependency graph is wired:
    Settings → Qdrant → Embedder → SearchService → AgentGraph
    """
    settings = get_settings()
    logger.info("app_starting", env=settings.app_env)

    # 1. Qdrant client
    qdrant = get_qdrant_client()

    # 2. Ensure collections exist
    ensure_collection(qdrant, settings.qdrant_collection)
    ensure_collection(qdrant, settings.qdrant_memory_collection)

    # 3. Embedding service (lazy-loads model on first use)
    embedder = get_embedding_service()

    # 4. Search service
    search_service = SearchService(
        client=qdrant,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
    )

    # 5. LLM router
    llm_router = LLMRouter()

    # 6. Agent graph
    agent_graph = compile_agent_graph(
        router=llm_router,
        search_service=search_service,
    )

    # 7. Cache service
    cache_service = CacheService()

    # Register all services
    _services.update({
        "search_service": search_service,
        "agent_graph": agent_graph,
        "cache_service": cache_service,
        "llm_router": llm_router,
    })

    logger.info("app_started", services=list(_services.keys()))

    yield

    # Cleanup
    logger.info("app_shutting_down")
    _services.clear()
