"""
Collection management — CRUD operations for Qdrant collections.

SRP: only handles collection lifecycle (create, delete, info).
"""

from __future__ import annotations

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


def ensure_collection(
    client: QdrantClient,
    collection_name: str | None = None,
    dimension: int | None = None,
) -> None:
    """
    Create collection if it doesn't exist.

    Idempotent — safe to call on every startup.
    """
    settings = get_settings()
    name = collection_name or settings.qdrant_collection
    dim = dimension or settings.embedding_dimension

    try:
        info = client.get_collection(name)
        logger.info(
            "collection_exists",
            name=name,
            points=info.points_count,
        )
    except (UnexpectedResponse, Exception):
        logger.info("creating_collection", name=name, dimension=dim)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("collection_created", name=name)


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """Delete a collection by name."""
    client.delete_collection(collection_name)
    logger.info("collection_deleted", name=collection_name)


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Return collection metadata as a dict."""
    info = client.get_collection(collection_name)
    return {
        "name": collection_name,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": info.status.value if info.status else "unknown",
    }
