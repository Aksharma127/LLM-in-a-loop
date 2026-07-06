"""
Long-term memory — persistent memory stored in a separate Qdrant collection.

Each user/session gets its own namespace within the memory collection.
Memories are embedded and searchable, enabling the agent to recall
relevant past interactions.

SRP: only handles long-term memory storage and retrieval.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from src.config.settings import get_settings
from src.ingestion.embedder import EmbeddingService

logger = structlog.get_logger(__name__)


class LongTermMemory:
    """
    Qdrant-backed persistent memory per session.

    Stores interaction summaries that the agent can search
    to provide contextual continuity across sessions.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingService,
        session_id: str,
    ):
        self._client = client
        self._embedder = embedder
        self._session_id = session_id
        self._collection = get_settings().qdrant_memory_collection

    async def store_memory(
        self,
        content: str,
        memory_type: str = "interaction",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a memory point with session scoping.

        Returns the point ID.
        """
        embedding = self._embedder.embed_query(content)
        point_id = self._content_hash(f"{self._session_id}:{content}")

        payload = {
            "text": content,
            "session_id": self._session_id,
            "memory_type": memory_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(id=point_id, vector=embedding, payload=payload)
            ],
        )

        logger.debug(
            "memory_stored",
            session=self._session_id,
            type=memory_type,
            id=point_id,
        )
        return point_id

    async def recall(
        self,
        query: str,
        top_k: int = 3,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search memories relevant to a query, scoped to session.
        """
        query_vector = self._embedder.embed_query(query)

        must_conditions = [
            FieldCondition(
                key="session_id",
                match=MatchValue(value=self._session_id),
            )
        ]

        if memory_type:
            must_conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_type),
                )
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=Filter(must=must_conditions),
            with_payload=True,
        )
        results = response.points

        return [
            {
                "text": r.payload.get("text", ""),
                "memory_type": r.payload.get("memory_type", ""),
                "timestamp": r.payload.get("timestamp", ""),
                "score": r.score,
            }
            for r in results
        ]

    @staticmethod
    def _content_hash(text: str) -> str:
        hex_digest = hashlib.sha256(text.encode()).hexdigest()
        return str(uuid.UUID(hex_digest[:32]))
