"""
Hybrid search — dense vector + BM25 keyword search with score fusion.

This is the core retrieval engine. It combines:
1. Dense (semantic) search via Qdrant's ANN index
2. Sparse (keyword) search via BM25 in-memory
3. Reciprocal Rank Fusion (RRF) to merge results

SRP: only handles search and indexing of vectors/points.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import structlog
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
)
from rank_bm25 import BM25Okapi

from src.config.settings import get_settings
from src.ingestion.embedder import EmbeddingService

logger = structlog.get_logger(__name__)


class SearchService:
    """
    Hybrid search service combining dense + sparse retrieval.

    Dependency Inversion: takes EmbeddingService and QdrantClient
    as constructor dependencies — testable and swappable.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingService,
        collection_name: str | None = None,
    ):
        self._client = client
        self._embedder = embedder
        self._settings = get_settings()
        self._collection = collection_name or self._settings.qdrant_collection

        # BM25 index — rebuilt from stored documents
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict[str, Any]] = []

    # ── Indexing ──────────────────────────────────────────────

    def index_documents(self, documents: list[Document]) -> int:
        """
        Embed and upsert documents into Qdrant.

        Returns the number of points upserted.
        """
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self._embedder.embed_texts(texts)

        points = []
        for doc, embedding in zip(documents, embeddings):
            # Deterministic ID from content hash — idempotent upserts
            point_id = self._content_hash(doc.page_content)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )

        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
            )

        logger.info(
            "documents_indexed",
            collection=self._collection,
            count=len(points),
        )

        # Invalidate BM25 cache — it needs rebuilding
        self._bm25 = None
        return len(points)

    # ── Dense search ──────────────────────────────────────────

    def dense_search(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search using query embedding against Qdrant ANN index."""
        k = top_k or self._settings.retrieval_top_k
        query_vector = self._embedder.embed_query(query)

        # Optional source filter
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        results: list[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "metadata": r.payload.get("metadata", {}),
                "score": r.score,
                "id": r.id,
                "search_type": "dense",
            }
            for r in results
        ]

    # ── BM25 sparse search ───────────────────────────────────

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the collection."""
        # Scroll through all points
        all_points, _ = self._client.scroll(
            collection_name=self._collection,
            limit=10000,
            with_payload=True,
        )

        self._bm25_docs = []
        corpus: list[list[str]] = []

        for point in all_points:
            text = point.payload.get("text", "")
            self._bm25_docs.append(
                {
                    "text": text,
                    "metadata": point.payload.get("metadata", {}),
                    "id": point.id,
                }
            )
            corpus.append(text.lower().split())

        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

        logger.debug("bm25_index_built", doc_count=len(corpus))

    def sparse_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Keyword search using BM25."""
        if self._bm25 is None:
            self._build_bm25_index()

        if self._bm25 is None or not self._bm25_docs:
            return []

        k = top_k or self._settings.retrieval_top_k
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                **self._bm25_docs[i],
                "score": float(scores[i]),
                "search_type": "sparse",
            }
            for i in top_indices
            if scores[i] > 0
        ]

    # ── Hybrid search with RRF ───────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Combine dense + sparse search using Reciprocal Rank Fusion.

        RRF score = Σ (weight / (rank + k)) where k=60 (standard constant).
        This is more robust than raw score interpolation because it's
        scale-invariant across the two scoring functions.
        """
        k = top_k or self._settings.retrieval_top_k
        rrf_k = 60  # Standard RRF constant

        dense_results = self.dense_search(query, top_k=k * 2)
        sparse_results = self.sparse_search(query, top_k=k * 2)

        # Build RRF scores keyed by text content
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(dense_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + dense_weight / (rank + rrf_k)
            result_map[text] = result

        for rank, result in enumerate(sparse_results):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + sparse_weight / (rank + rrf_k)
            if text not in result_map:
                result_map[text] = result

        # Sort by RRF score and return top-k
        sorted_texts = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        return [
            {
                **result_map[text],
                "score": rrf_scores[text],
                "search_type": "hybrid",
            }
            for text in sorted_texts
        ]

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        """Deterministic UUID from content — ensures idempotent upserts."""
        hex_digest = hashlib.sha256(text.encode()).hexdigest()
        return str(uuid.UUID(hex_digest[:32]))
