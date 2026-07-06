"""
Tests for the vector store — search service, collections.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vectorstore.search import SearchService


class TestSearchService:
    """Test the hybrid search service."""

    def test_content_hash_deterministic(self):
        """Same content should produce same hash."""
        hash1 = SearchService._content_hash("hello world")
        hash2 = SearchService._content_hash("hello world")
        assert hash1 == hash2

    def test_content_hash_different(self):
        """Different content should produce different hashes."""
        hash1 = SearchService._content_hash("hello")
        hash2 = SearchService._content_hash("world")
        assert hash1 != hash2

    def test_empty_embed_returns_empty(self):
        """Embedding an empty list should return empty."""
        mock_client = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = []

        service = SearchService(mock_client, mock_embedder)
        result = service.index_documents([])
        assert result == 0
