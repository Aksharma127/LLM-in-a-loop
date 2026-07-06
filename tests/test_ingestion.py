"""
Tests for the ingestion pipeline — loaders, chunker, embedder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import PDFLoader, TextLoader, WebLoader, create_loader


class TestTextLoader:
    """Test plain text document loading."""

    def test_load_text_file(self, tmp_path: Path):
        """Should load a .txt file and return a Document."""
        file = tmp_path / "test.txt"
        file.write_text("Hello, this is a test document.\nWith multiple lines.")

        loader = TextLoader(file)
        docs = loader.load()

        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
        assert docs[0].metadata["type"] == "text"
        assert docs[0].metadata["filename"] == "test.txt"

    def test_source_id(self, tmp_path: Path):
        """Should return the file path as source ID."""
        file = tmp_path / "doc.md"
        file.write_text("# Markdown")

        loader = TextLoader(file)
        assert str(file) == loader.source_id()


class TestLoaderFactory:
    """Test auto-detection factory."""

    def test_creates_text_loader(self, tmp_path: Path):
        """Should create TextLoader for .txt files."""
        file = tmp_path / "test.txt"
        file.write_text("content")

        loader = create_loader(file)
        assert isinstance(loader, TextLoader)

    def test_creates_web_loader(self):
        """Should create WebLoader for URLs."""
        loader = create_loader("https://example.com")
        assert isinstance(loader, WebLoader)

    def test_raises_for_unsupported(self, tmp_path: Path):
        """Should raise ValueError for unsupported types."""
        file = tmp_path / "test.xyz"
        file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported"):
            create_loader(file)

    def test_raises_for_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            create_loader("/nonexistent/file.txt")


class TestChunker:
    """Test document chunking."""

    def test_chunks_long_document(self, tmp_path: Path):
        """Should split a long document into multiple chunks."""
        from langchain_core.documents import Document

        long_text = "This is a sentence. " * 200  # ~4000 chars
        doc = Document(
            page_content=long_text,
            metadata={"source": "test", "type": "text"},
        )

        chunks = chunk_documents([doc], chunk_size=256, chunk_overlap=32)

        assert len(chunks) > 1
        # Each chunk should have metadata
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["source"] == "test"

    def test_short_document_single_chunk(self):
        """Short documents should remain as a single chunk."""
        from langchain_core.documents import Document

        doc = Document(page_content="Short text.", metadata={"source": "test"})
        chunks = chunk_documents([doc])

        assert len(chunks) == 1
