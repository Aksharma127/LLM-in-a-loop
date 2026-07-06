"""
Text chunking strategies.

SRP: this module only splits documents into chunks.
Uses LangChain's RecursiveCharacterTextSplitter for smart boundary detection.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config.settings import get_settings


def create_chunker(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with configured defaults."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into smaller, embedding-friendly chunks.

    Each chunk preserves the parent document's metadata plus
    a `chunk_index` field for ordering.
    """
    splitter = create_chunker(chunk_size, chunk_overlap)
    chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_index"] = i
            split.metadata["total_chunks"] = len(splits)
            chunks.append(split)

    return chunks
