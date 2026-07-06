"""
Seed script — populate the knowledge base with sample documents.

Usage:
    python -m scripts.seed_data
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embedding_service
from src.ingestion.loaders import create_loader
from src.vectorstore.collections import ensure_collection
from src.vectorstore.qdrant_client import get_qdrant_client
from src.vectorstore.search import SearchService


def seed_from_directory(directory: str | Path) -> None:
    """Ingest all supported files from a directory."""
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    settings = get_settings()
    client = get_qdrant_client()
    embedder = get_embedding_service()
    ensure_collection(client)

    search_service = SearchService(client, embedder)

    supported = {".pdf", ".txt", ".md", ".rst", ".csv"}
    files = [f for f in directory.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"No supported files found in {directory}")
        return

    print(f"Found {len(files)} files to ingest")

    total_indexed = 0
    for file_path in files:
        try:
            loader = create_loader(file_path)
            documents = loader.load()
            chunks = chunk_documents(documents)
            indexed = search_service.index_documents(chunks)
            total_indexed += indexed
            print(f"  ✅ {file_path.name}: {len(chunks)} chunks, {indexed} indexed")
        except Exception as e:
            print(f"  ❌ {file_path.name}: {e}")

    print(f"\nTotal indexed: {total_indexed} points")


def seed_sample_texts() -> None:
    """Create and ingest sample documents for testing."""
    from langchain_core.documents import Document

    samples = [
        Document(
            page_content=(
                "LangGraph is a library for building stateful, multi-actor applications "
                "with LLMs. It extends LangChain with the ability to coordinate multiple "
                "chains or actors across multiple steps of computation in a cyclic manner. "
                "It is built on top of LangChain and is designed to handle complex "
                "multi-step workflows that require maintaining state across interactions."
            ),
            metadata={"source": "sample", "type": "text", "topic": "langgraph"},
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation (RAG) is a technique that combines "
                "retrieval of relevant documents with text generation. It enhances "
                "LLM responses by grounding them in specific knowledge bases. "
                "RAG systems typically consist of a retriever (vector search), "
                "a generator (LLM), and sometimes a reranker to improve result quality."
            ),
            metadata={"source": "sample", "type": "text", "topic": "rag"},
        ),
        Document(
            page_content=(
                "Vector databases like Qdrant store high-dimensional embeddings "
                "and enable efficient similarity search using algorithms like HNSW. "
                "They are essential components of modern RAG systems. Qdrant supports "
                "both dense and sparse vectors, filtering, and payload storage. "
                "The free cloud tier provides 1GB of storage."
            ),
            metadata={"source": "sample", "type": "text", "topic": "vectordb"},
        ),
        Document(
            page_content=(
                "Hybrid search combines dense vector search (semantic similarity) "
                "with sparse keyword search (BM25). Reciprocal Rank Fusion (RRF) "
                "merges results from both methods. The formula is: "
                "RRF_score = sum(1 / (k + rank)) where k is typically 60. "
                "This approach is more robust than either method alone."
            ),
            metadata={"source": "sample", "type": "text", "topic": "search"},
        ),
    ]

    settings = get_settings()
    client = get_qdrant_client()
    embedder = get_embedding_service()
    ensure_collection(client)

    search_service = SearchService(client, embedder)
    chunks = chunk_documents(samples)
    indexed = search_service.index_documents(chunks)

    print(f"✅ Seeded {indexed} sample documents")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the knowledge base")
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory of documents to ingest",
        default=None,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Seed with sample documents for testing",
    )

    args = parser.parse_args()

    if args.sample:
        seed_sample_texts()
    elif args.dir:
        seed_from_directory(args.dir)
    else:
        print("Usage: python -m scripts.seed_data --sample")
        print("       python -m scripts.seed_data --dir /path/to/documents")
