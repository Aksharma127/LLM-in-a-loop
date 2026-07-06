"""
Document ingestion endpoints.

Upload files or provide URLs to add documents to the knowledge base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from src.api.dependencies import get_search_service
from src.ingestion.chunker import chunk_documents
from src.ingestion.loaders import PDFLoader, TextLoader, WebLoader
from src.vectorstore.collections import get_collection_info
from src.vectorstore.qdrant_client import get_qdrant_client

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


class WebIngestRequest(BaseModel):
    """Request body for web page ingestion."""
    url: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class IngestResponse(BaseModel):
    """Response after successful ingestion."""
    source: str
    chunks_created: int
    points_indexed: int


# ── File upload ──────────────────────────────────────────────

@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """
    Upload and ingest a document file (PDF, TXT, MD).

    Steps: load → chunk → embed → index in Qdrant.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".rst", ".csv"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .txt, .md, .rst, .csv",
        )

    logger.info("file_upload_started", filename=file.filename)

    try:
        # Load
        if suffix == ".pdf":
            loader = PDFLoader(stream=file.file)
        else:
            # For text files, read content and create a temp loader
            content = (await file.read()).decode("utf-8")
            # Write to temp path for the TextLoader
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            loader = TextLoader(tmp_path)

        documents = loader.load()

        # Chunk
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)

        # Index
        search_service = get_search_service()
        indexed = search_service.index_documents(chunks)

        logger.info(
            "file_ingested",
            filename=file.filename,
            chunks=len(chunks),
            indexed=indexed,
        )

        return IngestResponse(
            source=file.filename,
            chunks_created=len(chunks),
            points_indexed=indexed,
        )

    except Exception as e:
        logger.error("ingestion_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e!s}")


# ── Web page ingestion ───────────────────────────────────────

@router.post("/web", response_model=IngestResponse)
async def ingest_web_page(request: WebIngestRequest):
    """
    Ingest a web page by URL.

    Fetches the page, extracts text, chunks, embeds, and indexes.
    """
    logger.info("web_ingest_started", url=request.url)

    try:
        loader = WebLoader(request.url)
        documents = loader.load()

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found at the URL",
            )

        chunks = chunk_documents(
            documents, request.chunk_size, request.chunk_overlap
        )

        search_service = get_search_service()
        indexed = search_service.index_documents(chunks)

        return IngestResponse(
            source=request.url,
            chunks_created=len(chunks),
            points_indexed=indexed,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("web_ingestion_failed", url=request.url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Web ingestion failed: {e!s}")


# ── Collection info ──────────────────────────────────────────

@router.get("/status")
async def collection_status():
    """Return information about the document collection."""
    try:
        client = get_qdrant_client()
        from src.config.settings import get_settings
        settings = get_settings()
        info = get_collection_info(client, settings.qdrant_collection)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
