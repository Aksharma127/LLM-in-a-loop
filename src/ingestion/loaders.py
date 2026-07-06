"""
Document loaders — SRP: each loader handles exactly one source type.

All loaders implement the BaseLoader ABC so they're interchangeable
(Liskov Substitution) and the ingestion pipeline depends on the
abstraction, not on PDF/web specifics (Dependency Inversion).
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader

logger = structlog.get_logger(__name__)


# ── Abstract base ────────────────────────────────────────────────

class BaseLoader(ABC):
    """Contract every document loader must fulfill."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Return a list of LangChain Document objects."""
        ...

    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for the source (file path, URL, etc.)."""
        ...


# ── PDF Loader ───────────────────────────────────────────────────

class PDFLoader(BaseLoader):
    """Load text from a PDF file or binary stream."""

    def __init__(self, path: str | Path | None = None, stream: BinaryIO | None = None):
        if path is None and stream is None:
            raise ValueError("Provide either `path` or `stream`.")
        self._path = Path(path) if path else None
        self._stream = stream

    def source_id(self) -> str:
        return str(self._path) if self._path else "upload-stream"

    def load(self) -> list[Document]:
        logger.info("loading_pdf", source=self.source_id())

        if self._stream:
            # Write stream to temp file for pypdf
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(self._stream.read())
                tmp_path = Path(tmp.name)
            reader = PdfReader(tmp_path)
        else:
            reader = PdfReader(self._path)

        documents: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": self.source_id(),
                            "page": i + 1,
                            "type": "pdf",
                        },
                    )
                )

        logger.info("pdf_loaded", source=self.source_id(), pages=len(documents))
        return documents


# ── Web Loader ───────────────────────────────────────────────────

class WebLoader(BaseLoader):
    """Scrape and extract text from a web page."""

    def __init__(self, url: str):
        self._url = url

    def source_id(self) -> str:
        return self._url

    def load(self) -> list[Document]:
        logger.info("loading_web", url=self._url)

        response = httpx.get(
            self._url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "LLM-Loop-Ingester/1.0"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        if not text.strip():
            logger.warning("empty_web_page", url=self._url)
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": self._url,
                    "type": "web",
                    "title": soup.title.string if soup.title else "",
                },
            )
        ]


# ── Plain Text Loader ───────────────────────────────────────────

class TextLoader(BaseLoader):
    """Load plain text or markdown files."""

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def source_id(self) -> str:
        return str(self._path)

    def load(self) -> list[Document]:
        logger.info("loading_text", path=str(self._path))

        text = self._path.read_text(encoding="utf-8")

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(self._path),
                    "type": "text",
                    "filename": self._path.name,
                },
            )
        ]


# ── Factory ──────────────────────────────────────────────────────

def create_loader(source: str | Path) -> BaseLoader:
    """
    Auto-detect and return the appropriate loader.

    Open/Closed: add new loader types here without modifying existing loaders.
    """
    source_str = str(source)

    if source_str.startswith(("http://", "https://")):
        return WebLoader(source_str)

    path = Path(source_str)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {source_str}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader(path=path)
    if suffix in {".txt", ".md", ".rst", ".csv"}:
        return TextLoader(path=path)

    raise ValueError(f"Unsupported file type: {suffix}")
