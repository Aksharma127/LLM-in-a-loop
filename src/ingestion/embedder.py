"""
Local embedding service — MiniLM-L6-v2 on CPU.

SRP: this module only converts text → vectors.
~90MB model, ~500MB RAM at peak, runs fine on CPU.
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Wraps sentence-transformers for local CPU embedding.

    Dependency Inversion: agents and search modules depend on this
    service's interface, not on sentence-transformers directly.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._device = device or settings.embedding_device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use to avoid startup cost."""
        if self._model is None:
            logger.info(
                "loading_embedding_model",
                model=self._model_name,
                device=self._device,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            logger.info("embedding_model_loaded", model=self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a batch of texts and return dense vectors.

        Args:
            texts: list of strings to embed.
            batch_size: encoding batch size (tune for your RAM).

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []

        logger.debug("embedding_texts", count=len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Cosine similarity via dot product
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Singleton factory — model loaded once, reused everywhere."""
    return EmbeddingService()
