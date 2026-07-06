"""
LLM provider adapters.

Open/Closed: each provider extends BaseLLMProvider.
Add new providers (OpenAI, Anthropic, etc.) without touching existing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


# ── Abstract base ────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """Contract for LLM providers — Liskov-substitutable."""

    @abstractmethod
    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from messages."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


# ── Groq Provider ────────────────────────────────────────────────

class GroqProvider(BaseLLMProvider):
    """Groq cloud provider — free tier, extremely fast inference."""

    def __init__(self, model: str, api_key: str | None = None):
        settings = get_settings()
        self._model = model
        self._api_key = api_key or settings.groq_api_key
        self._client = ChatGroq(
            model=self._model,
            api_key=self._api_key,
            temperature=0.0,
            max_tokens=2048,
        )

    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[BaseMessage],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        logger.debug(
            "groq_generate",
            model=self._model,
            message_count=len(messages),
        )

        self._client.temperature = temperature
        self._client.max_tokens = max_tokens

        response = await self._client.ainvoke(messages, **kwargs)
        return response.content


# ── Provider factory ─────────────────────────────────────────────

def create_provider(model: str) -> BaseLLMProvider:
    """
    Factory: given a model name, return the appropriate provider.

    Open/Closed: extend this to support OpenAI, Anthropic, etc.
    """
    # For now, everything routes through Groq
    return GroqProvider(model=model)
