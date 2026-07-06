"""
LLM Router — routes tasks to the right model by complexity.

Same pattern as Headless BAI: cheap/fast model for planning and
classification, strong model for final synthesis.

SRP: only decides which model to use and delegates.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import structlog
from langchain_core.messages import BaseMessage

from src.config.settings import get_settings
from src.llm.providers import BaseLLMProvider, create_provider

logger = structlog.get_logger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for routing."""
    FAST = "fast"       # Planning, classification, simple extraction
    STRONG = "strong"   # Final synthesis, complex reasoning


class LLMRouter:
    """
    Routes LLM calls to the appropriate model based on task complexity.

    Interface Segregation: consumers call route() with a complexity tag
    and don't need to know about provider details.
    """

    def __init__(self):
        settings = get_settings()
        self._providers: dict[TaskComplexity, BaseLLMProvider] = {
            TaskComplexity.FAST: create_provider(settings.fast_model),
            TaskComplexity.STRONG: create_provider(settings.strong_model),
        }

    async def route(
        self,
        messages: list[BaseMessage],
        complexity: TaskComplexity = TaskComplexity.FAST,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Route a request to the appropriate model.

        Args:
            messages: The conversation messages.
            complexity: FAST for planning/routing, STRONG for synthesis.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            The generated text.
        """
        provider = self._providers[complexity]
        logger.info(
            "llm_routing",
            complexity=complexity.value,
            model=provider.model_name(),
            message_count=len(messages),
        )

        return await provider.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_model_name(self, complexity: TaskComplexity) -> str:
        """Return the model name for a given complexity level."""
        return self._providers[complexity].model_name()
