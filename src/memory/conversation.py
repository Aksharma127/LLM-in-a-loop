"""
Short-term conversation buffer.

Keeps recent conversation turns in-process with optional
Redis persistence for stateless worker deployments.

SRP: only manages the conversation window.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = structlog.get_logger(__name__)


class ConversationBuffer:
    """
    Fixed-size sliding window of conversation turns.

    For stateless deployments, serialize to/from Redis
    using to_dict() / from_dict().
    """

    def __init__(self, max_turns: int = 10):
        self._max_turns = max_turns
        self._history: deque[dict[str, Any]] = deque(maxlen=max_turns * 2)
        self._session_id: str = ""
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    def add_user_message(self, content: str) -> None:
        """Record a user turn."""
        self._history.append({
            "role": "human",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_ai_message(self, content: str) -> None:
        """Record an AI turn."""
        self._history.append({
            "role": "ai",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_messages(self) -> list[BaseMessage]:
        """Return conversation history as LangChain messages."""
        messages: list[BaseMessage] = []
        for turn in self._history:
            if turn["role"] == "human":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def get_context_window(self, last_n: int | None = None) -> list[BaseMessage]:
        """Return the last N turns as messages."""
        messages = self.get_messages()
        if last_n is not None:
            return messages[-(last_n * 2) :]
        return messages

    def clear(self) -> None:
        """Reset conversation history."""
        self._history.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis storage."""
        return {
            "session_id": self._session_id,
            "created_at": self._created_at,
            "history": list(self._history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], max_turns: int = 10) -> ConversationBuffer:
        """Deserialize from Redis storage."""
        buf = cls(max_turns=max_turns)
        buf._session_id = data.get("session_id", "")
        buf._created_at = data.get("created_at", "")
        for turn in data.get("history", []):
            buf._history.append(turn)
        return buf

    def __len__(self) -> int:
        return len(self._history)
