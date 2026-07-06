"""
Tests for agent state and graph construction.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.state import AgentState


class TestAgentState:
    """Test the shared state schema."""

    def test_state_creation(self):
        """Should create a valid state dict."""
        state: AgentState = {
            "query": "test question",
            "retry_count": 0,
        }
        assert state["query"] == "test question"
        assert state["retry_count"] == 0

    def test_state_with_all_fields(self):
        """Should accept all defined fields."""
        state: AgentState = {
            "query": "test",
            "route": "retrieve",
            "needs_retrieval": True,
            "draft_answer": "draft",
            "final_answer": "final",
            "critic_verdict": "pass",
            "retry_count": 0,
        }
        assert state["route"] == "retrieve"
        assert state["critic_verdict"] == "pass"
