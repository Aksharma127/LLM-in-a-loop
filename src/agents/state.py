"""
LangGraph shared state schema.

All agents read from and write to this state.
TypedDict gives us type safety and clear contracts.
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph StateGraph.

    Each agent reads what it needs and writes its outputs.
    Using total=False so agents only need to set their own fields.
    """

    # ── Input ─────────────────────────────────────────────────
    query: str                          # Original user query
    conversation_history: list[dict]    # Prior turns

    # ── Planner output ────────────────────────────────────────
    plan: str                           # The planner's decision
    route: Literal[                     # Where to go next
        "retrieve",     # Need to search documents
        "direct",       # Can answer directly (no retrieval needed)
        "web_search",   # Need external web search
    ]
    needs_retrieval: bool               # Whether retrieval is needed

    # ── Retriever output ──────────────────────────────────────
    retrieved_contexts: list[dict[str, Any]]   # Search results
    retrieval_query: str                       # Possibly rewritten query

    # ── Synthesis ─────────────────────────────────────────────
    draft_answer: str                   # Pre-verification answer
    final_answer: str                   # Post-critic verified answer

    # ── Critic output ─────────────────────────────────────────
    critic_verdict: Literal[            # Critic's judgment
        "pass",         # Answer is well-grounded
        "retry",        # Need to re-retrieve or rephrase
        "fail",         # Cannot answer from available context
    ]
    critic_feedback: str                # Explanation of verdict
    retry_count: int                    # Number of retries attempted

    # ── Metadata ──────────────────────────────────────────────
    error: str | None                   # Error message if any
    model_used: str                     # Which model was used
    total_tokens: int                   # Token usage tracking
