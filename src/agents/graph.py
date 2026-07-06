"""
LangGraph StateGraph — wires the three agents together.

This is the heart of the orchestration:
  Planner → (retrieve | direct) → Critic → (pass | retry | fail)

Conditional edges route based on planner decisions and critic verdicts.
The retry loop gives the system a second chance at retrieval.

No local compute cost — it's just orchestration logic.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import structlog
from langgraph.graph import END, StateGraph

from src.agents.critic import critic_node
from src.agents.planner import planner_node
from src.agents.retriever import direct_answer_node, retriever_node
from src.agents.state import AgentState
from src.llm.router import LLMRouter
from src.vectorstore.search import SearchService

logger = structlog.get_logger(__name__)

MAX_RETRIES = 1  # Maximum retry loops before giving up


def build_agent_graph(
    router: LLMRouter,
    search_service: SearchService,
) -> StateGraph:
    """
    Build the multi-agent LangGraph.

    Architecture:
    ```
    START → planner
              │
              ├─ route="retrieve" → retriever → critic
              │                                    │
              │                    ┌───────────────┤
              │                    │               │
              │              verdict="pass"   verdict="retry" (→ retriever, max 1)
              │                    │               │
              │                    ▼          verdict="fail"
              │                   END              │
              │                                    ▼
              ├─ route="direct" → direct_answer → critic → END
              │
              └─ route="web_search" → direct_answer → critic → END
    ```
    """

    # ── Bind dependencies to node functions ──────────────────
    # Partial application: inject router & search_service so
    # LangGraph only needs to pass the state dict.

    async def _planner(state: AgentState) -> dict[str, Any]:
        return await planner_node(state, router)

    async def _retriever(state: AgentState) -> dict[str, Any]:
        return await retriever_node(state, router, search_service)

    async def _direct(state: AgentState) -> dict[str, Any]:
        return await direct_answer_node(state, router)

    async def _critic(state: AgentState) -> dict[str, Any]:
        return await critic_node(state, router)

    # ── Build graph ──────────────────────────────────────────
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", _planner)
    graph.add_node("retriever", _retriever)
    graph.add_node("direct_answer", _direct)
    graph.add_node("critic", _critic)

    # Set entry point
    graph.set_entry_point("planner")

    # ── Conditional edges ────────────────────────────────────

    # After planner: route based on classification
    def route_after_planner(state: AgentState) -> str:
        route = state.get("route", "retrieve")
        if route == "retrieve":
            return "retriever"
        # Both "direct" and "web_search" go to direct_answer for now
        return "direct_answer"

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "direct_answer": "direct_answer",
        },
    )

    # After retriever → always go to critic
    graph.add_edge("retriever", "critic")

    # After direct_answer → always go to critic
    graph.add_edge("direct_answer", "critic")

    # After critic: check verdict
    def route_after_critic(state: AgentState) -> str:
        verdict = state.get("critic_verdict", "pass")
        retry_count = state.get("retry_count", 0)

        if verdict == "pass" or verdict == "fail":
            return "end"
        if verdict == "retry" and retry_count < MAX_RETRIES:
            return "retriever"
        # Max retries exhausted
        return "end"

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "retriever": "retriever",
        },
    )

    return graph


def compile_agent_graph(
    router: LLMRouter,
    search_service: SearchService,
) -> Any:
    """Build and compile the agent graph, ready for invocation."""
    graph = build_agent_graph(router, search_service)
    compiled = graph.compile()
    logger.info("agent_graph_compiled")
    return compiled
