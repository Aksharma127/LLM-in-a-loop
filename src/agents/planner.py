"""
Planner agent — decides the execution route.

Uses the FAST model (8B) to classify the query and decide:
- retrieve: query needs document context
- direct: can answer from general knowledge
- web_search: needs live web data

SRP: only classifies and routes — never generates answers.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm.router import LLMRouter, TaskComplexity

logger = structlog.get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a query routing agent. Your ONLY job is to classify the user's query and decide the next step.

Analyze the query and respond with EXACTLY ONE of these routes:
- "retrieve" — if the query needs information from uploaded documents or the knowledge base
- "direct" — if you can answer from general knowledge without any retrieval
- "web_search" — if the query needs live/current web data

Also provide a brief plan (1-2 sentences) explaining your routing decision.

Respond in this exact format:
ROUTE: <route>
PLAN: <brief plan>

Examples:
- "What does chapter 3 say about X?" → ROUTE: retrieve
- "What is 2+2?" → ROUTE: direct
- "What is the current stock price of AAPL?" → ROUTE: web_search"""


async def planner_node(state: AgentState, router: LLMRouter) -> dict[str, Any]:
    """
    LangGraph node: classify query and decide execution route.

    Uses the FAST model since this is a simple classification task.
    """
    query = state["query"]
    logger.info("planner_executing", query_preview=query[:80])

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    response = await router.route(
        messages=messages,
        complexity=TaskComplexity.FAST,
        temperature=0.0,
        max_tokens=256,
    )

    # Parse the structured response
    route = "retrieve"  # Default to retrieval if parsing fails
    plan = response

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("ROUTE:"):
            route_value = line.split(":", 1)[1].strip().lower()
            if route_value in ("retrieve", "direct", "web_search"):
                route = route_value
        elif line.upper().startswith("PLAN:"):
            plan = line.split(":", 1)[1].strip()

    logger.info("planner_decided", route=route, plan=plan[:100])

    return {
        "route": route,
        "plan": plan,
        "needs_retrieval": route == "retrieve",
        "model_used": router.get_model_name(TaskComplexity.FAST),
    }
