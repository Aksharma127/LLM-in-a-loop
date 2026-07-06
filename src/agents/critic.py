"""
Critic agent — verifies the draft answer against retrieved context.

This is the piece that separates "demo-grade" from "production-grade".
It checks for:
- Hallucination (claims not supported by context)
- Completeness (did we address all parts of the query?)
- Grounding (are citations accurate?)

Uses the FAST model since this is structured evaluation, not synthesis.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm.router import LLMRouter, TaskComplexity

logger = structlog.get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a strict answer verification agent. Your job is to evaluate a draft answer against the retrieved context.

Check for:
1. GROUNDING: Is every claim in the answer supported by the provided context?
2. COMPLETENESS: Does the answer address all parts of the user's query?
3. ACCURACY: Are the source citations correct?
4. HALLUCINATION: Does the answer make claims NOT found in the context?

Respond in this exact format:
VERDICT: <pass|retry|fail>
FEEDBACK: <brief explanation of your verdict>

Verdicts:
- "pass" — Answer is well-grounded in context and addresses the query
- "retry" — Answer has minor issues that could be fixed with better retrieval
- "fail" — Answer cannot be reliably constructed from available context

Be strict but fair. A good answer that acknowledges limitations should pass."""


async def critic_node(state: AgentState, router: LLMRouter) -> dict[str, Any]:
    """
    LangGraph node: verify draft answer against retrieved context.

    If verdict is "pass", the draft becomes the final answer.
    If "retry", we loop back to retriever (up to max retries).
    If "fail", we return an honest "I don't know" response.
    """
    query = state["query"]
    draft = state.get("draft_answer", "")
    contexts = state.get("retrieved_contexts", [])
    retry_count = state.get("retry_count", 0)

    # Skip critic for direct answers (no retrieval to verify against)
    if not contexts:
        logger.info("critic_skipped", reason="no_contexts")
        return {
            "critic_verdict": "pass",
            "critic_feedback": "Direct answer — no retrieval to verify against.",
            "final_answer": draft,
            "retry_count": retry_count,
        }

    logger.info("critic_evaluating", retry_count=retry_count)

    # Format context for the critic
    context_summary = "\n\n".join(
        f"[Chunk {i + 1}]: {r['text'][:300]}..."
        for i, r in enumerate(contexts)
    )

    critic_prompt = f"""User query: {query}

Retrieved context:
{context_summary}

Draft answer:
{draft}

Evaluate the draft answer against the context."""

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=critic_prompt),
    ]

    response = await router.route(
        messages=messages,
        complexity=TaskComplexity.FAST,
        temperature=0.0,
        max_tokens=256,
    )

    # Parse verdict
    verdict = "pass"
    feedback = response

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().lower()
            if v in ("pass", "retry", "fail"):
                verdict = v
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    logger.info("critic_verdict", verdict=verdict, feedback=feedback[:100])

    # Determine final answer based on verdict
    final_answer = draft
    if verdict == "fail":
        final_answer = (
            "I don't have enough reliable information in the knowledge base "
            "to answer this question accurately. The retrieved context doesn't "
            "sufficiently support a confident answer."
        )

    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "final_answer": final_answer,
        "retry_count": retry_count + (1 if verdict == "retry" else 0),
    }
