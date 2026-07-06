"""
Retriever agent — hybrid search + answer synthesis.

Uses the SearchService for retrieval and the STRONG model
for final synthesis from retrieved context.

SRP: searches and synthesizes — doesn't validate or route.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.llm.router import LLMRouter, TaskComplexity
from src.vectorstore.search import SearchService

logger = structlog.get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a precise, helpful assistant. Answer the user's question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context chunks.
2. If the context doesn't contain enough information, say so explicitly.
3. Cite which context chunks you used (by number).
4. Be concise but thorough.
5. If the context is contradictory, acknowledge the contradiction.

Context chunks:
{context}"""

QUERY_REWRITE_PROMPT = """Given the conversation history and current query, rewrite the query to be self-contained and optimized for semantic search.

Conversation history:
{history}

Current query: {query}

Rewritten query (just the query, nothing else):"""


async def retriever_node(
    state: AgentState,
    router: LLMRouter,
    search_service: SearchService,
) -> dict[str, Any]:
    """
    LangGraph node: retrieve relevant context and synthesize an answer.

    Steps:
    1. Optionally rewrite query using conversation context
    2. Hybrid search (dense + BM25)
    3. Synthesize answer from retrieved chunks using STRONG model
    """
    query = state["query"]
    history = state.get("conversation_history", [])

    logger.info("retriever_executing", query_preview=query[:80])

    # Step 1: Query rewriting if there's conversation history
    retrieval_query = query
    if history:
        history_text = "\n".join(
            f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            for turn in history[-6:]  # Last 3 exchanges
        )
        rewrite_messages = [
            HumanMessage(
                content=QUERY_REWRITE_PROMPT.format(
                    history=history_text,
                    query=query,
                )
            )
        ]
        retrieval_query = await router.route(
            messages=rewrite_messages,
            complexity=TaskComplexity.FAST,
            temperature=0.0,
            max_tokens=128,
        )
        logger.debug("query_rewritten", original=query[:50], rewritten=retrieval_query[:50])

    # Step 2: Hybrid search
    results = search_service.hybrid_search(retrieval_query)

    if not results:
        logger.warning("no_results_found", query=retrieval_query[:50])
        return {
            "retrieved_contexts": [],
            "retrieval_query": retrieval_query,
            "draft_answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
        }

    logger.info("results_retrieved", count=len(results))

    # Step 3: Format context for synthesis
    context_text = "\n\n".join(
        f"[Chunk {i + 1}] (score: {r['score']:.3f}, source: {r.get('metadata', {}).get('source', 'unknown')})\n{r['text']}"
        for i, r in enumerate(results)
    )

    # Step 4: Synthesize answer with STRONG model
    synthesis_messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT.format(context=context_text)),
        HumanMessage(content=query),
    ]

    draft_answer = await router.route(
        messages=synthesis_messages,
        complexity=TaskComplexity.STRONG,
        temperature=0.1,
        max_tokens=2048,
    )

    return {
        "retrieved_contexts": results,
        "retrieval_query": retrieval_query,
        "draft_answer": draft_answer,
        "model_used": router.get_model_name(TaskComplexity.STRONG),
    }


async def direct_answer_node(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """
    LangGraph node: answer directly without retrieval.

    Used when the planner decides the query doesn't need document context.
    """
    query = state["query"]
    logger.info("direct_answer_executing", query_preview=query[:80])

    messages = [
        SystemMessage(
            content="You are a helpful assistant. Answer the user's question directly and concisely."
        ),
        HumanMessage(content=query),
    ]

    draft_answer = await router.route(
        messages=messages,
        complexity=TaskComplexity.FAST,
        temperature=0.3,
        max_tokens=1024,
    )

    return {
        "retrieved_contexts": [],
        "draft_answer": draft_answer,
        "model_used": router.get_model_name(TaskComplexity.FAST),
    }
