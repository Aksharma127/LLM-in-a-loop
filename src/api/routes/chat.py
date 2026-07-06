"""
Chat endpoints — the main user-facing API.

Runs the full multi-agent pipeline: Planner → Retriever → Critic
with response caching and conversation history.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_agent_graph, get_cache_service
from src.memory.conversation import ConversationBuffer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-process session store (for dev; use Redis in production)
_sessions: dict[str, ConversationBuffer] = {}


class ChatRequest(BaseModel):
    """Incoming chat message."""
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response with metadata."""
    answer: str
    session_id: str
    route: str
    critic_verdict: str
    sources: list[dict[str, Any]]
    model_used: str
    cached: bool


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message through the multi-agent pipeline.

    Flow:
    1. Check response cache
    2. Get/create conversation session
    3. Run agent graph (Planner → Retriever/Direct → Critic)
    4. Cache response
    5. Update conversation history
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "chat_request",
        session=session_id,
        message_preview=message[:80],
    )

    # 1. Check cache
    cache = get_cache_service()
    cached_response = await cache.get_cached_response(message)
    if cached_response:
        return ChatResponse(
            answer=cached_response,
            session_id=session_id,
            route="cached",
            critic_verdict="pass",
            sources=[],
            model_used="cache",
            cached=True,
        )

    # 2. Get or create session
    if session_id not in _sessions:
        _sessions[session_id] = ConversationBuffer(max_turns=10)
        _sessions[session_id].session_id = session_id
    session = _sessions[session_id]

    # 3. Build initial state
    history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in session._history
    ]

    initial_state = {
        "query": message,
        "conversation_history": history,
        "retry_count": 0,
    }

    # 4. Run agent graph
    try:
        graph = get_agent_graph()
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error("agent_graph_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Agent pipeline failed: {e!s}",
        )

    # 5. Extract results
    final_answer = result.get("final_answer", result.get("draft_answer", "No answer generated."))
    route = result.get("route", "unknown")
    verdict = result.get("critic_verdict", "unknown")
    model_used = result.get("model_used", "unknown")

    sources = [
        {
            "text": ctx.get("text", "")[:200],
            "source": ctx.get("metadata", {}).get("source", "unknown"),
            "score": ctx.get("score", 0.0),
        }
        for ctx in result.get("retrieved_contexts", [])
    ]

    # 6. Cache the response
    await cache.cache_response(message, final_answer)

    # 7. Update conversation history
    session.add_user_message(message)
    session.add_ai_message(final_answer)

    logger.info(
        "chat_response",
        session=session_id,
        route=route,
        verdict=verdict,
        sources_count=len(sources),
    )

    return ChatResponse(
        answer=final_answer,
        session_id=session_id,
        route=route,
        critic_verdict=verdict,
        sources=sources,
        model_used=model_used,
        cached=False,
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
