"""
FastAPI application factory.

Single responsibility: construct and configure the app.
All routing and middleware setup is delegated.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI

from src.api.dependencies import lifespan
from src.api.middleware import setup_middleware
from src.api.routes import chat, health, ingest
from src.config.settings import get_settings


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    Called once at startup. All services are initialized via lifespan.
    """
    settings = get_settings()

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    app = FastAPI(
        title="LLM Loop — Multi-Agent RAG",
        description=(
            "Production-ready multi-agent RAG system with Planner, "
            "Retriever, and Critic agents orchestrated via LangGraph."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
    setup_middleware(app)

    # Routes
    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")

    return app


# ── Uvicorn entrypoint ───────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
