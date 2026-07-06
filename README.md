# 🔄 LLM Loop — Multi-Agent RAG System

A production-ready, scalable multi-agent Retrieval-Augmented Generation system built for low-resource machines (i3 11th gen + 8GB RAM). Heavy compute stays in the cloud; local footprint is minimal.

## Architecture

```
Data Sources → Ingestion (MiniLM, local CPU) → Qdrant Vector Store
                                                      ↓
                                              LangGraph Orchestration
                                              ┌───────┼───────┐
                                          Planner  Retriever  Critic
                                              └───────┼───────┘
                                                      ↓
                                              LLM Router (Groq Cloud)
                                              ┌───────┼───────┐
                                          8B Fast    70B Strong
                                              └───────┼───────┘
                                                      ↓
                                              Memory & Cache (Redis)
                                                      ↓
                                              FastAPI → Streamlit
```

### Three Agents

| Agent | Role | Model |
|-------|------|-------|
| **Planner** | Classifies query, decides route (retrieve/direct/web) | Llama-3.1-8B (fast) |
| **Retriever** | Hybrid search (dense + BM25 + RRF), synthesizes answer | Llama-3.3-70B (strong) |
| **Critic** | Verifies answer against context — catches hallucinations | Llama-3.1-8B (fast) |

### Resource Budget

| Component | Where | RAM Cost |
|-----------|-------|----------|
| MiniLM embeddings | Local CPU | ~500MB |
| Qdrant | Cloud free tier | 0 |
| LangGraph + FastAPI | Local | ~100MB |
| LLM inference | Groq (free) | 0 |
| Redis | Docker | ~50MB |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant + Redis)
- Groq API key (free at [console.groq.com](https://console.groq.com))

### 2. Setup

```bash
# Clone and enter
cd LLM_loop

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU-only PyTorch)
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### 3. Start infrastructure

```bash
# Start Qdrant + Redis
docker compose up -d

# Or use Qdrant Cloud (recommended for 8GB RAM):
# Set QDRANT_URL and QDRANT_API_KEY in .env
```

### 4. Seed sample data

```bash
python -m scripts.seed_data --sample
```

### 5. Run the API

```bash
uvicorn src.api.main:app --reload
```

### 6. Run the Streamlit client

```bash
streamlit run src/client/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/api/v1/chat` | Send a message through the agent pipeline |
| `POST` | `/api/v1/ingest/upload` | Upload and ingest a document |
| `POST` | `/api/v1/ingest/web` | Ingest a web page by URL |
| `GET` | `/api/v1/ingest/status` | Collection information |
| `DELETE` | `/api/v1/chat/session/{id}` | Clear a conversation session |

## SOLID Principles

- **S**ingle Responsibility: Each module owns one concern
- **O**pen/Closed: Provider adapters are extensible via ABC
- **L**iskov Substitution: All providers/loaders are interchangeable
- **I**nterface Segregation: Small, focused interfaces
- **D**ependency Inversion: Agents depend on abstractions, not implementations

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Run evaluation
python -m scripts.evaluate
```

## License

MIT
