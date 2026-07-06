"""
Streamlit client — clean, functional chat interface.

Connects to the FastAPI backend for all operations.
"""

from __future__ import annotations

import requests
import streamlit as st

# ── Configuration ────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Loop — Multi-Agent RAG",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0a0a0f;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d0d14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-pass { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
    .badge-retry { background: rgba(250, 204, 21, 0.15); color: #facc15; }
    .badge-fail { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .badge-cached { background: rgba(99, 102, 241, 0.15); color: #6366f1; }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-label {
        font-size: 0.8em;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔄 LLM Loop")
    st.markdown("*Multi-Agent RAG System*")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        qdrant_status = health.get("services", {}).get("qdrant", "unknown")
        redis_status = health.get("services", {}).get("redis", "unknown")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"**Qdrant:** {'🟢' if qdrant_status == 'connected' else '🔴'} {qdrant_status}"
            )
        with col2:
            st.markdown(
                f"**Redis:** {'🟢' if redis_status == 'connected' else '🟡'} {redis_status}"
            )
    except Exception:
        st.warning("⚠️ API not reachable. Start the FastAPI server first.")

    st.divider()

    # Document ingestion
    st.markdown("### 📄 Ingest Documents")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT, or MD files to the knowledge base",
    )

    if uploaded_file and st.button("📥 Ingest File", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                resp = requests.post(
                    f"{API_BASE}/api/v1/ingest/upload",
                    files=files,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Ingested **{data['source']}** — "
                        f"{data['chunks_created']} chunks, "
                        f"{data['points_indexed']} indexed"
                    )
                else:
                    st.error(f"❌ {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

    # Web ingestion
    web_url = st.text_input("Or enter a URL", placeholder="https://example.com/article")
    if web_url and st.button("🌐 Ingest URL", use_container_width=True):
        with st.spinner("Fetching and ingesting..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/ingest/web",
                    json={"url": web_url},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Ingested **{data['source'][:40]}...** — "
                        f"{data['chunks_created']} chunks"
                    )
                else:
                    st.error(f"❌ {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

    st.divider()

    # Collection status
    if st.button("📊 Collection Status", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/api/v1/ingest/status", timeout=5)
            if resp.status_code == 200:
                info = resp.json()
                st.json(info)
        except Exception:
            st.error("Could not fetch status")

    st.divider()

    # Session management
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ── Main chat area ───────────────────────────────────────────

st.markdown("# 🔄 LLM Loop")
st.markdown(
    "*Multi-agent RAG with Planner → Retriever → Critic verification*",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show metadata for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            with st.expander("🔍 Pipeline Details", expanded=False):
                cols = st.columns(4)
                with cols[0]:
                    route = meta.get("route", "?")
                    st.markdown(f"**Route:** `{route}`")
                with cols[1]:
                    verdict = meta.get("verdict", "?")
                    badge_class = f"badge-{verdict}"
                    st.markdown(
                        f'**Critic:** <span class="badge {badge_class}">{verdict.upper()}</span>',
                        unsafe_allow_html=True,
                    )
                with cols[2]:
                    st.markdown(f"**Model:** `{meta.get('model', '?')}`")
                with cols[3]:
                    cached = meta.get("cached", False)
                    if cached:
                        st.markdown(
                            '<span class="badge badge-cached">CACHED</span>',
                            unsafe_allow_html=True,
                        )

                # Sources
                sources = meta.get("sources", [])
                if sources:
                    st.markdown("**Sources:**")
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📎 **{src.get("source", "unknown")}** '
                            f'(score: {src.get("score", 0):.3f})<br/>'
                            f'{src.get("text", "")[:150]}...'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Planner → Retriever → Critic)"):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    st.session_state.session_id = data["session_id"]

                    st.markdown(answer)

                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "route": data.get("route", "?"),
                            "verdict": data.get("critic_verdict", "?"),
                            "model": data.get("model_used", "?"),
                            "cached": data.get("cached", False),
                            "sources": data.get("sources", []),
                        },
                    })

                    # Show pipeline details inline
                    with st.expander("🔍 Pipeline Details", expanded=False):
                        cols = st.columns(4)
                        with cols[0]:
                            st.markdown(f"**Route:** `{data.get('route', '?')}`")
                        with cols[1]:
                            verdict = data.get("critic_verdict", "?")
                            st.markdown(f"**Critic:** `{verdict}`")
                        with cols[2]:
                            st.markdown(f"**Model:** `{data.get('model_used', '?')}`")
                        with cols[3]:
                            if data.get("cached"):
                                st.markdown("**⚡ Cached**")

                        sources = data.get("sources", [])
                        if sources:
                            st.markdown("**Sources:**")
                            for src in sources:
                                st.markdown(
                                    f"- `{src.get('source', 'unknown')}` "
                                    f"(score: {src.get('score', 0):.3f})"
                                )
                else:
                    error = resp.json().get("detail", "Unknown error")
                    st.error(f"❌ API error: {error}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {error}",
                    })

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API. "
                    "Make sure the FastAPI server is running: "
                    "`uvicorn src.api.main:app --reload`"
                )
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
