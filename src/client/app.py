"""
Streamlit client — premium knowledge loop workspace.

Connects to the FastAPI backend for all operations.
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from markdown_it import MarkdownIt

md = MarkdownIt()

# ── Configuration ────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Loop Studio — Semantic Research Engine",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom Styles ────────────────────────────────────────

css_path = os.path.join(os.path.dirname(__file__), "workspace.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span style="font-size: 1.5rem;">🔄</span>
        <span style="font-weight: 700; font-size: 1.25rem; letter-spacing: -0.02em;">LOOP STUDIO</span>
    </div>
    <div style="font-size: 0.8rem; color: #71717a; margin-bottom: 1.5rem; font-family: 'JetBrains Mono', monospace;">v0.1.0 // SEMANTIC SEARCH</div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # Health check status indicators
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        qdrant_status = health.get("services", {}).get("qdrant", "unknown")
        redis_status = health.get("services", {}).get("redis", "unknown")

        qdrant_dot = "dot-green" if qdrant_status == "connected" else "dot-red"
        redis_dot = "dot-green" if redis_status == "connected" else "dot-yellow"

        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 1rem;">
            <div class="status-pill">
                <span class="status-dot {qdrant_dot}"></span>
                <span>Qdrant Cloud: <b>{qdrant_status.upper()}</b></span>
            </div>
            <div class="status-pill">
                <span class="status-dot {redis_dot}"></span>
                <span>Redis Cache: <b>{redis_status.upper()}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class="status-pill" style="border-color: rgba(239, 68, 68, 0.2); background: rgba(239, 68, 68, 0.05); color: #f87171;">
            <span class="status-dot dot-red"></span>
            <span>API Server Offline</span>
        </div>
        <div style="font-size: 0.75rem; color: #a1a1aa; margin-top: 0.4rem;">
            Ensure the backend is running at port 8000.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Document ingestion
    st.markdown("### 📥 Source Ingestion")

    # File upload
    uploaded_file = st.file_uploader(
        "Load Document (PDF, TXT, MD)",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed"
    )

    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner("Processing document chunks..."):
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
                        f"Indexed {data['source']}\n"
                        f"Created {data['chunks_created']} chunks"
                    )
                else:
                    st.error(f"Ingest failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Web URL ingestion
    web_url = st.text_input("Ingest Web Article", placeholder="https://example.com/resource")
    if web_url and st.button("Fetch & Index URL", use_container_width=True):
        with st.spinner("Extracting content..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/ingest/web",
                    json={"url": web_url},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Indexed web source\nSplit into {data['chunks_created']} chunks")
                else:
                    st.error(f"Scrape failed: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.divider()

    # Database collection stats
    if st.button("Database Collection Stats", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/api/v1/ingest/status", timeout=5)
            if resp.status_code == 200:
                st.json(resp.json())
        except Exception:
            st.error("Stats service unreachable")

    st.divider()

    # Clear current session
    if st.button("Reset Current Workspace", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ── Main Workspace area ───────────────────────────────────────

st.markdown("""
<div class="workspace-header">
    <div>
        <div class="workspace-title">Loop Studio</div>
        <div class="workspace-subtitle">Multi-Agent Cognitive Engine // Semantic Retrieval & Verification</div>
    </div>
    <div class="stats-row">
        <span class="stat-badge">DB: QDRANT CLOUD</span>
        <span class="stat-badge">CPU: MINILM-L6</span>
        <span class="stat-badge">LLM: GROQ ROUTER</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Render document query and answer log
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-block">
            <div class="user-block-header">RESEARCH QUERY // PROMPT INPUT</div>
            <div style="font-size: 0.95rem; line-height: 1.5; color: #f4f4f5;">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        meta = msg.get("metadata", {})
        route = meta.get("route", "direct")
        verdict = meta.get("verdict", "pass")
        model = meta.get("model", "unknown")
        cached = meta.get("cached", False)

        badge_html = f'<span class="custom-badge badge-route">{route.upper()} ROUTE</span>'
        badge_html += f'<span class="custom-badge badge-{verdict}">CRITIC: {verdict.upper()}</span>'
        badge_html += f'<span class="custom-badge badge-model">{model}</span>'
        if cached:
            badge_html += '<span class="custom-badge badge-cached">⚡ CACHED</span>'

        # Render markdown content as HTML
        rendered_html = md.render(msg["content"])

        # Construct sources section if present
        sources_html = ""
        sources = meta.get("sources", [])
        if sources:
            source_cards_html = ""
            for src in sources:
                source_cards_html += f"""
                <div class="source-card-v2">
                    <div class="source-card-title">
                        <span>📎</span>
                        <span>{src.get('source', 'unknown')}</span>
                    </div>
                    <div class="source-card-text">{src.get('text', '')[:160]}...</div>
                    <div class="source-card-meta">Relevance Score: {src.get('score', 0):.3f}</div>
                </div>
                """
            sources_html = f"""
            <div style="margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #a1a1aa; margin-bottom: 0.75rem;">RETRIEVED SOURCE VERIFICATIONS</div>
                <div class="sources-grid">
                    {source_cards_html}
                </div>
            </div>
            """

        st.markdown(f"""<div class="assistant-block">
<div class="assistant-block-header">
<span class="assistant-block-title">VERIFIED SYNTHESIS // RESPONSE LOG</span>
<div class="badge-group">{badge_html}</div>
</div>
<div class="assistant-block-content">{rendered_html}</div>
{sources_html}
</div>""", unsafe_allow_html=True)


# Query prompt input bar
if prompt := st.chat_input("Execute semantic query..."):
    # Display user query in logs
    st.markdown(f"""
    <div class="user-block">
        <div class="user-block-header">RESEARCH QUERY // PROMPT INPUT</div>
        <div style="font-size: 0.95rem; line-height: 1.5; color: #f4f4f5;">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve response from workspace API
    with st.spinner("Processing loop graph (Planner ➔ Retriever ➔ Critic)..."):
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

                meta = {
                    "route": data.get("route", "direct"),
                    "verdict": data.get("critic_verdict", "pass"),
                    "model": data.get("model_used", "unknown"),
                    "cached": data.get("cached", False),
                    "sources": data.get("sources", []),
                }

                # Construct badges
                badge_html = f'<span class="custom-badge badge-route">{meta["route"].upper()} ROUTE</span>'
                badge_html += f'<span class="custom-badge badge-{meta["verdict"]}">CRITIC: {meta["verdict"].upper()}</span>'
                badge_html += f'<span class="custom-badge badge-model">{meta["model"]}</span>'
                if meta["cached"]:
                    badge_html += '<span class="custom-badge badge-cached">⚡ CACHED</span>'

                # Render markdown content as HTML
                rendered_html = md.render(answer)

                # Construct sources section if present
                sources_html = ""
                sources = meta["sources"]
                if sources:
                    source_cards_html = ""
                    for src in sources:
                        source_cards_html += f"""
                        <div class="source-card-v2">
                            <div class="source-card-title">
                                <span>📎</span>
                                <span>{src.get('source', 'unknown')}</span>
                            </div>
                            <div class="source-card-text">{src.get('text', '')[:160]}...</div>
                            <div class="source-card-meta">Relevance Score: {src.get('score', 0):.3f}</div>
                        </div>
                        """
                    sources_html = f"""
                    <div style="margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1rem;">
                        <div style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #a1a1aa; margin-bottom: 0.75rem;">RETRIEVED SOURCE VERIFICATIONS</div>
                        <div class="sources-grid">
                            {source_cards_html}
                        </div>
                    </div>
                    """

                st.markdown(f"""<div class="assistant-block">
<div class="assistant-block-header">
<span class="assistant-block-title">VERIFIED SYNTHESIS // RESPONSE LOG</span>
<div class="badge-group">{badge_html}</div>
</div>
<div class="assistant-block-content">{rendered_html}</div>
{sources_html}
</div>""", unsafe_allow_html=True)

                # Append response to chat logs
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": meta,
                })
                
                # Force browser reload to keep rendering in order
                st.rerun()

            else:
                error_msg = resp.json().get("detail", "Unknown backend error")
                st.error(f"API Error: {error_msg}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"API Error: {error_msg}",
                    "metadata": {}
                })

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API server. Ensure `uvicorn src.api.main:app` is running.")
        except Exception as e:
            st.error(f"Error during query execution: {e}")
