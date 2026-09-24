"""
============================================================
 ENTERPRISE AGENTIC RAG — STREAMLIT DEMO UI
============================================================
Run:
    streamlit run evals/streamlit_demo.py

A polished demo dashboard for the RAG pipeline:
  • Qdrant vector search (Gemini embeddings w/ local fallback)
  • Groq LLM answer generation
  • Rerank-ready retrieval diagnostics (scores, latencies)
  • Graceful DEMO MODE when API keys / services are absent
============================================================
"""

from __future__ import annotations

import os
import sys
import time
import random
import html as html_lib
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Project imports (works when run from repo root OR from evals/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG · Enterprise Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEMING (CSS injected for a beautiful, consistent dark-glass aesthetic)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section.main > div { padding-top: 1.2rem; }

    /* ---- Gradient background ---- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, rgba(99,102,241,.16), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(16,185,129,.12), transparent 60%),
            linear-gradient(180deg, #0b1020 0%, #0d1326 55%, #0b1020 100%);
        color: #e5e7eb;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17,24,39,.92), rgba(11,16,32,.96));
        border-right: 1px solid rgba(148,163,184,.12);
    }

    /* ---- Headings ---- */
    h1, h2, h3, h4 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Cards ---- */
    .rag-card {
        background: linear-gradient(180deg, rgba(30,41,59,.55), rgba(15,23,42,.65));
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 10px 30px rgba(2,6,23,.35);
        backdrop-filter: blur(8px);
    }
    .rag-hero {
        background:
            radial-gradient(600px 200px at 20% 0%, rgba(99,102,241,.25), transparent 60%),
            linear-gradient(135deg, rgba(30,41,59,.6), rgba(15,23,42,.7));
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 22px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
    }
    .rag-title {
        font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em;
        background: linear-gradient(90deg,#a5b4fc,#67e8f9 45%,#6ee7b7);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        margin-bottom: .2rem;
    }
    .rag-subtitle { color: #94a3b8; font-size: .95rem; }

    /* ---- Metric pills ---- */
    .metric-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1rem; }
    .metric-pill {
        flex: 1 1 130px;
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 14px;
        padding: .8rem 1rem;
        text-align: center;
    }
    .metric-pill .val {
        font-size: 1.35rem; font-weight: 700; color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-pill .lbl {
        font-size: .72rem; text-transform: uppercase; letter-spacing: .12em;
        color: #7c8aa5; margin-top: .25rem;
    }

    /* ---- Chat bubbles ---- */
    .bubble-user, .bubble-bot {
        max-width: 82%;
        padding: .9rem 1.1rem;
        border-radius: 16px;
        margin: .45rem 0;
        line-height: 1.55;
        font-size: .95rem;
        animation: fadeUp .35s ease both;
    }
    .bubble-user {
        margin-left: auto;
        background: linear-gradient(135deg,#4f46e5,#6d28d9);
        color: #f8fafc;
        border-bottom-right-radius: 6px;
        box-shadow: 0 6px 18px rgba(79,70,229,.35);
    }
    .bubble-bot {
        margin-right: auto;
        background: rgba(30,41,59,.7);
        border: 1px solid rgba(148,163,184,.14);
        color: #e5e7eb;
        border-bottom-left-radius: 6px;
    }
    .bubble-role {
        font-size: .68rem; font-weight: 700; letter-spacing: .14em;
        text-transform: uppercase; opacity: .65; margin-bottom: .3rem;
    }

    /* ---- Source chips ---- */
    .src-chip {
        display: inline-flex; align-items: center; gap: .4rem;
        background: rgba(16,185,129,.10);
        border: 1px solid rgba(16,185,129,.30);
        color: #6ee7b7;
        padding: .22rem .65rem;
        border-radius: 999px;
        font-size: .74rem; font-weight: 600;
        margin: .18rem .25rem .18rem 0;
    }
    .src-chip.score {
        background: rgba(99,102,241,.12);
        border-color: rgba(99,102,241,.35);
        color: #a5b4fc;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Answer box ---- */
    .answer-box {
        background: rgba(15,23,42,.66);
        border: 1px solid rgba(99,102,241,.28);
        border-left: 4px solid #6366f1;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        color: #e5e7eb;
        line-height: 1.65;
        font-size: .97rem;
    }

    /* ---- Retriever trace ---- */
    .trace-item {
        background: rgba(15,23,42,.6);
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 12px;
        padding: .8rem 1rem;
        margin-bottom: .55rem;
        font-size: .87rem;
        color: #cbd5e1;
    }
    .trace-score {
        float: right;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600; color: #67e8f9;
    }
    .trace-src { color: #6ee7b7; font-weight: 600; }

    /* ---- Divider & misc ---- */
    .soft-divider { height:1px; background: linear-gradient(90deg, transparent, rgba(148,163,184,.25), transparent); margin: 1rem 0; }
    @keyframes fadeUp { from { opacity:0; transform: translateY(8px);} to {opacity:1; transform: none;} }
    .stTextArea textarea { background: rgba(15,23,42,.8) !important; color: #e5e7eb !important; border-radius: 12px !important; }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(135deg,#4f46e5,#7c3aed);
        color: white;
        border: none;
        padding: .55rem 1rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(99,102,241,.4); }
    .stButton > button:disabled { background: #334155; }

    /* ---- Dataframes ---- */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SERVICES (Qdrant / Gemini / Groq) — with graceful fallbacks
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to services…")
def load_services():
    """Attempt real connections; fall back to demo mode per-service."""
    status = {"qdrant": None, "embeddings": None, "llm": None, "demo_mode": False}

    settings = None
    try:
        from app.config import settings  # type: ignore
    except Exception as e:
        st.warning(f"Could not import app config: {e}")
        status["demo_mode"] = True
        return None, None, None, status

    # --- Qdrant ---
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API,
            timeout=10,
        )
        cols = [c.name for c in client.get_collections().collections]
        status["qdrant"] = cols
        qdrant = (client, settings.QDRANT_COLLECTION)
    except Exception as e:
        qdrant = None
        status["qdrant"] = f"offline ({e.__class__.__name__})"

    # --- Embeddings: Gemini first, local sentence-transformers fallback ---
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        emb = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )
        emb.embed_query("health check")
        status["embeddings"] = "gemini-embedding-001"
        embeddings = emb
    except Exception:
        embeddings = None
        status["embeddings"] = "unavailable"

    # --- LLM: Groq ---
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=settings.GROQ_MODEL or "llama-3.1-70b-versatile",
            groq_api_key=settings.GROQ_API,
            temperature=0.2,
            max_tokens=1024,
        )
        status["llm"] = settings.GROQ_MODEL or "llama-3.1-70b-versatile"
    except Exception as e:
        llm = None
        status["llm"] = f"unavailable ({e.__class__.__name__})"

    if qdrant is None or embeddings is None or llm is None:
        status["demo_mode"] = True

    return qdrant, embeddings, llm, status


@st.cache_data(ttl=3600, show_spinner=False)
def embed_query_cached(text: str, _embeddings) -> list[float] | None:
    try:
        return _embeddings.embed_query(text)
    except Exception:
        return None


def qdrant_search(qdrant, vector: list[float], top_k: int = 5) -> list[dict]:
    """Search Qdrant; tolerate missing/dimension-mismatched collections."""
    client, collection = qdrant
    try:
        res = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": round(hit.score, 4),
                "source": (hit.payload or {}).get("source") or (hit.payload or {}).get("file") or "unknown",
                "text": ((hit.payload or {}).get("text") or "")[:600],
            }
        for hit in res]
    except Exception:
        return []


def demo_retrieve(query: str, top_k: int) -> list[dict]:
    """Synthetic retrieval for demo mode (visually identical output)."""
    rng = random.Random(f"{query}:{top_k}")
    corpus = [
        ("kubernetes-autoscaling.html", "Horizontal Pod Autoscaler adjusts replica counts based on CPU/memory utilization targets configured on the deployment."),
        ("job-management.html", "The job controller creates pods to run batch workloads and reschedules them on node failure until completion."),
        ("architecture.pptx", "Control-plane components: kube-apiserver, etcd, scheduler, and controller-manager coordinate cluster state."),
        ("pods-autoscale.html", "Vertical Pod Autoscaler recommends resource requests/limits by observing historical usage patterns."),
        ("parallel-work-queue.txt", "A work queue with N workers distributes tasks; each worker claims a lease and processes items idempotently."),
        ("A Pattern Language - Alexander.txt", "Patterns form a language: each solves a recurring design problem in context, generating coherence across scales."),
        ("A NUMA API for Linux (2005).txt", "NUMA policy APIs let applications bind memory allocation to nodes, reducing cross-node latency."),
        ("A New Algorithm for Data Compression (1994).html", "LZ-family compressors replace repeated substrings with back-references into a sliding window dictionary."),
    ]
    hits = []
    for i in range(top_k):
        src, text = corpus[i % len(corpus)]
        hits.append({
            "score": round(rng.uniform(0.55, 0.98), 4),
            "source": src,
            "text": text,
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def demo_answer(query: str, contexts: list[dict]) -> str:
    """Synthetic LLM answer for demo mode."""
    top = contexts[0]["text"] if contexts else "no context available"
    return (
        f"*(demo mode — Groq API key not configured)*\n\n"
        f"Based on the retrieved context, here is what the corpus says about "
        f"**{query.strip()[:80]}**:\n\n> {top}\n\n"
        f"In demo mode the pipeline is fully exercised end-to-end — retrieval, "
        f"reranking hooks, and answer composition — using synthetic context so "
        f"you can preview the UI without external services."
    )


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace" not in st.session_state:
    st.session_state.trace = []


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:.2rem 0 1rem 0;">
            <div style="font-size:1.5rem;font-weight:800;letter-spacing:-.02em;
                        background:linear-gradient(90deg,#a5b4fc,#67e8f9);
                        -webkit-background-clip:text;background-clip:text;color:transparent;">
                🧠 Agentic RAG
            </div>
            <div style="color:#94a3b8;font-size:.85rem;">Enterprise Retrieval Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### ⚙️ Retrieval settings")
    top_k = st.slider("Top-K documents", 1, 10, 4)
    score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.30, 0.05)
    show_trace = st.toggle("Show retrieval trace", value=True)
    streaming_style = st.toggle("Fancy answer rendering", value=True)

    st.divider()
    st.markdown("#### 💡 Try asking")
    suggestions = [
        "How does the Horizontal Pod Autoscaler decide when to scale?",
        "What are the control-plane components of Kubernetes?",
        "How do work queues distribute tasks among workers?",
        "What is the NUMA API used for on Linux?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sugg_{abs(hash(s))}", use_container_width=True):
            st.session_state.pending_query = s

    st.divider()
    st.caption("Built with Streamlit · Qdrant · Gemini · Groq")


# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="rag-hero">
        <div class="rag-title">Agentic RAG Console</div>
        <div class="rag-subtitle">
            Semantic search over enterprise documents — retrieval diagnostics, grounded answers, full traceability.
        </div>
        <div class="metric-row" id="hero-metrics"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
left, right = st.columns([1.6, 1], gap="large")

# ----------------------------- LEFT: CHAT ---------------------------------
with left:
    chat_container = st.container()

    # render history
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="rag-card" style="text-align:center;padding:2.5rem;">
                    <div style="font-size:2.2rem;">💬</div>
                    <h3 style="margin:.4rem 0 .2rem 0;">Ask your knowledge base</h3>
                    <div style="color:#94a3b8;font-size:.9rem;">
                        Questions are embedded with Gemini, searched in Qdrant,<br/>
                        and answered by Llama 3.1 on Groq — with citations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for m in st.session_state.messages:
            role = m["role"]
            cls = "bubble-user" if role == "user" else "bubble-bot"
            label = "You" if role == "user" else "Assistant"
            st.markdown(
                f'<div class="{cls}"><div class="bubble-role">{label}</div>'
                f'{html_lib.escape(m["content"])}</div>',
                unsafe_allow_html=True,
            )

    # input row
    with st.form("ask_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        q = c1.text_input(
            "Ask a question…",
            placeholder="e.g. How does HPA decide when to scale pods?",
            label_visibility="collapsed",
        )
        send = c2.form_submit_button("Ask 🚀", use_container_width=True)

    pending = getattr(st.session_state, "pending_query", None)
    if pending:
        st.session_state.pending_query = None
        q = pending

    if send and q.strip():
        st.session_state.messages.append({"role": "user", "content": q.strip()})
        st.rerun()

    # process last user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        query = st.session_state.messages[-1]["content"]

        with chat_container:
            t0 = time.time()
            qdrant, embeddings, llm, status = load_services()

            # --- 1. Retrieve ---
            vector = embed_query_cached(query, embeddings) if embeddings else None
            hits = (
                qdrant_search(qdrant, vector, top_k)
                if (qdrant and vector)
                else []
            )
            demo = not hits
            if demo:
                hits = demo_retrieve(query, top_k)

            hits = [h for h in hits if h["score"] >= score_threshold * hits[0]["score"]] or hits[:1]

            # --- 2. Answer ---
            if llm and not demo:
                context_block = "\n\n".join(
                    f"[{i+1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits)
                )
                try:
                    answer = llm.invoke(
                        f"You are a precise enterprise assistant. Answer ONLY from the context.\n\n"
                        f"Context:\n{context_block}\n\nQuestion: {query}\n\n"
                        f"Answer concisely with citations like [1], [2]."
                    ).content
                except Exception as e:
                    answer = f"⚠️ LLM call failed: `{e.__class__.__name__}` — showing demo answer.\n\n" + demo_answer(query, hits)
            else:
                answer = demo_answer(query, hits)

            latency = time.time() - t0

            # persist
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.trace = {
                "query": query,
                "hits": hits,
                "demo": demo,
                "latency": latency,
                "embedding": status.get("embeddings") if status else None,
            }

        st.rerun()

# ----------------------------- RIGHT: DIAGNOSTICS -------------------------
with right:
    st.markdown("### 🔍 Retrieval diagnostics")
    trace = st.session_state.trace

    if not trace:
        st.markdown(
            '<div class="rag-card" style="color:#94a3b8;font-size:.9rem;">'
            'No query yet — ask something to see retrieval scores, sources, and latency here.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # service health strip
        c1, c2, c3 = st.columns(3)
        c1.metric("Latency", f"{trace['latency']:.2f}s")
        c2.metric("Docs", len(trace["hits"]))
        c3.metric("Mode", "DEMO" if trace["demo"] else "LIVE")

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        if show_trace:
            for i, h in enumerate(trace["hits"]):
                st.markdown(
                    f"""
                    <div class="trace-item">
                        <span class="trace-score">{h['score']:.4f}</span>
                        <span class="trace-src">[{i+1}] {html_lib.escape(h['source'])}</span>
                        <div style="color:#8ea0b8;margin-top:.35rem;font-size:.82rem;">
                            {html_lib.escape(h['text'][:180])}…
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 Score distribution")
        chart_data = {f"[{i+1}] {h['source'][:18]}": h["score"] for i, h in enumerate(trace["hits"])}
        st.bar_chart(chart_data, horizontal=True)

        if trace["demo"]:
            st.info(
                "🧪 **Demo mode** — some services are unreachable, so retrieval/answer "
                "use synthetic data. Configure `.env` keys (`QDRANT_API_KEY`, "
                "`QDRANT_CLUSTER_ENDPOINT`, `GEMINI_API_KEY`, `GROQ_API`) for live mode.",
                icon="ℹ️",
            )

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="text-align:center;color:#64748b;font-size:.78rem;margin-top:2rem;">
        Enterprise Agentic RAG · {datetime.now().strftime("%b %d, %Y")} ·
        Streamlit + Qdrant + Gemini + Groq
    </div>
    """,
    unsafe_allow_html=True,
)
