"""
Associative Memory Retriever for Long-Context QA
=================================================
Streamlit UI — Modern Hopfield Network-powered document QA system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import time
import json
import re
from typing import List, Optional

# ─── Page Config (must be first) ──────────────────────────────────
st.set_page_config(
    page_title="Hopfield QA — Associative Memory Retriever",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fonts & Root */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
  --primary: #00D4FF;
  --secondary: #7B2FBE;
  --accent: #FF6B35;
  --success: #00C49A;
  --warning: #FFB800;
  --bg: #080C18;
  --bg-card: #0F1623;
  --bg-input: #131A2B;
  --border: #1E2D45;
  --text: #E2E8F0;
  --text-muted: #64748B;
  --text-dim: #94A3B8;
}

/* Global */
.stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1rem !important; max-width: 1400px; }

/* Header */
.hero-header {
  background: linear-gradient(135deg, #0A0E1A 0%, #0d1528 50%, #111827 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.hero-header::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 20%, rgba(123,47,190,0.08) 0%, transparent 50%);
  pointer-events: none;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 60%, #FF6B35 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  line-height: 1.2;
}
.hero-subtitle {
  font-family: 'Space Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-dim);
  margin-top: 0.5rem;
  letter-spacing: 0.05em;
}
.hero-badge {
  display: inline-block;
  background: rgba(0,212,255,0.1);
  border: 1px solid rgba(0,212,255,0.3);
  color: var(--primary);
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  padding: 3px 10px;
  border-radius: 20px;
  margin-right: 6px;
  letter-spacing: 0.05em;
}

/* Cards */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
}
.card-header {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  color: var(--primary);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}

/* Answer Box */
.answer-box {
  background: linear-gradient(135deg, #0d1528, #111827);
  border: 1px solid rgba(0,212,255,0.25);
  border-left: 4px solid var(--primary);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1rem 0;
  position: relative;
}
.answer-box::before {
  content: '⟳ RETRIEVED ANSWER';
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem;
  color: var(--primary);
  letter-spacing: 0.15em;
  display: block;
  margin-bottom: 0.75rem;
}
.answer-text {
  font-size: 0.95rem;
  line-height: 1.75;
  color: var(--text);
}

/* Evidence chunks */
.evidence-chunk {
  background: #0A0F1E;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  position: relative;
  transition: border-color 0.2s;
}
.evidence-chunk:hover { border-color: rgba(0,212,255,0.3); }
.chunk-rank {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem;
  color: var(--primary);
  position: absolute;
  top: 8px; right: 10px;
}
.chunk-source {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  color: var(--secondary);
  margin-bottom: 0.4rem;
}
.chunk-text {
  font-size: 0.82rem;
  color: var(--text-dim);
  line-height: 1.6;
}
.chunk-scores {
  display: flex;
  gap: 12px;
  margin-top: 0.5rem;
}
.score-pill {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem;
  padding: 2px 8px;
  border-radius: 12px;
  background: rgba(0,0,0,0.4);
}

/* Confidence Meter */
.confidence-bar {
  height: 6px;
  border-radius: 3px;
  background: linear-gradient(90deg, #1E2D45 0%, #1E2D45 100%);
  overflow: hidden;
  margin-top: 6px;
}
.confidence-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease;
}

/* Query Variants */
.variant-tag {
  display: inline-block;
  background: rgba(123,47,190,0.15);
  border: 1px solid rgba(123,47,190,0.3);
  color: #A78BFA;
  font-size: 0.7rem;
  padding: 3px 10px;
  border-radius: 20px;
  margin: 2px;
  font-family: 'Space Mono', monospace;
}
.keyword-tag {
  display: inline-block;
  background: rgba(0,212,255,0.08);
  border: 1px solid rgba(0,212,255,0.2);
  color: var(--primary);
  font-size: 0.68rem;
  padding: 2px 8px;
  border-radius: 12px;
  margin: 2px;
  font-family: 'Space Mono', monospace;
}

/* Noise badge */
.noise-high { color: #FF6B35; background: rgba(255,107,53,0.1); border-color: rgba(255,107,53,0.3); }
.noise-low  { color: #00C49A; background: rgba(0,196,154,0.1); border-color: rgba(0,196,154,0.3); }

/* Stats grid */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0.5rem 0; }
.stat-cell {
  background: #080C18;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem;
  text-align: center;
}
.stat-value {
  font-family: 'Space Mono', monospace;
  font-size: 1.1rem;
  color: var(--primary);
  display: block;
}
.stat-label { font-size: 0.65rem; color: var(--text-muted); margin-top: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #080C18 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSlider label { color: var(--text-dim) !important; }

/* Buttons */
.stButton button {
  background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Input */
.stTextArea textarea, .stTextInput input {
  background: var(--bg-input) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 1px rgba(0,212,255,0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--primary) !important; }

/* Metrics */
.stMetric { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; }
.stMetric label { color: var(--text-muted) !important; font-size: 0.7rem !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--primary) !important; font-family: 'Space Mono', monospace !important; }

/* File uploader */
.stFileUploader { background: var(--bg-input) !important; border-color: var(--border) !important; border-radius: 10px !important; }

/* Expander */
.streamlit-expanderHeader { color: var(--text-dim) !important; font-size: 0.8rem !important; }
.streamlit-expanderContent { background: var(--bg-card) !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, var(--primary), var(--secondary)) !important; }

/* Separator */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pipeline": None,
        "query_history": [],
        "last_result": None,
        "docs_loaded": False,
        "loading": False,
        "doc_count": 0,
        "chunk_count": 0,
        "show_viz": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─── Pipeline Initialization ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    hopfield_beta: float,
    top_k: int,
):
    """
    Initialize and cache the retrieval pipeline (embedding + Hopfield memory).

    LLM settings are intentionally excluded from the cache key — they are
    ephemeral UI state that changes without needing to rebuild the index.
    They are passed at query-time via llm_config instead.
    """
    from core.pipeline import HopfieldQAPipeline, PipelineConfig
    config = PipelineConfig(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        hopfield_beta=hopfield_beta,
        top_k=top_k,
        use_fusion=True,
    )
    return HopfieldQAPipeline(config)


def build_llm_config(
    use_llm: bool,
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """
    Build the llm_config dict passed to pipeline.query().

    Returns an empty dict when LLM is disabled so no LLM code is touched.
    """
    if not use_llm:
        return {}
    return {
        "use_llm":     True,
        "provider":    provider,
        "model":       model,
        "api_key":     api_key,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }


# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
         background:linear-gradient(135deg,#00D4FF,#7B2FBE);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         background-clip:text;margin-bottom:0.25rem;">
    🧠 HopfieldQA
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#64748B;
         margin-bottom:1.5rem;">Associative Memory Retriever</div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Model Settings")

    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
        ],
        help="Sentence transformer model for generating embeddings",
        index=0,
    )

    st.markdown("### 📄 Chunking")
    chunk_size = st.slider("Chunk Size (tokens)", 128, 1024, 512, 64)
    chunk_overlap = st.slider("Chunk Overlap", 0, 256, 64, 16)

    st.markdown("### 🌡️ Hopfield Network")
    hopfield_beta = st.slider(
        "β (Inverse Temperature)",
        min_value=1.0, max_value=32.0, value=8.0, step=0.5,
        help="Higher β = sharper, more focused retrieval. Lower = softer, distributed."
    )
    top_k = st.slider("Top-K Chunks", 2, 15, 5)

    st.markdown("### 🔍 Retrieval")
    show_viz = st.toggle("Show Visualizations", value=True)
    st.session_state.show_viz = show_viz

    st.divider()

    # ── LLM Answer Generation ─────────────────────────────────
    st.markdown("### 🤖 LLM Answer Generation")

    use_llm = st.toggle(
        "Enable LLM", value=False,
        help="Use a language model for abstractive answers. "
             "Requires an API key for OpenAI or Anthropic (or no key for Local).",
    )

    # Provider-independent defaults (groq is first in the list)
    llm_provider       = "groq"
    llm_model          = ""
    llm_api_key        = ""
    llm_temperature    = 0.2
    llm_max_new_tokens = 512

    if use_llm:
        from core.llm_provider import (
            validate_provider_config, AVAILABLE_MODELS, DEFAULT_MODELS, ENV_KEYS
        )

        llm_provider = st.selectbox(
            "Provider",
            options=["groq", "anthropic", "openai", "local"],
            format_func=lambda x: {
                "groq":      "⚡ Groq (Fast & Free)",
                "anthropic": "🟣 Anthropic",
                "openai":    "🟢 OpenAI",
                "local":     "🤗 HuggingFace Inference API",
            }[x],
            help="Which LLM backend to use. Each provider is independent — "
                 "only the selected provider's key is checked.",
        )

        # Model list is driven entirely by the selected provider
        llm_model = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS.get(llm_provider, [DEFAULT_MODELS[llm_provider]]),
            help="Model to use for generation.",
        )

        # Key field shown only for API-based providers, labelled per-provider
        if llm_provider == "groq":
            llm_api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="GROQ_API_KEY  (or set env var)",
                help="Leave blank to read from the GROQ_API_KEY environment variable.",
            )
            st.caption(
                "⚡ Groq runs open-source models at extremely fast inference speeds. "
                "Get a **free** API key at [console.groq.com/keys](https://console.groq.com/keys)."
            )
        elif llm_provider == "anthropic":
            llm_api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="ANTHROPIC_API_KEY  (or set env var)",
                help="Leave blank to read from the ANTHROPIC_API_KEY environment variable.",
            )
        elif llm_provider == "openai":
            llm_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="OPENAI_API_KEY  (or set env var)",
                help="Leave blank to read from the OPENAI_API_KEY environment variable.",
            )
        else:  # local — HF Inference API, token optional for public serverless models
            llm_api_key = st.text_input(
                "HuggingFace Token (optional)",
                type="password",
                placeholder="HF_TOKEN  —  leave blank for public serverless models",
                help=(
                    "Uses the HuggingFace Inference API — no local GPU or PyTorch needed. "
                    "A free token from huggingface.co/settings/tokens avoids rate limits "
                    "and is required for gated models."
                ),
            )
            st.caption(
                "💡 Models run on HF serverless endpoints via the Inference API. "
                "No local download required."
            )

        with st.expander("Advanced LLM Settings"):
            llm_temperature = st.slider(
                "Temperature", 0.0, 1.0, 0.2, 0.05,
                help="Lower = more focused / deterministic; higher = more creative.",
            )
            llm_max_new_tokens = st.slider(
                "Max Output Tokens", 128, 1024, 512, 64,
                help="Maximum tokens the LLM will generate.",
            )

        # ── Live provider validation (only for the selected provider) ─────────
        ok, err = validate_provider_config(llm_provider, llm_api_key, llm_model)
        if ok:
            env_note = ""
            if not llm_api_key.strip() and llm_provider != "local":
                env_note = f" (key from {ENV_KEYS[llm_provider]})"
            st.markdown(
                f"""<div style="font-family:'Space Mono',monospace;font-size:0.60rem;
                color:#22c55e;padding:4px 0;">
                ✅ {llm_provider.upper()} · {llm_model}{env_note}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="font-family:'Space Mono',monospace;font-size:0.60rem;
                color:#ef4444;padding:4px 0;">
                ❌ {err}
                </div>""",
                unsafe_allow_html=True,
            )

    # Persist full LLM config in session state so query widgets can read it
    st.session_state["llm_cfg"] = {
        "use_llm":     use_llm,
        "provider":    llm_provider,
        "model":       llm_model,
        "api_key":     llm_api_key,
        "temperature": llm_temperature,
        "max_tokens":  llm_max_new_tokens,
    }
    # Legacy keys kept for answer-display badge
    st.session_state["use_llm"]      = use_llm
    st.session_state["llm_provider"] = llm_provider
    st.session_state["llm_model"]    = llm_model

    # Update beta dynamically
    if st.session_state.pipeline is not None:
        st.session_state.pipeline.update_hopfield_beta(hopfield_beta)

    st.divider()

    # Pipeline Stats
    if st.session_state.docs_loaded and st.session_state.pipeline:
        p = st.session_state.pipeline
        stats = p.get_stats()
        st.markdown("### 📊 Memory Stats")

        pipeline_stats = stats.get("pipeline", {})
        hop_stats = stats.get("hopfield_network", {})

        st.markdown(f"""
        <div class="card" style="padding:0.75rem;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-family:'Space Mono',monospace;font-size:0.65rem;">
            <div style="color:#64748B;">Documents</div>
            <div style="color:#00D4FF;text-align:right;">{pipeline_stats.get('n_documents', 0)}</div>
            <div style="color:#64748B;">Chunks</div>
            <div style="color:#00D4FF;text-align:right;">{pipeline_stats.get('n_chunks', 0)}</div>
            <div style="color:#64748B;">Patterns</div>
            <div style="color:#00D4FF;text-align:right;">{hop_stats.get('n_patterns', 0)}</div>
            <div style="color:#64748B;">Embed Dim</div>
            <div style="color:#00D4FF;text-align:right;">{hop_stats.get('embedding_dim', 0)}</div>
        </div>
        <div style="font-size:0.6rem;color:#7B2FBE;margin-top:8px;font-family:'Space Mono',monospace;">
            Capacity: {hop_stats.get('theoretical_capacity', 'N/A')}
        </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Main Layout ────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">Associative Memory Retriever</p>
  <p class="hero-subtitle">Modern Hopfield Networks for Long-Context Document QA</p>
  <div style="margin-top:0.75rem;">
    <span class="hero-badge">MHN</span>
    <span class="hero-badge">ENERGY MINIMIZATION</span>
    <span class="hero-badge">MULTI-QUERY FUSION</span>
    <span class="hero-badge">NOISE ROBUST</span>
    <span class="hero-badge">LLM SYNTHESIS</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────
tab_qa, tab_docs, tab_theory, tab_history = st.tabs([
    "🔍 Query & Answer",
    "📄 Document Management",
    "🔬 Theory & Architecture",
    "📜 Query History"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: Query & Answer
# ══════════════════════════════════════════════════════════════════
with tab_qa:
    if not st.session_state.docs_loaded:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
          <div style="font-size:2.5rem;margin-bottom:1rem;">🧠</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#E2E8F0;margin-bottom:0.5rem;">
            No Documents in Memory
          </div>
          <div style="color:#64748B;font-size:0.85rem;">
            Load documents in the <strong>Document Management</strong> tab to begin querying.
            <br>Or use the built-in sample documents to try the system immediately.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Query Input ───────────────────────────────────────────
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "Ask a question",
                placeholder="e.g. How do modern Hopfield networks differ from classical ones?",
                label_visibility="collapsed",
            )
        with col_btn:
            search_btn = st.button("⚡ Search", use_container_width=True)

        # Example queries
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#64748B;margin-bottom:4px;">
        EXAMPLE QUERIES:
        </div>
        """, unsafe_allow_html=True)

        ex_cols = st.columns(4)
        examples = [
            "What is the storage capacity of Hopfield networks?",
            "How does RAG work?",
            "Explain transformer attention mechanism",
            "What are the limitations of traditional RAG?",
        ]
        for i, (col, ex) in enumerate(zip(ex_cols, examples)):
            with col:
                if st.button(ex[:35] + "...", key=f"ex_{i}", use_container_width=True):
                    query = ex
                    search_btn = True

        # ── Execute Query ─────────────────────────────────────────
        if search_btn and query and st.session_state.pipeline:
            with st.spinner(""):
                progress_ph = st.empty()
                progress_ph.markdown("""
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00D4FF;
                     animation:pulse 1s infinite;">
                ⟳ Running associative retrieval...
                </div>
                """, unsafe_allow_html=True)

                try:
                    _llm_cfg = st.session_state.get("llm_cfg", {})
                    result = st.session_state.pipeline.query(
                        query,
                        top_k=top_k,
                        include_attention_landscape=show_viz,
                        llm_config=_llm_cfg,
                    )
                    st.session_state.last_result = result
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": result.answer.answer,
                        "confidence": result.answer.confidence,
                        "n_chunks": len(result.retrieved_chunks),
                        "time_ms": result.total_time_ms,
                        "type": result.processed_query.query_type,
                    })
                    progress_ph.empty()
                except Exception as e:
                    progress_ph.empty()
                    st.error(f"Query failed: {e}")
                    result = None

            if result:
                # ── Query Processing Panel ────────────────────────
                pq = result.processed_query
                with st.expander("🔄 Query Processing Details", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Cleaned Query**")
                        st.code(pq.cleaned, language=None)
                    with c2:
                        st.markdown("**Query Type**")
                        type_colors = {
                            "factual": "#00D4FF", "procedural": "#00C49A",
                            "comparative": "#FFB800", "conceptual": "#7B2FBE"
                        }
                        tc = type_colors.get(pq.query_type, "#64748B")
                        st.markdown(f"""
                        <span style="background:rgba(0,0,0,0.3);border:1px solid {tc};
                             color:{tc};padding:4px 12px;border-radius:20px;
                             font-family:'Space Mono',monospace;font-size:0.7rem;">
                        {pq.query_type.upper()}
                        </span>
                        """, unsafe_allow_html=True)

                        noise_cls = "noise-high" if pq.is_noisy else "noise-low"
                        st.markdown(f"""
                        <span class="variant-tag {noise_cls}" style="margin-top:4px;display:inline-block;">
                        {'⚠ NOISY' if pq.is_noisy else '✓ CLEAN'} ({pq.noise_score:.2f})
                        </span>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown("**Query Variants** (for fusion)")
                        for v in pq.variants:
                            st.markdown(f'<span class="variant-tag">{v[:50]}</span>',
                                        unsafe_allow_html=True)

                    st.markdown("**Keywords**")
                    kw_html = "".join(
                        f'<span class="keyword-tag">{kw}</span>' for kw in pq.keywords
                    )
                    st.markdown(kw_html, unsafe_allow_html=True)

                st.divider()

                # ── Performance Metrics ───────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("⚡ Total Time", f"{result.total_time_ms:.0f} ms")
                with m2:
                    st.metric("🔍 Retrieval", f"{result.retrieval_time_ms:.1f} ms")
                with m3:
                    st.metric("📦 Chunks Retrieved", len(result.retrieved_chunks))
                with m4:
                    conf_pct = int(result.answer.confidence * 100)
                    conf_label, _ = ("High", "#00C49A") if result.answer.confidence > 0.6 else \
                                    ("Medium", "#FFB800") if result.answer.confidence > 0.3 else \
                                    ("Low", "#FF6B35")
                    st.metric("🎯 Confidence", f"{conf_pct}% ({conf_label})")

                # ── Answer Box ────────────────────────────────────
                conf_color = "#00C49A" if result.answer.confidence > 0.6 else \
                             "#FFB800" if result.answer.confidence > 0.3 else "#FF6B35"
                conf_width = int(result.answer.confidence * 100)

                gen_stats   = result.answer.generation_stats
                llm_used    = gen_stats.get("llm_used", False)
                llm_err     = gen_stats.get("llm_error", "")
                ans_provider= gen_stats.get("llm_provider", "")
                ans_model   = gen_stats.get("llm_model", "")
                ans_type    = result.answer.answer_type

                if llm_err:
                    st.warning(
                        f"⚠️ LLM call failed — showing extractive answer. "
                        f"Error: `{llm_err[:120]}`"
                    )

                badge_html = ""
                if llm_used:
                    badge_html = (
                        f'<span style="background:rgba(123,47,190,0.15);border:1px solid #7B2FBE;'                        f'color:#7B2FBE;font-family:\'Space Mono\',monospace;font-size:0.58rem;'                        f'padding:2px 8px;border-radius:12px;margin-right:6px;">'
                        f'🤖 {ans_provider.upper()} · {ans_model}</span>'
                    )
                else:
                    badge_html = (
                        '<span style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);'                        'color:#00D4FF;font-family:\'Space Mono\',monospace;font-size:0.58rem;'                        'padding:2px 8px;border-radius:12px;margin-right:6px;">'
                        '⚡ EXTRACTIVE</span>'
                    )

                st.markdown(f"""
                <div class="answer-box">
                  <div style="margin-bottom:0.5rem;">{badge_html}</div>
                  <div class="answer-text">{result.answer.answer}</div>
                  <div style="margin-top:1rem;display:flex;align-items:center;gap:12px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#64748B;">
                      CONFIDENCE
                    </div>
                    <div class="confidence-bar" style="flex:1;">
                      <div class="confidence-fill"
                           style="width:{conf_width}%;background:{conf_color};"></div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:{conf_color};">
                      {conf_pct}%
                    </div>
                  </div>
                  <div style="margin-top:0.75rem;font-family:'Space Mono',monospace;font-size:0.6rem;color:#64748B;">
                    SOURCES: {' · '.join(result.answer.sources[:4])}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Evidence Chunks ───────────────────────────────
                st.markdown("""
                <div class="card-header" style="margin-top:1rem;">📎 Retrieved Evidence</div>
                """, unsafe_allow_html=True)

                for i, chunk in enumerate(result.retrieved_chunks[:top_k]):
                    sim_color = "#00D4FF" if chunk.similarity_score > 0.5 else \
                                "#7B2FBE" if chunk.similarity_score > 0.3 else "#64748B"

                    display_text = chunk.text[:280] + ("..." if len(chunk.text) > 280 else "")

                    # Highlight keywords in display
                    for kw in pq.keywords[:5]:
                        display_text = re.sub(
                            f'({re.escape(kw)})',
                            r'<mark style="background:rgba(0,212,255,0.15);color:#00D4FF;'
                            r'border-radius:2px;padding:0 2px;">\1</mark>',
                            display_text,
                            flags=re.IGNORECASE
                        )

                    st.markdown(f"""
                    <div class="evidence-chunk">
                      <div class="chunk-rank">RANK #{chunk.rank}</div>
                      <div class="chunk-source">📁 {chunk.source} · Chunk {chunk.chunk_index}</div>
                      <div class="chunk-text">{display_text}</div>
                      <div class="chunk-scores">
                        <span class="score-pill" style="color:{sim_color};">
                          cos: {chunk.similarity_score:.3f}
                        </span>
                        <span class="score-pill" style="color:#7B2FBE;">
                          attn: {chunk.hopfield_attention:.4f}
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Visualizations ────────────────────────────────
                if show_viz and result.retrieved_chunks:
                    st.markdown("""
                    <div class="card-header" style="margin-top:1.5rem;">📊 Retrieval Analytics</div>
                    """, unsafe_allow_html=True)

                    from utils.visualization import (
                        plot_similarity_distribution,
                        plot_attention_heatmap,
                        plot_energy_landscape,
                        plot_embedding_scatter,
                    )

                    vc1, vc2 = st.columns(2)

                    with vc1:
                        fig_sim = plot_similarity_distribution(result.retrieved_chunks)
                        st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

                    with vc2:
                        if result.attention_landscape:
                            landscape = result.attention_landscape
                            attn = landscape["attention_weights"]
                            pipeline = st.session_state.pipeline
                            labels = [
                                f"{c.source[:10]}·{c.chunk_index}"
                                for c in pipeline._all_chunks
                            ]
                            fig_attn = plot_attention_heatmap(attn, labels)
                            st.plotly_chart(fig_attn, use_container_width=True,
                                            config={"displayModeBar": False})

                    # Energy landscape
                    if result.attention_landscape:
                        pipeline = st.session_state.pipeline
                        if pipeline._all_chunks and len(pipeline._all_chunks) > 0:
                            q_emb = pipeline.embedder.embed_query(result.query)
                            pat_matrix = np.vstack([c.embedding for c in pipeline._all_chunks])
                            fig_energy = plot_energy_landscape(q_emb, pat_matrix)
                            st.plotly_chart(fig_energy, use_container_width=True,
                                            config={"displayModeBar": False})

                    # Embedding scatter (PCA)
                    try:
                        pipeline = st.session_state.pipeline
                        if len(pipeline._all_chunks) > 3:
                            embs = np.vstack([c.embedding for c in pipeline._all_chunks])
                            q_emb = pipeline.embedder.embed_query(result.query)
                            sources = [c.source for c in pipeline._all_chunks]
                            retrieved_ids = [
                                idx for idx, c in enumerate(pipeline._all_chunks)
                                if c.chunk_id in {r.chunk_id for r in result.retrieved_chunks}
                            ]
                            fig_scatter = plot_embedding_scatter(
                                embs, q_emb, sources, retrieved_ids
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True,
                                            config={"displayModeBar": False})
                    except Exception:
                        pass  # Skip scatter if sklearn not available


# ══════════════════════════════════════════════════════════════════
# TAB 2: Document Management
# ══════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("""
    <div class="card-header">📄 Document Management — Load Documents into Hopfield Memory</div>
    """, unsafe_allow_html=True)

    col_load, col_upload = st.columns(2)

    with col_load:
        st.markdown("#### 📚 Built-in Sample Documents")
        st.markdown("""
        <div style="font-size:0.82rem;color:#94A3B8;margin-bottom:0.75rem;">
        Load curated documents about AI, Hopfield Networks, and RAG systems
        to explore the system immediately.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Load Sample Documents", use_container_width=True):
            with st.spinner("Initializing pipeline..."):
                # Get or create pipeline
                pipeline = get_pipeline(
                    embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k,
                )

                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_cb(msg, pct):
                    status_text.markdown(
                        f'<span style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
                        f'color:#00D4FF;">{msg}</span>',
                        unsafe_allow_html=True
                    )
                    progress_bar.progress(pct)

                results = pipeline.load_sample_documents(progress_callback=progress_cb)
                st.session_state.pipeline = pipeline
                st.session_state.docs_loaded = True

                progress_bar.empty()
                status_text.empty()

            st.success(f"✅ Loaded {len(results)} sample documents!")
            for r in results:
                st.markdown(f"""
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00C49A;">
                ✓ {r.get('source', '?')} → {r.get('n_chunks', 0)} chunks
                </div>
                """, unsafe_allow_html=True)

    with col_upload:
        st.markdown("#### 📎 Upload Your Documents")
        st.markdown("""
        <div style="font-size:0.82rem;color:#94A3B8;margin-bottom:0.75rem;">
        Upload .txt, .pdf, .docx, or .md files to add to the Hopfield memory.
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "md"],
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("⚡ Process & Store Files", use_container_width=True):
            from utils.file_utils import extract_text_from_file

            pipeline = get_pipeline(
                embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k,
            )
            st.session_state.pipeline = pipeline

            progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                try:
                    text, source = extract_text_from_file(f, f.name)
                    result = pipeline.add_document(text, source=source)
                    st.success(f"✅ {source}: {result.get('n_chunks', 0)} chunks")
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")
                progress.progress((i + 1) / len(uploaded_files))

            st.session_state.docs_loaded = True
            st.session_state.pipeline = pipeline

    st.divider()

    # ── Custom Text Input ─────────────────────────────────────────
    st.markdown("#### ✏️ Paste Custom Text")
    custom_text = st.text_area(
        "Paste your document text here",
        height=180,
        placeholder="Paste any text here and give it a name...",
        label_visibility="collapsed",
    )
    custom_name = st.text_input("Document Name", value="custom_doc", max_chars=50)

    if st.button("💾 Add to Memory", use_container_width=False) and custom_text.strip():
        pipeline = get_pipeline(
            embedding_model, chunk_size, chunk_overlap, hopfield_beta, top_k,
        )
        with st.spinner("Processing..."):
            result = pipeline.add_document(custom_text.strip(), source=custom_name)
        st.session_state.pipeline = pipeline
        st.session_state.docs_loaded = True
        st.success(f"✅ Added '{custom_name}': {result.get('n_chunks', 0)} chunks stored in memory")

    # ── Document List ─────────────────────────────────────────────
    if st.session_state.docs_loaded and st.session_state.pipeline:
        st.divider()
        st.markdown("""<div class="card-header">📋 Documents in Memory</div>""",
                    unsafe_allow_html=True)
        pipeline = st.session_state.pipeline
        for source, text in pipeline._documents.items():
            chunks_for_source = [c for c in pipeline._all_chunks if c.source == source]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                 padding:0.6rem 1rem;background:#0A0F1E;border:1px solid #1E2D45;
                 border-radius:8px;margin-bottom:6px;">
              <div>
                <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#E2E8F0;">
                  📁 {source}
                </span>
                <span style="font-size:0.72rem;color:#64748B;margin-left:12px;">
                  {len(text):,} chars
                </span>
              </div>
              <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
                   color:#00D4FF;background:rgba(0,212,255,0.08);
                   border:1px solid rgba(0,212,255,0.2);padding:2px 10px;border-radius:12px;">
                {len(chunks_for_source)} chunks
              </span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All Documents"):
            st.session_state.pipeline.clear()
            st.session_state.docs_loaded = False
            st.session_state.last_result = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 3: Theory & Architecture
# ══════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("""
    <div class="card-header">🔬 Modern Hopfield Networks — Theory</div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        <div class="card">
        <div class="card-header">⚡ Energy Function</div>

        <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#00D4FF;
             background:#080C18;padding:1rem;border-radius:8px;margin-bottom:1rem;
             border-left:3px solid #00D4FF;">
        E = −lse(β, X<sup>T</sup>ξ) + ½ξ<sup>T</sup>ξ + (1/β)log N + ½M²
        </div>

        <div style="font-size:0.82rem;color:#94A3B8;line-height:1.7;">
        <strong style="color:#E2E8F0;">Where:</strong><br>
        • <span style="color:#00D4FF;">X</span> = stored memory patterns (d × N matrix)<br>
        • <span style="color:#00D4FF;">ξ</span> = current state (query vector, d-dim)<br>
        • <span style="color:#00D4FF;">β</span> = inverse temperature (retrieval sharpness)<br>
        • <span style="color:#00D4FF;">lse</span> = log-sum-exp function<br>
        • <span style="color:#00D4FF;">N</span> = number of stored patterns<br>
        • <span style="color:#00D4FF;">M</span> = largest pattern norm
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <div class="card-header">🔄 Update Rule (1-Step Convergence)</div>

        <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#7B2FBE;
             background:#080C18;padding:1rem;border-radius:8px;margin-bottom:1rem;
             border-left:3px solid #7B2FBE;">
        ξ<sub>new</sub> = X · softmax(β · X<sup>T</sup> · ξ)
        </div>

        <div style="font-size:0.82rem;color:#94A3B8;line-height:1.7;">
        This is equivalent to a <strong style="color:#7B2FBE;">single attention head</strong>
        in transformer attention, connecting associative memory directly to
        the attention mechanism.<br><br>
        <strong style="color:#E2E8F0;">Key insight:</strong> For well-separated patterns,
        convergence occurs in <em>exactly one step</em> — making retrieval O(N·d).
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="card">
        <div class="card-header">📊 Capacity Comparison</div>
        <table style="width:100%;font-size:0.78rem;border-collapse:collapse;">
          <tr style="border-bottom:1px solid #1E2D45;">
            <th style="text-align:left;color:#64748B;padding:6px 0;font-weight:500;">Property</th>
            <th style="text-align:center;color:#64748B;padding:6px;">Classical</th>
            <th style="text-align:center;color:#00D4FF;padding:6px;">Modern</th>
          </tr>
          <tr style="border-bottom:1px solid #1E2D45;">
            <td style="color:#94A3B8;padding:6px 0;">State Space</td>
            <td style="text-align:center;color:#E2E8F0;">Binary {-1,+1}</td>
            <td style="text-align:center;color:#00D4FF;">Continuous ℝ<sup>d</sup></td>
          </tr>
          <tr style="border-bottom:1px solid #1E2D45;">
            <td style="color:#94A3B8;padding:6px 0;">Capacity</td>
            <td style="text-align:center;color:#E2E8F0;">0.14 · N</td>
            <td style="text-align:center;color:#00D4FF;">exp(d/2)</td>
          </tr>
          <tr style="border-bottom:1px solid #1E2D45;">
            <td style="color:#94A3B8;padding:6px 0;">Convergence</td>
            <td style="text-align:center;color:#E2E8F0;">Many steps</td>
            <td style="text-align:center;color:#00D4FF;">1 step</td>
          </tr>
          <tr style="border-bottom:1px solid #1E2D45;">
            <td style="color:#94A3B8;padding:6px 0;">Energy</td>
            <td style="text-align:center;color:#E2E8F0;">Quadratic</td>
            <td style="text-align:center;color:#00D4FF;">log-sum-exp</td>
          </tr>
          <tr>
            <td style="color:#94A3B8;padding:6px 0;">Attention Link</td>
            <td style="text-align:center;color:#E2E8F0;">❌</td>
            <td style="text-align:center;color:#00C49A;">✓ Exact</td>
          </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <div class="card-header">🏗️ System Architecture</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#94A3B8;
             line-height:2.2;">
        <span style="color:#00D4FF;">Document</span>
          → <span style="color:#E2E8F0;">Chunker</span>
          → <span style="color:#00D4FF;">Embedder</span>
          → <span style="color:#7B2FBE;">Hopfield Memory</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑<br>
        <span style="color:#FF6B35;">Query</span>
          → <span style="color:#E2E8F0;">QueryProcessor</span>
          → <span style="color:#E2E8F0;">Variants</span>
          → <span style="color:#7B2FBE;">Energy Min.</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#E2E8F0;">Top-K Chunks</span>
          → <span style="color:#00C49A;">RRF Fusion</span>
          → <span style="color:#E2E8F0;">AnswerGen</span>
          → <span style="color:#00D4FF;">Answer</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:0.5rem;">
    <div class="card-header">📖 Key References</div>
    <div style="font-size:0.8rem;color:#94A3B8;line-height:1.8;">
    • <strong style="color:#E2E8F0;">Ramsauer et al. (2021)</strong> — "Hopfield Networks is All You Need"
      <em style="color:#64748B;">ICLR 2021</em><br>
    • <strong style="color:#E2E8F0;">Hopfield (1982)</strong> — "Neural networks and physical systems with
      emergent collective computational abilities" <em style="color:#64748B;">PNAS</em><br>
    • <strong style="color:#E2E8F0;">Lewis et al. (2020)</strong> — "Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks" <em style="color:#64748B;">NeurIPS 2020</em><br>
    • <strong style="color:#E2E8F0;">Vaswani et al. (2017)</strong> — "Attention is All You Need"
      <em style="color:#64748B;">NeurIPS 2017</em>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: Query History
# ══════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("""
    <div class="card-header">📜 Query History</div>
    """, unsafe_allow_html=True)

    if not st.session_state.query_history:
        st.markdown("""
        <div style="text-align:center;color:#64748B;font-size:0.85rem;padding:2rem;">
        No queries yet. Ask something in the Query & Answer tab!
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()

        for i, item in enumerate(reversed(st.session_state.query_history)):
            conf_c = "#00C49A" if item["confidence"] > 0.6 else \
                     "#FFB800" if item["confidence"] > 0.3 else "#FF6B35"
            st.markdown(f"""
            <div class="card" style="margin-bottom:0.75rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                <div style="font-size:0.85rem;color:#E2E8F0;font-weight:500;">
                  {len(st.session_state.query_history) - i}. {item['query']}
                </div>
                <div style="display:flex;gap:6px;align-items:center;">
                  <span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                       background:rgba(0,0,0,0.3);color:#7B2FBE;padding:2px 8px;
                       border-radius:10px;">{item['type']}</span>
                  <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
                       color:{conf_c};">{int(item['confidence']*100)}%</span>
                  <span style="font-size:0.65rem;color:#64748B;">{item['time_ms']:.0f}ms</span>
                </div>
              </div>
              <div style="font-size:0.8rem;color:#94A3B8;line-height:1.6;">
                {item['answer'][:300]}{'...' if len(item['answer']) > 300 else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1rem;
     font-family:'Space Mono',monospace;font-size:0.6rem;color:#1E2D45;">
HOPFIELD QA · MODERN HOPFIELD NETWORKS · ASSOCIATIVE MEMORY RETRIEVAL
</div>
""", unsafe_allow_html=True)
