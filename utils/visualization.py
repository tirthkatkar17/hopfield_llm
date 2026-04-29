"""
Utilities: Visualization helpers for Streamlit UI
"""

import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
from core.hopfield import RetrievalResult


# ─── Color Palette ────────────────────────────────────────────────
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#7B2FBE",
    "accent": "#FF6B35",
    "success": "#00C49A",
    "warning": "#FFB800",
    "bg_dark": "#0A0E1A",
    "bg_card": "#111827",
    "text": "#E2E8F0",
    "muted": "#64748B",
}


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    chunk_labels: List[str],
    title: str = "Hopfield Attention Distribution",
) -> go.Figure:
    """
    Visualize attention weights across stored patterns.
    """
    top_k = min(30, len(attention_weights))
    top_indices = np.argsort(attention_weights)[::-1][:top_k]
    top_weights = attention_weights[top_indices]
    top_labels = [chunk_labels[i] if i < len(chunk_labels) else f"Chunk {i}"
                  for i in top_indices]

    # Normalize for display
    normalized = top_weights / (top_weights.max() + 1e-8)

    fig = go.Figure(go.Bar(
        x=normalized,
        y=top_labels,
        orientation='h',
        marker=dict(
            color=normalized,
            colorscale=[[0, '#1a1f35'], [0.5, '#7B2FBE'], [1.0, '#00D4FF']],
            showscale=True,
            colorbar=dict(title="Attention", thickness=10),
        ),
        hovertemplate="<b>%{y}</b><br>Attention: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"], size=10),
        height=max(300, top_k * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor="#1E293B", title="Normalized Attention"),
    )
    return fig


def plot_similarity_distribution(
    retrieved: List[RetrievalResult],
    title: str = "Retrieval Score Distribution",
) -> go.Figure:
    """Bar chart of similarity scores for retrieved chunks."""
    if not retrieved:
        return go.Figure()

    labels = [f"#{r.rank} {r.source[:15]}" for r in retrieved]
    similarities = [r.similarity_score for r in retrieved]
    attentions = [r.hopfield_attention for r in retrieved]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Cosine Similarity",
        x=labels,
        y=similarities,
        marker_color=COLORS["primary"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Similarity: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Hopfield Attention",
        x=labels,
        y=[a * 10 for a in attentions],  # scale for visibility
        marker_color=COLORS["accent"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Attention×10: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"], size=11),
        barmode="group",
        height=280,
        margin=dict(l=10, r=10, t=40, b=60),
        legend=dict(
            bgcolor=COLORS["bg_dark"],
            bordercolor="#1E293B",
            font=dict(size=10),
        ),
        xaxis=dict(tickangle=-30, gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B", title="Score"),
    )
    return fig


def plot_energy_landscape(
    query_embedding: np.ndarray,
    pattern_matrix: np.ndarray,
    beta_values: List[float] = None,
) -> go.Figure:
    """
    Visualize how Hopfield energy changes with different β values.
    Shows the retrieval sharpness trade-off.
    """
    if beta_values is None:
        beta_values = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    # Compute similarities
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    p_norm = pattern_matrix / (np.linalg.norm(pattern_matrix, axis=1, keepdims=True) + 1e-8)
    sims = p_norm @ q_norm  # (N,)

    entropy_values = []
    max_attn_values = []

    for beta in beta_values:
        logits = beta * sims
        logits_stable = logits - logits.max()
        attn = np.exp(logits_stable)
        attn = attn / attn.sum()

        # Entropy (lower = sharper)
        entropy = -np.sum(attn * np.log(attn + 1e-10))
        entropy_values.append(entropy)
        max_attn_values.append(attn.max())

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=beta_values, y=entropy_values,
        name="Attention Entropy",
        line=dict(color=COLORS["primary"], width=2.5),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="β=%{x}<br>Entropy: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=beta_values, y=max_attn_values,
        name="Max Attention",
        line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=7),
        yaxis="y2",
        hovertemplate="β=%{x}<br>Max Attn: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Hopfield Energy Landscape: β (Inverse Temperature) Effect",
            font=dict(color=COLORS["text"], size=13)
        ),
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"], size=11),
        height=280,
        margin=dict(l=10, r=60, t=50, b=40),
        legend=dict(bgcolor=COLORS["bg_dark"], bordercolor="#1E293B"),
        xaxis=dict(title="β (Inverse Temperature)", gridcolor="#1E293B"),
        yaxis=dict(title="Entropy", gridcolor="#1E293B", color=COLORS["primary"]),
        yaxis2=dict(
            title="Max Attention",
            overlaying="y", side="right",
            color=COLORS["accent"], showgrid=False,
        ),
    )
    return fig


def plot_embedding_scatter(
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    sources: List[str],
    retrieved_indices: List[int] = None,
    title: str = "Embedding Space (PCA 2D)",
) -> go.Figure:
    """
    2D PCA scatter of chunk embeddings and query.
    Highlights retrieved chunks.
    """
    from sklearn.decomposition import PCA

    all_embs = np.vstack([chunk_embeddings, query_embedding.reshape(1, -1)])
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(all_embs)

    chunk_2d = reduced[:-1]
    query_2d = reduced[-1]

    # Color by source
    unique_sources = list(set(sources))
    source_colors = px.colors.qualitative.Pastel
    color_map = {s: source_colors[i % len(source_colors)]
                 for i, s in enumerate(unique_sources)}
    colors_list = [color_map[s] for s in sources]

    # Mark retrieved
    retrieved_set = set(retrieved_indices or [])
    sizes = [12 if i in retrieved_set else 6 for i in range(len(sources))]
    symbols = ["star" if i in retrieved_set else "circle" for i in range(len(sources))]

    fig = go.Figure()

    # Chunks
    for src in unique_sources:
        mask = [s == src for s in sources]
        idxs = [i for i, m in enumerate(mask) if m]
        fig.add_trace(go.Scatter(
            x=chunk_2d[idxs, 0],
            y=chunk_2d[idxs, 1],
            mode="markers",
            name=src[:20],
            marker=dict(
                color=color_map[src],
                size=[sizes[i] for i in idxs],
                symbol=[symbols[i] for i in idxs],
                opacity=0.8,
                line=dict(color="white", width=1),
            ),
            hovertemplate=f"Source: {src}<br>Chunk %{{text}}<extra></extra>",
            text=[str(i) for i in idxs],
        ))

    # Query point
    fig.add_trace(go.Scatter(
        x=[query_2d[0]], y=[query_2d[1]],
        mode="markers",
        name="Query",
        marker=dict(
            color=COLORS["accent"],
            size=16,
            symbol="x",
            line=dict(color="white", width=2),
        ),
        hovertemplate="QUERY<extra></extra>",
    ))

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(
            text=f"{title} (PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%})",
            font=dict(color=COLORS["text"], size=13),
        ),
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_dark"],
        font=dict(color=COLORS["text"], size=10),
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor=COLORS["bg_dark"], font=dict(size=9)),
        xaxis=dict(gridcolor="#1E293B", zeroline=False),
        yaxis=dict(gridcolor="#1E293B", zeroline=False),
    )
    return fig


def confidence_badge(confidence: float) -> tuple:
    """Return (label, color) for confidence display."""
    if confidence >= 0.65:
        return "High Confidence", COLORS["success"]
    elif confidence >= 0.35:
        return "Medium Confidence", COLORS["warning"]
    else:
        return "Low Confidence", COLORS["accent"]


def format_chunk_display(result: RetrievalResult, keywords: List[str]) -> str:
    """Format a retrieved chunk for display with keyword highlighting."""
    text = result.text
    # Bold keywords
    for kw in keywords[:5]:
        text = re.sub(
            f'({re.escape(kw)})',
            r'**\1**',
            text,
            flags=re.IGNORECASE
        )
    return text
