"""
Modern Hopfield Network Retrieval Module
==========================================
Implements associative memory retrieval using Modern Hopfield Networks (MHN).

Key Paper: "Hopfield Networks is All You Need" (Ramsauer et al., 2021)
The energy function: E = -lse(β, Xᵀξ) + 0.5 ξᵀξ + (1/β) log N + 0.5 M²

Classical Hopfield: Binary patterns, limited capacity O(N)
Modern Hopfield:    Continuous patterns, exponential capacity exp(N/2)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class RetrievalResult:
    """A single retrieval result from the Hopfield memory."""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    similarity_score: float
    hopfield_attention: float
    rank: int
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModernHopfieldNetwork:
    """
    Modern Hopfield Network for associative memory retrieval.

    The network stores memory patterns (document embeddings) and retrieves
    the most relevant patterns given a query via energy minimization.

    The update rule converges in ONE step for well-separated patterns:
        ξ_new = X · softmax(β · Xᵀ · ξ)

    Where:
        X   = stored memory patterns (d × N matrix)
        ξ   = query / state vector (d-dimensional)
        β   = inverse temperature (controls retrieval sharpness)
        N   = number of stored patterns

    Storage Capacity: exp(d/2) patterns (exponential in embedding dim d)
    """

    def __init__(
        self,
        beta: float = 8.0,
        n_iter: int = 1,
        normalize_patterns: bool = True,
        similarity_threshold: float = 0.0,
    ):
        """
        Args:
            beta: Inverse temperature. Higher = sharper, more focused retrieval.
                  Lower = softer, more distributed retrieval.
                  Range: 1.0 (soft) to 32.0 (very sharp)
            n_iter: Number of Hopfield update iterations (1 usually sufficient).
            normalize_patterns: Normalize stored patterns to unit sphere.
            similarity_threshold: Minimum similarity score for retrieval.
        """
        self.beta = beta
        self.n_iter = n_iter
        self.normalize_patterns = normalize_patterns
        self.similarity_threshold = similarity_threshold

        # Memory storage
        self._patterns: Optional[np.ndarray] = None  # Shape: (N, d)
        self._pattern_norms: Optional[np.ndarray] = None
        self._metadata: List[Dict] = []
        self._is_built = False

    @property
    def n_patterns(self) -> int:
        return len(self._metadata) if self._metadata else 0

    @property
    def embedding_dim(self) -> int:
        return self._patterns.shape[1] if self._patterns is not None else 0

    @property
    def theoretical_capacity(self) -> float:
        """Theoretical storage capacity of the network."""
        if self._patterns is None:
            return 0.0
        d = self.embedding_dim
        # exp(d/2) overflows for d>=768; clamp exponent to avoid inf
        exponent = d / 2
        if exponent > 700:
            return float("inf")
        return float(np.exp(exponent))

    def store(self, patterns: np.ndarray, metadata: List[Dict]) -> None:
        """
        Store memory patterns in the Hopfield network.

        Args:
            patterns: Array of shape (N, d) — document embeddings.
            metadata: List of N metadata dicts for each pattern.
        """
        assert len(patterns) == len(metadata), \
            f"Patterns ({len(patterns)}) and metadata ({len(metadata)}) must match."

        if self.normalize_patterns:
            norms = np.linalg.norm(patterns, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            self._patterns = patterns / norms
        else:
            self._patterns = patterns.copy()

        self._pattern_norms = np.linalg.norm(self._patterns, axis=1)
        self._metadata = metadata
        self._is_built = True

    def _log_sum_exp(self, x: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        c = x.max()
        return c + np.log(np.exp(x - c).sum())

    def _hopfield_energy(self, query: np.ndarray) -> float:
        """
        Compute Hopfield energy for a query:
        E = -lse(β, Xᵀξ) + 0.5 ξᵀξ + (1/β) log N + 0.5 M²
        """
        if not self._is_built:
            return 0.0
        similarities = self._patterns @ query  # (N,)
        lse = self._log_sum_exp(self.beta * similarities)
        xi_norm_sq = np.dot(query, query)
        M_sq = np.max(self._pattern_norms) ** 2
        energy = -lse + 0.5 * xi_norm_sq + (1.0 / self.beta) * np.log(self.n_patterns) + 0.5 * M_sq
        return float(energy)

    def _hopfield_update(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        One Hopfield update step (Modern Hopfield retrieval).

        Returns:
            new_state: Updated query state (d,)
            attention_weights: Softmax attention over patterns (N,)
        """
        # Compute similarities: (N,)
        similarities = self._patterns @ query

        # Softmax with temperature β
        logits = self.beta * similarities
        logits_stable = logits - logits.max()
        attention = np.exp(logits_stable)
        attention = attention / (attention.sum() + 1e-10)

        # New state: weighted sum of patterns
        new_state = self._patterns.T @ attention  # (d,)

        # Normalize new state
        norm = np.linalg.norm(new_state)
        if norm > 1e-8:
            new_state = new_state / norm

        return new_state, attention

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k most relevant patterns for a query.

        The retrieval process:
        1. Initialize state ξ from query embedding
        2. Run n_iter Hopfield update steps
        3. Compute final attention weights
        4. Return top-k patterns by attention weight

        Args:
            query_embedding: Query vector of shape (d,).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        if not self._is_built or self._patterns is None:
            return []

        # Normalize query
        query = query_embedding.copy()
        norm = np.linalg.norm(query)
        if norm > 1e-8:
            query = query / norm

        # Iterative Hopfield updates
        state = query
        final_attention = None

        for _ in range(self.n_iter):
            state, attention = self._hopfield_update(state)
            final_attention = attention

        # Compute cosine similarity between final state and patterns
        cos_similarities = self._patterns @ query  # raw cosine sim with original query

        # Get top-k by attention weight
        top_k = min(top_k, self.n_patterns)
        top_indices = np.argsort(final_attention)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            meta = self._metadata[idx]
            sim_score = float(cos_similarities[idx])
            hop_attn = float(final_attention[idx])

            if sim_score < self.similarity_threshold:
                continue

            results.append(RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=sim_score,
                hopfield_attention=hop_attn,
                rank=rank + 1,
                metadata=meta,
            ))

        return results

    def retrieve_with_fusion(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 5,
        fusion_weights: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Multi-query fusion retrieval — robust to noisy/ambiguous queries.

        Retrieves using multiple query variants and fuses scores via
        Reciprocal Rank Fusion (RRF).

        Args:
            query_embeddings: List of query embedding variants.
            top_k: Number of results to return.
            fusion_weights: Optional weights per query variant.

        Returns:
            Fused and re-ranked list of RetrievalResult objects.
        """
        if not query_embeddings:
            return []

        if fusion_weights is None:
            fusion_weights = [1.0] * len(query_embeddings)

        # Normalize weights
        total = sum(fusion_weights)
        fusion_weights = [w / total for w in fusion_weights]

        # Collect per-query results
        all_results: Dict[str, Dict] = {}
        rrf_k = 60  # RRF constant

        for q_idx, (q_emb, q_weight) in enumerate(zip(query_embeddings, fusion_weights)):
            results = self.retrieve(q_emb, top_k=top_k * 2)

            for result in results:
                cid = result.chunk_id
                rrf_score = q_weight / (rrf_k + result.rank)

                if cid not in all_results:
                    all_results[cid] = {
                        "result": result,
                        "rrf_score": 0.0,
                        "max_similarity": result.similarity_score,
                    }
                all_results[cid]["rrf_score"] += rrf_score
                all_results[cid]["max_similarity"] = min(
                    max(all_results[cid]["max_similarity"], result.similarity_score),
                    1.0,
                )

        # Sort by fused RRF score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        fused = []
        for rank, item in enumerate(sorted_results):
            r = item["result"]
            r.rank = rank + 1
            r.hopfield_attention = item["rrf_score"]
            r.similarity_score = item["max_similarity"]
            fused.append(r)

        return fused

    def get_attention_landscape(
        self, query_embedding: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute full attention landscape for visualization.

        Returns attention weights across ALL stored patterns,
        useful for understanding retrieval focus.
        """
        if not self._is_built:
            return {}

        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        _, attention = self._hopfield_update(query)
        similarities = self._patterns @ query

        return {
            "attention_weights": attention,
            "cosine_similarities": similarities,
            "pattern_indices": np.arange(self.n_patterns),
        }

    def get_network_stats(self) -> Dict:
        """Return statistics about the stored network."""
        if not self._is_built:
            return {"status": "empty"}

        similarities_matrix = self._patterns @ self._patterns.T
        np.fill_diagonal(similarities_matrix, 0)
        avg_sim = similarities_matrix.mean()
        max_sim = similarities_matrix.max()

        return {
            "n_patterns": self.n_patterns,
            "embedding_dim": self.embedding_dim,
            "beta": self.beta,
            "n_iter": self.n_iter,
            "theoretical_capacity": "~∞" if self.theoretical_capacity == float("inf") else f"~{self.theoretical_capacity:.2e}",
            "avg_inter_pattern_similarity": float(avg_sim),
            "max_inter_pattern_similarity": float(max_sim),
            "capacity_utilization": "≈0%" if self.theoretical_capacity == float("inf") else f"{(self.n_patterns / self.theoretical_capacity) * 100:.6f}%",
        }
