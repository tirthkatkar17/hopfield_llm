"""
Main Pipeline Orchestrator
===========================
Ties together: Embeddings → Hopfield Memory → Query Processing → Answer Generation
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

from core.embeddings import DocumentChunker, EmbeddingEngine, Chunk
from core.hopfield import ModernHopfieldNetwork, RetrievalResult
from core.query_handler import QueryProcessor, ProcessedQuery
from core.answer_generator import AnswerGenerator, LLMAnswerGenerator, GeneratedAnswer
from core.llm_provider import validate_provider_config, DEFAULT_MODELS, AVAILABLE_MODELS


@dataclass
class PipelineConfig:
    """Configuration for the retrieval pipeline (embedding + Hopfield).

    LLM settings are intentionally NOT stored here — they are ephemeral UI
    state that must be applied at query time, not at pipeline-construction
    time.  This means the @st.cache_resource-cached pipeline object is
    never stale with respect to provider or key changes.
    """
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Hopfield
    hopfield_beta: float = 8.0
    hopfield_n_iter: int = 1
    similarity_threshold: float = 0.05

    # Retrieval
    top_k: int = 5
    use_fusion: bool = True
    n_query_variants: int = 3

    # Answer generation (extractive baseline)
    max_answer_length: int = 600
    min_confidence: float = 0.1

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class QueryResult:
    """Complete result of a pipeline query."""
    query: str
    processed_query: ProcessedQuery
    retrieved_chunks: List[RetrievalResult]
    answer: GeneratedAnswer
    retrieval_time_ms: float
    total_time_ms: float
    attention_landscape: Optional[Dict] = None


class HopfieldQAPipeline:
    """
    End-to-end QA pipeline using Modern Hopfield Networks.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │  Document Ingestion                                    │
    │  ─────────────────────────────────────────────────     │
    │  Text → Chunks → Embeddings → Hopfield Memory Store   │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Query Processing                                      │
    │  ─────────────────────────────────────────────────     │
    │  Raw Query → Clean → Expand → Variants               │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Hopfield Retrieval                                    │
    │  ─────────────────────────────────────────────────     │
    │  Query Embeddings → Energy Minimization → Top-K      │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Answer Generation                                     │
    │  ─────────────────────────────────────────────────     │
    │  Chunks → Sentence Extraction → Answer + Confidence  │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        )
        self.embedder = EmbeddingEngine(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )
        self.memory = ModernHopfieldNetwork(
            beta=self.config.hopfield_beta,
            n_iter=self.config.hopfield_n_iter,
            similarity_threshold=self.config.similarity_threshold,
        )
        self.query_processor = QueryProcessor()

        # Always start with the extractive generator.
        # LLM is injected at query-time via query(..., llm_config=...) so that
        # switching providers in the UI never requires rebuilding the pipeline.
        self.answer_generator = AnswerGenerator(
            max_answer_length=self.config.max_answer_length,
            min_confidence_threshold=self.config.min_confidence,
        )

        # State
        self._all_chunks: List[Chunk] = []
        self._documents: Dict[str, str] = {}  # source → original text
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.memory.n_patterns > 0

    @property
    def n_documents(self) -> int:
        return len(self._documents)

    @property
    def n_chunks(self) -> int:
        return len(self._all_chunks)

    def add_document(
        self,
        text: str,
        source: str = "document",
        progress_callback=None,
    ) -> Dict:
        """
        Add a document to the Hopfield memory.

        Process:
        1. Chunk the text
        2. Embed chunks
        3. Store in Hopfield network

        Returns stats about the added document.
        """
        if not text.strip():
            return {"error": "Empty document text"}

        t0 = time.time()

        # Chunk
        if progress_callback:
            progress_callback("Chunking document...", 0.2)
        chunks = self.chunker.chunk_text(text, source=source)

        # Embed
        if progress_callback:
            progress_callback("Generating embeddings...", 0.5)
        chunks = self.embedder.embed_chunks(chunks, show_progress=False)

        # Store original
        self._documents[source] = text
        self._all_chunks.extend(chunks)

        # Rebuild Hopfield memory with all chunks
        if progress_callback:
            progress_callback("Storing in Hopfield memory...", 0.8)
        self._rebuild_memory()

        elapsed = (time.time() - t0) * 1000
        self._is_ready = True

        if progress_callback:
            progress_callback("Done!", 1.0)

        return {
            "source": source,
            "n_chunks": len(chunks),
            "n_total_chunks": self.n_chunks,
            "embedding_dim": self.embedder.embedding_dim,
            "processing_time_ms": elapsed,
        }

    def add_documents(
        self,
        documents: List[Tuple[str, str]],  # [(text, source), ...]
        progress_callback=None,
    ) -> List[Dict]:
        """Add multiple documents efficiently — rebuilds memory only once at the end."""
        results = []
        t0 = time.time()

        for i, (text, source) in enumerate(documents):
            if not text.strip():
                results.append({"error": f"Empty document: {source}"})
                continue

            if progress_callback:
                progress_callback(
                    f"Chunking & embedding {source}...",
                    (i + 0.5) / len(documents)
                )

            chunks = self.chunker.chunk_text(text, source=source)
            chunks = self.embedder.embed_chunks(chunks, show_progress=False)

            self._documents[source] = text
            self._all_chunks.extend(chunks)

            results.append({
                "source": source,
                "n_chunks": len(chunks),
                "embedding_dim": self.embedder.embedding_dim,
            })

        # Rebuild memory once for all documents
        if progress_callback:
            progress_callback("Storing all chunks in Hopfield memory...", 0.95)
        self._rebuild_memory()
        self._is_ready = True

        total_ms = (time.time() - t0) * 1000
        for r in results:
            if "error" not in r:
                r["n_total_chunks"] = self.n_chunks
                r["processing_time_ms"] = total_ms / max(len(results), 1)

        return results

    def _rebuild_memory(self):
        """Rebuild Hopfield memory from all stored chunks."""
        if not self._all_chunks:
            return

        # Stack all embeddings
        patterns = np.vstack([c.embedding for c in self._all_chunks])

        # Build metadata list
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in self._all_chunks
        ]

        self.memory.store(patterns, metadata)

    def query(
        self,
        raw_query: str,
        top_k: Optional[int] = None,
        include_attention_landscape: bool = False,
        llm_config: Optional[Dict] = None,
    ) -> QueryResult:
        """
        Execute a complete query through the pipeline.

        Args:
            raw_query: User's raw query string.
            top_k: Number of chunks to retrieve (defaults to config).
            include_attention_landscape: Whether to include full attention map.
            llm_config: Optional dict with LLM settings for this query:
                {
                  "use_llm":      bool,
                  "provider":     str,   # "openai" | "anthropic" | "local"
                  "model":        str,
                  "api_key":      str,
                  "temperature":  float,
                  "max_tokens":   int,
                }
                When provided, a fresh LLMAnswerGenerator is constructed for
                exactly this provider — no stale state from previous calls.

        Returns:
            QueryResult with answer, evidence, and performance stats.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Please add documents first.")

        k = top_k or self.config.top_k
        t0 = time.time()

        # 1. Process query
        processed = self.query_processor.process(raw_query)

        # 2. Embed query variants
        variant_embeddings = []
        for variant in processed.variants[:self.config.n_query_variants]:
            emb = self.embedder.embed_query(variant)
            variant_embeddings.append(emb)

        t_retrieval_start = time.time()

        # 3. Retrieve via Hopfield network
        if self.config.use_fusion and len(variant_embeddings) > 1:
            fusion_weights = [1.0] + [0.7] * (len(variant_embeddings) - 1)
            retrieved = self.memory.retrieve_with_fusion(
                variant_embeddings,
                top_k=k,
                fusion_weights=fusion_weights,
            )
        else:
            retrieved = self.memory.retrieve(variant_embeddings[0], top_k=k)

        retrieval_time = (time.time() - t_retrieval_start) * 1000

        # 4. Optional: attention landscape
        landscape = None
        if include_attention_landscape and variant_embeddings:
            landscape = self.memory.get_attention_landscape(variant_embeddings[0])

        # 5. Select answer generator
        #    Build a fresh LLMAnswerGenerator each time LLM is requested so
        #    that provider/key changes take effect immediately without any
        #    cached object needing to be invalidated.
        if llm_config and llm_config.get("use_llm"):
            provider = llm_config.get("provider", "anthropic")
            model    = llm_config.get("model", "") or DEFAULT_MODELS.get(provider, "")
            generator = LLMAnswerGenerator(
                llm_provider=provider,
                model_name=model,
                api_key=llm_config.get("api_key", ""),
                temperature=llm_config.get("temperature", 0.2),
                max_new_tokens=llm_config.get("max_tokens", 512),
                max_answer_length=self.config.max_answer_length,
                min_confidence_threshold=self.config.min_confidence,
            )
        else:
            generator = self.answer_generator   # plain extractive

        answer = generator.generate(processed, retrieved)

        total_time = (time.time() - t0) * 1000

        return QueryResult(
            query=raw_query,
            processed_query=processed,
            retrieved_chunks=retrieved,
            answer=answer,
            retrieval_time_ms=retrieval_time,
            total_time_ms=total_time,
            attention_landscape=landscape,
        )

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            "pipeline": {
                "n_documents": self.n_documents,
                "n_chunks": self.n_chunks,
                "is_ready": self.is_ready,
                "embedding_model": self.config.embedding_model,
            },
            "hopfield_network": self.memory.get_network_stats(),
            "embedding_cache": self.embedder.cache_stats(),
            "config": self.config.to_dict(),
        }
        return stats

    def update_hopfield_beta(self, new_beta: float):
        """Dynamically update Hopfield temperature parameter."""
        self.memory.beta = new_beta

    def clear(self):
        """Clear all documents and reset the pipeline."""
        self._all_chunks = []
        self._documents = {}
        self._is_ready = False
        self.memory = ModernHopfieldNetwork(
            beta=self.config.hopfield_beta,
            n_iter=self.config.hopfield_n_iter,
        )
        self.embedder.clear_cache()

    def load_sample_documents(self, progress_callback=None) -> List[Dict]:
        """Load built-in sample documents for demo purposes."""
        samples = [
            (SAMPLE_DOC_AI, "AI_Overview.txt"),
            (SAMPLE_DOC_HOPFIELD, "Hopfield_Networks.txt"),
            (SAMPLE_DOC_RAG, "RAG_Systems.txt"),
        ]
        return self.add_documents(samples, progress_callback=progress_callback)


# ─────────────────────────────────────────────────────────────
# Sample documents for demonstration
# ─────────────────────────────────────────────────────────────

SAMPLE_DOC_AI = """
Artificial Intelligence: An Overview

Artificial intelligence (AI) refers to the simulation of human intelligence processes by
computer systems. These processes include learning (the acquisition of information and rules
for using the information), reasoning (using rules to reach approximate or definite
conclusions), and self-correction.

Machine Learning is a subset of AI that provides systems the ability to automatically
learn and improve from experience without being explicitly programmed. Machine learning
focuses on the development of computer programs that can access data and use it to learn
for themselves.

Deep Learning is part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural
networks, convolutional neural networks, and transformers have been applied to fields
including computer vision, natural language processing, and speech recognition.

Natural Language Processing (NLP) is a subfield of linguistics, computer science, and
artificial intelligence concerned with the interactions between computers and human
language, in particular how to program computers to process and analyze large amounts of
natural language data. NLP combines computational linguistics—rule-based modeling of
human language—with statistical, machine learning, and deep learning models.

Large Language Models (LLMs) are a type of AI system trained on vast amounts of text data.
They can generate text, answer questions, translate languages, and perform many other
language tasks. Examples include GPT-4, Claude, and LLaMA. These models have billions of
parameters and are trained using transformer architectures with attention mechanisms.

Transformers, introduced in "Attention is All You Need" (Vaswani et al., 2017), are
sequence-to-sequence models that use self-attention mechanisms. They have become the
dominant architecture for NLP tasks and have been extended to vision, audio, and
multimodal applications.

Reinforcement Learning from Human Feedback (RLHF) is a technique used to align language
models with human preferences. The model generates outputs, human raters evaluate them,
and the ratings are used to train a reward model. The language model is then fine-tuned
using reinforcement learning to maximize the reward.
"""

SAMPLE_DOC_HOPFIELD = """
Hopfield Networks and Associative Memory

Hopfield Networks are a form of recurrent neural network that serve as content-addressable
memory systems with binary threshold nodes. They were introduced by John Hopfield in 1982
and are used to model associative memory in biological neural systems.

Classical Hopfield Networks store binary patterns and can retrieve a stored pattern when
presented with a partial or noisy version. The network has a storage capacity of approximately
0.14 * N patterns, where N is the number of neurons. Retrieval works by minimizing an energy
function through iterative state updates.

The energy function of a Hopfield network is defined as:
E = -0.5 * Σᵢⱼ wᵢⱼ sᵢ sⱼ
where wᵢⱼ are the weights (connection strengths) and sᵢ, sⱼ are the neuron states.

Modern Hopfield Networks (MHN), introduced by Ramsauer et al. in 2021, extend classical
Hopfield networks to continuous states and exponentially increase storage capacity.
The storage capacity of modern Hopfield networks scales as exp(d/2) where d is the
dimensionality of the stored patterns — an exponential improvement over classical networks.

The energy function for modern Hopfield networks is:
E = -lse(β, X^T ξ) + 0.5 ξ^T ξ + (1/β) log N + 0.5 M²

where:
- X is the matrix of stored patterns (d × N)
- ξ is the current state vector
- β is the inverse temperature controlling retrieval sharpness
- lse is the log-sum-exp function
- N is the number of patterns, M is the largest norm

The update rule for modern Hopfield networks is:
ξ_new = X · softmax(β · X^T · ξ)

This is equivalent to a single attention head in the transformer attention mechanism,
establishing a deep connection between associative memory and attention.

A key insight is that the Hopfield update rule converges in ONE STEP for well-separated
patterns, making retrieval extremely efficient. The inverse temperature β controls the
trade-off between retrieval specificity and generalization.

Applications of Modern Hopfield Networks include:
1. Drug discovery and molecular property prediction
2. Document retrieval and question answering
3. Time series classification
4. Immune repertoire classification
5. Replacing traditional nearest-neighbor search in recommendation systems

The connection to transformers is particularly important: the self-attention mechanism in
transformers can be understood as performing associative memory retrieval in a modern
Hopfield network. Keys and values in attention correspond to stored patterns and their
associated outputs.
"""

SAMPLE_DOC_RAG = """
Retrieval-Augmented Generation (RAG) Systems

Retrieval-Augmented Generation (RAG) is a technique that combines the power of
large language models with information retrieval to produce more accurate, up-to-date,
and verifiable responses. RAG was introduced by Lewis et al. in 2020.

Traditional RAG Pipeline:
1. Document Ingestion: Documents are split into chunks, typically 256-512 tokens.
2. Embedding: Each chunk is encoded into a dense vector using an embedding model.
3. Vector Store: Embeddings are stored in a vector database (e.g., FAISS, Pinecone, Weaviate).
4. Retrieval: For a given query, the most similar chunks are retrieved using approximate
   nearest neighbor search (ANN).
5. Generation: Retrieved chunks are provided as context to an LLM which generates an answer.

Limitations of Traditional RAG:
- Chunking artifacts: Splitting documents can lose context that spans chunk boundaries.
- Fixed retrieval: Standard cosine similarity doesn't model query-document interactions.
- Sensitivity to query phrasing: Small changes in query wording can dramatically change results.
- No noise handling: Typos and ambiguous queries are not explicitly handled.
- Memory inefficiency: All embeddings must be stored and searched.

Advanced RAG Techniques:
- HyDE (Hypothetical Document Embeddings): Generate a hypothetical answer and use it for retrieval.
- Multi-Query Retrieval: Generate multiple query variants and fuse results.
- Parent-Child Chunking: Retrieve small chunks but return their larger parent for context.
- Contextual Compression: Extract only the relevant portion of a retrieved chunk.
- Re-ranking: Use a cross-encoder to re-rank initially retrieved chunks.
- Reciprocal Rank Fusion: Combine rankings from multiple retrieval systems.

Associative Memory as an Alternative to Traditional RAG:
Modern Hopfield Networks offer a compelling alternative to traditional vector similarity search.
Rather than finding the globally nearest neighbor, Hopfield retrieval performs energy
minimization that naturally handles:
- Noisy queries: The energy landscape smoothly handles perturbations
- Multi-pattern retrieval: Attention weights distributed across relevant patterns
- Context-aware similarity: The iterative update incorporates pattern relationships

Evaluation Metrics for RAG Systems:
- Faithfulness: Is the answer supported by the retrieved context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are the retrieved chunks relevant?
- Context Recall: Are all relevant chunks retrieved?
- RAGAS (RAG Assessment) is a popular evaluation framework.

Vector Databases for RAG:
- FAISS: Facebook AI Similarity Search, highly optimized for GPU
- Pinecone: Managed vector database with filtering
- Weaviate: Open-source with GraphQL interface
- Chroma: Lightweight, easy to use for prototyping
- Milvus: Distributed vector database for production scale
"""
