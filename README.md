# 🧠 Associative Memory Retriever for Long-Context QA
### Using Modern Hopfield Networks

> A production-ready QA system that replaces traditional RAG with energy-minimization-based
> associative memory retrieval — robust to noisy queries and long documents.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DOCUMENT INGESTION                                             │
│  Text → DocumentChunker → EmbeddingEngine → HopfieldMemory     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  QUERY PROCESSING                                               │
│  RawQuery → Clean → NoiseDetect → Expand → Variants           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  HOPFIELD RETRIEVAL                                             │
│  QueryEmbedding → EnergyMinimization → SoftmaxAttention → Top-K│
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  ANSWER GENERATION                                              │
│  Chunks → SentenceScoring → Synthesis → ConfidenceScore        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 The Math: Modern Hopfield Networks

### Energy Function
```
E = −lse(β, X^T ξ) + ½ξ^Tξ + (1/β)log N + ½M²
```

### Update Rule (1-Step Convergence)
```
ξ_new = X · softmax(β · X^T · ξ)
```

### Capacity
- **Classical Hopfield:** 0.14 × N patterns
- **Modern Hopfield:** exp(d/2) patterns — exponential in embedding dimension!

| Property | Classical | Modern |
|---|---|---|
| State Space | Binary {-1,+1} | Continuous ℝ^d |
| Capacity | 0.14·N | exp(d/2) |
| Convergence | Many steps | **1 step** |
| Link to Attention | ❌ | ✅ Exact equivalence |

## 📁 Project Structure

```
hopfield_qa/
├── app.py                    # Streamlit UI
├── requirements.txt
├── core/
│   ├── embeddings.py         # DocumentChunker + EmbeddingEngine
│   ├── hopfield.py           # ModernHopfieldNetwork
│   ├── query_handler.py      # QueryProcessor (noise, expansion, variants)
│   ├── answer_generator.py   # AnswerGenerator (extractive + LLM)
│   └── pipeline.py           # HopfieldQAPipeline (orchestrator)
└── utils/
    ├── visualization.py      # Plotly charts
    └── file_utils.py         # Document loading (txt, pdf, docx, md)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run app.py
```

### 3. Programmatic Usage
```python
from core.pipeline import HopfieldQAPipeline, PipelineConfig

# Configure
config = PipelineConfig(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    hopfield_beta=8.0,    # Retrieval sharpness
    top_k=5,
    use_fusion=True,      # Multi-query fusion
)

# Initialize
pipeline = HopfieldQAPipeline(config)

# Load documents
pipeline.add_document("Your document text here...", source="my_doc")

# Query
result = pipeline.query("What is the storage capacity of Hopfield networks?")
print(result.answer.answer)
print(f"Confidence: {result.answer.confidence:.2%}")
print(f"Time: {result.total_time_ms:.1f}ms")
```

## ⚙️ Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `hopfield_beta` | 8.0 | Higher = sharper retrieval focus |
| `chunk_size` | 512 | Token-length of each memory pattern |
| `chunk_overlap` | 64 | Context preservation between chunks |
| `top_k` | 5 | Number of patterns to retrieve |
| `use_fusion` | True | Multi-query RRF fusion for noise robustness |

## 🧩 Key Features

- **Modern Hopfield Retrieval** — Energy minimization instead of cosine NN search
- **Multi-Query Fusion** — Generate query variants + Reciprocal Rank Fusion
- **Noise Handling** — Automatic detection and correction of noisy queries
- **Query Expansion** — Abbreviation expansion, type-specific reformulation
- **4 Query Types** — Factual, Procedural, Comparative, Conceptual
- **Confidence Scoring** — Multi-factor confidence estimation
- **Rich Visualizations** — Attention heatmaps, energy landscape, PCA scatter
- **Multi-format Support** — .txt, .pdf, .docx, .md
- **Fully offline** — No external API calls required

## 📖 References

1. Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021.
2. Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS.
3. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
4. Vaswani et al. (2017). "Attention is All You Need." NeurIPS 2017.
