"""
Answer Generation Module
=========================
Synthesizes answers from retrieved chunks using:
  - Extractive highlighting + template-based generation  (AnswerGenerator)
  - LLM-powered abstractive synthesis                   (LLMAnswerGenerator)
  - Confidence scoring
  - Source attribution

LLM routing is fully delegated to core.llm_provider, which is:
  • Stateless  — no shared client objects between provider switches
  • Isolated   — only the chosen provider's code is ever imported/called
  • Validated  — key checks happen per-provider, not globally
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.hopfield import RetrievalResult
from core.query_handler import ProcessedQuery


@dataclass
class GeneratedAnswer:
    """Complete answer with supporting evidence and metadata."""
    query: str
    answer: str
    confidence: float           # [0, 1]
    evidence_chunks: List[RetrievalResult]
    highlighted_passages: List[Dict]
    sources: List[str]
    answer_type: str            # extractive | abstractive | hybrid | no_answer
    generation_stats: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Extractive / template-based generator (no LLM dependency)
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates answers from retrieved document chunks.
    Works fully offline — no external LLM required.
    """

    SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def __init__(
        self,
        max_answer_length: int = 600,
        min_confidence_threshold: float = 0.1,
        top_sentences: int = 5,
    ):
        self.max_answer_length = max_answer_length
        self.min_confidence_threshold = min_confidence_threshold
        self.top_sentences = top_sentences

    def _split_into_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in self.SENT_SPLIT.split(text) if len(s.strip()) > 20]

    def _score_sentence(self, sentence: str, keywords: List[str], query: str) -> float:
        sl = sentence.lower()
        ql = query.lower()
        score = 0.0
        kw_hits = sum(1 for kw in keywords if kw.lower() in sl)
        score += (kw_hits / max(len(keywords), 1)) * 0.5
        overlap = len(set(ql.split()) & set(sl.split())) / max(len(ql.split()), 1)
        score += overlap * 0.3
        if re.search(r'\d+', sentence):
            score += 0.1
        wc = len(sentence.split())
        score += 0.1 if 10 <= wc <= 50 else (-0.2 if wc < 5 else 0)
        return max(0.0, score)

    def _extract_best_sentences(self, chunks, keywords, query):
        scored = []
        for chunk in chunks:
            weight = chunk.similarity_score * (1.0 / chunk.rank)
            for sent in self._split_into_sentences(chunk.text):
                s = self._score_sentence(sent, keywords, query) * weight
                scored.append((sent, s, chunk.chunk_id, chunk.source))
        scored.sort(key=lambda x: -x[1])
        return scored

    def _compute_confidence(self, top_chunks, answer_sentences) -> float:
        if not top_chunks:
            return 0.0
        avg_sim = np.mean([c.similarity_score for c in top_chunks[:3]])
        source_bonus = min(len(set(c.source for c in top_chunks)) * 0.05, 0.15)
        sent_bonus = min(np.mean([s[1] for s in answer_sentences[:3]]), 0.2) if answer_sentences else 0.0
        return min(float(avg_sim * 0.65 + source_bonus + sent_bonus), 1.0)

    def _deduplicate_sentences(self, sentences):
        seen, unique = set(), []
        for item in sentences:
            key = re.sub(r'[^a-z0-9\s]', '', item[0].lower())[:60].strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _rephrase(self, sent: str) -> str:
        sent = re.sub(r'\s*[\(\[]\s*\d{1,4}\s*[\)\]]', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        if sent and sent[-1] not in '.!?':
            sent += '.'
        return (sent[0].upper() + sent[1:]) if sent else sent

    def _build_summary_paragraph(self, selected, query_type):
        ordered = sorted(selected[:3], key=lambda x: x[2])
        parts = [self._rephrase(s[0]) for s in ordered]
        if query_type == "procedural":
            return f"{parts[0]} {parts[1]}" if len(parts) >= 2 else (parts[0] if parts else "")
        if query_type == "comparative" and len(parts) >= 2:
            return f"{parts[0]} In contrast, {parts[1][0].lower() + parts[1][1:]}"
        return " ".join(parts[:2])

    def _build_key_points(self, selected, query_type, keywords):
        points = []
        for sent, _, _, _ in selected[1:8]:
            r = self._rephrase(sent)
            if len(r.split()) < 6:
                continue
            if not any(len(set(r.lower().split()) & set(e.lower().split())) / max(len(r.split()), 1) > 0.6 for e in points):
                points.append(r)
            if len(points) >= 5:
                break
        return points[:5] if len(points) >= 3 else points

    def _build_example(self, selected, query_type):
        pat = re.compile(r'\bfor example\b|\bfor instance\b|\bsuch as\b|\be\.g\.'
                         r'|\bi\.e\.|=|→|:\s|\bstep\s+\d|\d+\.\s|\bformula\b|\bequation\b',
                         re.IGNORECASE)
        for sent, _, _, _ in selected:
            if pat.search(sent) and len(sent.split()) > 8:
                return self._rephrase(sent)
        return None

    def _build_answer_text(self, sentences, query_type, keywords=None) -> str:
        keywords = keywords or []
        if not sentences:
            return "No relevant information was found for this question."
        sentences = self._deduplicate_sentences(sentences)
        seen, selected = set(), []
        for item in sentences[:self.top_sentences * 3]:
            key = item[0][:50].lower()
            if key not in seen:
                seen.add(key)
                selected.append(item)
            if len(selected) >= self.top_sentences * 2:
                break
        if not selected:
            return "No relevant information was found for this question."
        lines = []
        summary = self._build_summary_paragraph(selected, query_type)
        if summary:
            lines.append(summary)
        kps = self._build_key_points(selected, query_type, keywords)
        if kps:
            lines.append("\n**Key Points:**")
            lines.extend(f"- {p}" for p in kps)
        example = self._build_example(selected, query_type)
        if example and query_type in ("factual", "conceptual", "procedural"):
            lines.append(f"\n**Example / Note:** {example}")
        answer = "\n".join(lines).strip()
        if len(answer) > self.max_answer_length + 400:
            answer = answer[:self.max_answer_length + 400].rsplit('\n', 1)[0] + "\n- *(truncated)*"
        return answer

    def _build_highlighted_passages(self, top_chunks, keywords):
        passages = []
        for chunk in top_chunks[:4]:
            hits = []
            for kw in keywords[:8]:
                hits.extend(m.start() for m in re.finditer(re.escape(kw), chunk.text, re.IGNORECASE))
            display = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
            passages.append({
                "text": display, "full_text": chunk.text,
                "chunk_id": chunk.chunk_id, "source": chunk.source,
                "similarity": chunk.similarity_score, "attention": chunk.hopfield_attention,
                "rank": chunk.rank, "keyword_hits": len(hits),
                "keywords_found": [kw for kw in keywords if kw.lower() in chunk.text.lower()],
            })
        return passages

    def generate(
        self,
        processed_query: ProcessedQuery,
        retrieved_chunks: List[RetrievalResult],
    ) -> GeneratedAnswer:
        query      = processed_query.cleaned
        keywords   = processed_query.keywords
        query_type = processed_query.query_type

        if not retrieved_chunks:
            return GeneratedAnswer(
                query=query,
                answer="No relevant documents found. Please ensure documents are loaded and try rephrasing your query.",
                confidence=0.0, evidence_chunks=[], highlighted_passages=[],
                sources=[], answer_type="no_answer",
                generation_stats={"n_chunks": 0, "n_sentences": 0},
            )

        filtered = [c for c in retrieved_chunks if c.similarity_score >= self.min_confidence_threshold] \
                   or retrieved_chunks[:2]

        scored    = self._extract_best_sentences(filtered, keywords, query)
        answer    = self._build_answer_text(scored, query_type, keywords)
        conf      = self._compute_confidence(filtered, scored)
        highlights= self._build_highlighted_passages(filtered, keywords)
        sources   = list(dict.fromkeys(c.source for c in filtered))
        atype     = "extractive" if conf > 0.6 else "hybrid" if conf > 0.3 else "low_confidence"

        return GeneratedAnswer(
            query=query, answer=answer, confidence=conf,
            evidence_chunks=filtered, highlighted_passages=highlights,
            sources=sources, answer_type=atype,
            generation_stats={
                "n_chunks": len(filtered),
                "n_sentences_considered": len(scored),
                "n_sentences_used": min(self.top_sentences, len(scored)),
                "query_type": query_type, "n_keywords": len(keywords),
                "llm_used": False,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM-powered answer generator
# ─────────────────────────────────────────────────────────────────────────────

class LLMAnswerGenerator(AnswerGenerator):
    """
    Wraps AnswerGenerator with LLM-powered abstractive synthesis.

    Provider routing is fully delegated to core.llm_provider:
      - Each provider (openai / anthropic / local) is independent
      - Only the *currently configured* provider's key is ever checked
      - Switching providers never touches another provider's state
      - Falls back to extractive output when LLM call fails
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        model_name: Optional[str] = None,
        api_key: str = "",
        max_context_tokens: int = 3000,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Import the registry here so nothing else in this file depends on it
        from core.llm_provider import DEFAULT_MODELS
        self.llm_provider       = llm_provider
        self.model_name         = model_name or DEFAULT_MODELS.get(llm_provider, "gpt-4o-mini")
        self.api_key            = api_key          # UI-supplied key (may be blank → env fallback)
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_context(self, chunks: List[RetrievalResult]) -> str:
        parts, total = [], 0
        limit = self.max_context_tokens * 4
        for i, c in enumerate(chunks, 1):
            txt = f"[Passage {i} from '{c.source}']\n{c.text}"
            if total + len(txt) > limit:
                break
            parts.append(txt)
            total += len(txt)
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a precise, helpful assistant that answers questions strictly "
            "based on the provided document passages. "
            "Always ground your answer in the passages. "
            "If the passages do not contain sufficient information, say so clearly. "
            "Be concise and structured. Use bullet points for key facts when helpful."
        )

    @staticmethod
    def _build_user_prompt(query: str, context: str, query_type: str) -> str:
        instruction = {
            "factual":    "Provide a concise, accurate factual answer.",
            "procedural": "Provide clear step-by-step instructions.",
            "comparative":"Compare and contrast the relevant aspects clearly.",
            "conceptual": "Explain the concept clearly with relevant details.",
        }.get(query_type, "Answer the question based on the provided passages.")
        return (
            f"DOCUMENT PASSAGES:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"INSTRUCTION: {instruction} "
            "Use only information from the passages above. "
            "If the passages lack the answer, say: "
            "'The provided documents do not contain enough information to answer this question.'\n\n"
            "ANSWER:"
        )

    # ── Main generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        processed_query: ProcessedQuery,
        retrieved_chunks: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Run extractive generation first, then attempt LLM synthesis.
        Falls back to extractive on any LLM error.
        """
        from core.llm_provider import call_llm, validate_provider_config

        base = super().generate(processed_query, retrieved_chunks)
        if not retrieved_chunks:
            return base

        # ── Validate provider config BEFORE calling (provider-specific check) ─
        ok, err = validate_provider_config(self.llm_provider, self.api_key, self.model_name)
        if not ok:
            base.generation_stats["llm_used"]  = False
            base.generation_stats["llm_error"] = err
            return base

        # ── Build prompts ─────────────────────────────────────────────────────
        context       = self._build_context(retrieved_chunks[:6])
        system_prompt = self._build_system_prompt()
        user_prompt   = self._build_user_prompt(
            processed_query.cleaned, context, processed_query.query_type
        )

        # ── Call LLM — only the selected provider is touched ──────────────────
        try:
            llm_answer = call_llm(
                provider=self.llm_provider,
                model=self.model_name,
                api_key=self.api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            return GeneratedAnswer(
                query=base.query,
                answer=llm_answer,
                confidence=min(base.confidence + 0.15, 1.0),
                evidence_chunks=base.evidence_chunks,
                highlighted_passages=base.highlighted_passages,
                sources=base.sources,
                answer_type="abstractive",
                generation_stats={
                    **base.generation_stats,
                    "llm_used":     True,
                    "llm_provider": self.llm_provider,
                    "llm_model":    self.model_name,
                },
            )
        except Exception as exc:
            base.generation_stats["llm_used"]  = False
            base.generation_stats["llm_error"] = str(exc)
            return base
