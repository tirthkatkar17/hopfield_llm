"""
Query Handling Module
=====================
Robust query processing with:
  - Query cleaning & normalization
  - Query expansion for noisy/vague inputs
  - Multi-variant generation for fusion retrieval
  - Noise detection and handling
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ProcessedQuery:
    """Result of query processing pipeline."""
    original: str
    cleaned: str
    variants: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_noisy: bool = False
    noise_score: float = 0.0
    query_type: str = "factual"  # factual | conceptual | procedural | comparative
    metadata: Dict = field(default_factory=dict)


class QueryProcessor:
    """
    Multi-stage query processing pipeline designed to handle:
    - Typos and spelling errors
    - Vague or underspecified queries
    - Long verbose questions
    - Technical jargon
    - Noisy real-world inputs
    """

    # Common stopwords to strip from keyword extraction
    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "this", "that", "be",
        "are", "was", "were", "been", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "what", "when", "where", "who", "why", "how", "which", "about",
        "from", "into", "through", "during", "before", "after", "above",
        "below", "between", "each", "than", "its", "their", "our", "your",
        "my", "his", "her", "we", "they", "you", "i", "me", "him", "us",
        "just", "also", "some", "any", "all", "more", "much", "many",
        "most", "other", "then", "so", "such", "no", "only", "same",
        "both", "while", "use", "get", "make", "go", "see", "tell",
        "know", "think", "come", "want", "give", "take", "say", "look",
        "even", "still", "need", "feel", "try", "find", "like"
    }

    # Query type signals
    QUERY_TYPE_PATTERNS = {
        "factual": [r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhen\s+did\b",
                    r"\bwhere\s+is\b", r"\bdefine\b", r"\bmeaning\s+of\b"],
        "procedural": [r"\bhow\s+to\b", r"\bhow\s+do\b", r"\bsteps\s+to\b",
                       r"\bprocess\s+of\b", r"\bprocedure\b"],
        "comparative": [r"\bvs\b", r"\bversus\b", r"\bdifference\s+between\b",
                        r"\bcompare\b", r"\bbetter\b", r"\badvantage\b"],
        "conceptual": [r"\bexplain\b", r"\bdescribe\b", r"\bwhy\b",
                       r"\bconcept\b", r"\btheory\b", r"\bprinciple\b"],
    }

    # Common abbreviations to expand
    ABBREVIATIONS = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "dl": "deep learning",
        "nn": "neural network",
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "mhn": "modern hopfield network",
        "qa": "question answering",
        "ir": "information retrieval",
        "bert": "bidirectional encoder representations from transformers",
        "gpt": "generative pre-trained transformer",
        "api": "application programming interface",
    }

    def __init__(self, expand_abbreviations: bool = True):
        self.expand_abbreviations = expand_abbreviations

    def _detect_noise(self, text: str) -> Tuple[bool, float]:
        """
        Detect noise in query: typos, fragments, excessive brevity.
        Returns (is_noisy, noise_score in [0,1])
        """
        score = 0.0
        tokens = text.split()

        # Too short
        if len(tokens) < 2:
            score += 0.4
        elif len(tokens) < 4:
            score += 0.15

        # Very long (may be copy-pasted, noisy)
        if len(tokens) > 60:
            score += 0.2

        # All caps
        if text.isupper() and len(text) > 5:
            score += 0.2

        # Missing spaces between words (runon)
        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
        if avg_word_len > 15:
            score += 0.3

        # Lots of punctuation or special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
        if special_char_ratio > 0.2:
            score += 0.2

        # Repeated characters (typoooo)
        repeated = re.findall(r'(.)\1{3,}', text)
        if repeated:
            score += 0.3

        score = min(score, 1.0)
        return score > 0.3, score

    def _clean_text(self, text: str) -> str:
        """Clean and normalize query text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove leading/trailing punctuation (but keep question mark context)
        text = re.sub(r'^[^\w\s]+', '', text)
        text = re.sub(r'[^\w\s?!.]+$', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Remove repeated punctuation
        text = re.sub(r'[!?]{2,}', '?', text)
        text = re.sub(r'\.{3,}', '...', text)

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations in the query."""
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().rstrip('s')  # handle plural
            if lower in self.ABBREVIATIONS:
                expanded.append(self.ABBREVIATIONS[lower])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from query."""
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        keywords = [t for t in tokens if t not in self.STOPWORDS]

        # Score by length (longer = more specific)
        keywords = sorted(set(keywords), key=lambda x: -len(x))
        return keywords[:10]

    def _detect_query_type(self, text: str) -> str:
        """Classify query type based on linguistic patterns."""
        text_lower = text.lower()
        for qtype, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return qtype
        return "factual"

    def _generate_variants(
        self, query: str, keywords: List[str], query_type: str
    ) -> List[str]:
        """
        Generate query variants for robust multi-query retrieval.
        Handles vague/noisy queries by reformulating.
        """
        variants = [query]  # original always included

        # Keyword-only variant (robust to stopwords/noise)
        if keywords:
            kw_variant = ' '.join(keywords[:6])
            if kw_variant != query.lower() and len(kw_variant) > 5:
                variants.append(kw_variant)

        # Type-specific reformulations
        if query_type == "factual":
            # Add "explain" prefix for broader semantic coverage
            if not query.lower().startswith("explain"):
                variants.append(f"explain {query}")

        elif query_type == "procedural":
            # Add "steps" for more grounded retrieval
            clean = re.sub(r'^how\s+to\s+', '', query.lower())
            variants.append(f"steps for {clean}")
            variants.append(f"process of {clean}")

        elif query_type == "comparative":
            # Extract the compared entities
            parts = re.split(r'\s+vs\.?\s+|\s+versus\s+|\s+compared\s+to\s+',
                             query, flags=re.IGNORECASE)
            if len(parts) == 2:
                variants.append(f"advantages of {parts[0].strip()}")
                variants.append(f"advantages of {parts[1].strip()}")

        elif query_type == "conceptual":
            clean = re.sub(r'^(explain|describe|what\s+is)\s+', '', query.lower())
            variants.append(f"definition of {clean}")
            variants.append(f"{clean} overview")

        # Deduplication while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_norm = v.strip().lower()
            if v_norm not in seen and len(v_norm) > 3:
                seen.add(v_norm)
                unique_variants.append(v.strip())

        return unique_variants[:5]  # cap at 5 variants

    def process(self, raw_query: str) -> ProcessedQuery:
        """
        Full query processing pipeline.

        Steps:
        1. Clean text
        2. Detect noise
        3. Expand abbreviations
        4. Extract keywords
        5. Detect query type
        6. Generate variants

        Returns:
            ProcessedQuery with all processed fields populated.
        """
        cleaned = self._clean_text(raw_query)
        is_noisy, noise_score = self._detect_noise(cleaned)

        if self.expand_abbreviations:
            cleaned = self._expand_abbreviations(cleaned)

        keywords = self._extract_keywords(cleaned)
        query_type = self._detect_query_type(cleaned)
        variants = self._generate_variants(cleaned, keywords, query_type)

        return ProcessedQuery(
            original=raw_query,
            cleaned=cleaned,
            variants=variants,
            keywords=keywords,
            is_noisy=is_noisy,
            noise_score=noise_score,
            query_type=query_type,
            metadata={
                "n_variants": len(variants),
                "n_keywords": len(keywords),
            }
        )

    def batch_process(self, queries: List[str]) -> List[ProcessedQuery]:
        return [self.process(q) for q in queries]
