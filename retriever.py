"""Phases G/H -- retrieval and reranking, scoped to what's honestly buildable
offline: no embedding model download or vector DB is available in this
environment, so the "hybrid dense+sparse" of Phase E collapses to real BM25
(the sparse half) rather than a fabricated dense score. This is disclosed,
not hidden -- see README for what a production deployment would add.

What IS implemented faithfully, because none of it needs a network call:
- BM25 scoring over the corpus's ~450 chunks (G1)
- mandatory rule-linked retrieval: every chunk backing a rule the engine
  fired is force-included, independent of its BM25 score (G3) -- this is
  the mechanism that keeps the explanation from drifting off the decision
- the exclusion sweep: every exclusion-type chunk for the matched
  condition, regardless of score (G4)
- diversity caps (H3) and decision-first context ordering (H4)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    from .chunker import Chunk
except ImportError:
    from chunker import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")
ORDER = ["overview", "criteria", "procedure", "exclusion", "investigation", "care_level"]


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Standard Okapi BM25 over the chunk corpus. No external deps."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_tokens = [_tokens(c.text) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str, indices: list[int] | None = None) -> list[tuple[int, float]]:
        q_terms = _tokens(query)
        candidates = indices if indices is not None else range(len(self.chunks))
        scores = []
        for i in candidates:
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                s += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    reason: str  # "bm25" | "rule_linked" | "exclusion_sweep" | "overview"


def find_chunk_for_citation(chunks: list[Chunk], condition: str, page: int, quote: str) -> Chunk | None:
    """Locate the chunk backing a fired rule, by condition + page + a
    substring match of the rule's citation_text against the chunk body."""
    candidates = [c for c in chunks if c.condition == condition and c.page == page]
    quote_words = set(_tokens(quote)[:12])
    best, best_overlap = None, 0
    for c in candidates:
        overlap = len(quote_words & set(_tokens(c.text_for_display)))
        if overlap > best_overlap:
            best, best_overlap = c, overlap
    return best


def retrieve(query: str, chunks: list[Chunk], bm25: BM25, rule_citations: list[dict[str, Any]],
             primary_condition: str | None = None, top_k: int = 20,
             max_per_type: int = 3, max_per_record: int = 8, final_k: int = 15) -> list[RetrievalResult]:
    by_id = {c.chunk_id: c for c in chunks}
    picked: dict[str, RetrievalResult] = {}

    # G3 -- mandatory rule-linked retrieval, guaranteed slots
    for citation in rule_citations:
        chunk = find_chunk_for_citation(chunks, citation["condition"], citation["page"], citation.get("quote", ""))
        if chunk:
            picked[chunk.chunk_id] = RetrievalResult(chunk, 1.0, "rule_linked")

    condition = primary_condition
    if condition is None and picked:
        condition = next(iter(picked.values())).chunk.condition

    # G1 -- BM25 over the full corpus
    ranked = bm25.score(query, indices=None)[:top_k]
    if condition is None and ranked:
        condition = chunks[ranked[0][0]].condition
    for idx, score in ranked:
        chunk = chunks[idx]
        if chunk.chunk_id not in picked:
            picked[chunk.chunk_id] = RetrievalResult(chunk, score, "bm25")

    # G4 -- exclusion sweep for the matched condition, regardless of score
    if condition:
        for chunk in chunks:
            if chunk.condition == condition and chunk.chunk_type == "exclusion" and chunk.chunk_id not in picked:
                picked[chunk.chunk_id] = RetrievalResult(chunk, 0.0, "exclusion_sweep")
        overview = next((c for c in chunks if c.condition == condition and c.chunk_type == "overview"), None)
        if overview and overview.chunk_id not in picked:
            picked[overview.chunk_id] = RetrievalResult(overview, 0.0, "overview")

    protected = {cid for cid, r in picked.items() if r.reason in ("rule_linked", "exclusion_sweep", "overview")}
    prunable = sorted((r for cid, r in picked.items() if cid not in protected), key=lambda r: -r.score)

    # H3 -- diversity caps (protected chunks are exempt)
    type_counts: Counter[str] = Counter(r.chunk.chunk_type for cid, r in picked.items() if cid in protected)
    record_counts: Counter[str] = Counter(r.chunk.parent_record_id for cid, r in picked.items() if cid in protected)
    kept = [picked[cid] for cid in protected]
    for r in prunable:
        if type_counts[r.chunk.chunk_type] >= max_per_type or record_counts[r.chunk.parent_record_id] >= max_per_record:
            continue
        kept.append(r)
        type_counts[r.chunk.chunk_type] += 1
        record_counts[r.chunk.parent_record_id] += 1
        if len(kept) >= final_k:
            break

    # H4 -- decision-first ordering
    def sort_key(r: RetrievalResult) -> tuple[int, float]:
        try:
            rank = ORDER.index(r.chunk.chunk_type)
        except ValueError:
            rank = len(ORDER)
        return (rank, -r.score)

    return sorted(kept, key=sort_key)
