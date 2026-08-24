"""The corpus as the LLM sees it: a routing index built from chunk context
headers, and complete per-condition dossiers.

No rule table is involved. `chunker.build_chunks` already stamps every chunk
with a context header of the form

    [Condition | Specialty | ICD-10 codes | chunk_type | ICMR STW 2019 Vol 1 p.N]

so routing means matching against those headers, and analysis means handing
over every chunk that shares a parent record. The largest dossier in the corpus
is ~1.8k tokens, so nothing is truncated or top-k sampled -- the model sees the
condition's complete guideline content: every procedure, drug, investigation,
referral rule, exclusion and workflow the STW records for it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prior_auth.chunker import Chunk, build_chunks

DEFAULT_SOURCE = Path(__file__).parent.parent / "icmr_stw_vol1_rules.json"

# Ordered so a dossier reads the way a clinician would work through it rather
# than in whatever order the source JSON happened to use.
_TYPE_ORDER = [
    "overview",
    "criteria",
    "score",
    "investigation",
    "procedure",
    "drug",
    "care_level",
    "referral",
    "admission",
    "discharge",
    "exclusion",
    "payer_note",
    "context",
]


@dataclass(frozen=True)
class Condition:
    """One STW record, with everything the model needs to route to it."""

    record_id: str
    condition: str
    specialty: str
    icd10: list[str]
    page: int
    chunk_types: list[str]

    def index_line(self) -> str:
        codes = ", ".join(self.icd10) or "none listed"
        return (
            f"{self.record_id} | {self.condition} | {self.specialty} "
            f"| ICD-10 {codes} | p.{self.page} | covers: {', '.join(self.chunk_types)}"
        )


class Catalog:
    """Loaded once per process; immutable afterwards."""

    def __init__(self, source: Path = DEFAULT_SOURCE):
        self.chunks: list[Chunk] = build_chunks(source)

        self._by_record: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunks:
            self._by_record[chunk.parent_record_id].append(chunk)

        self.conditions: dict[str, Condition] = {}
        for record_id, chunks in self._by_record.items():
            first = chunks[0]
            present = {c.chunk_type for c in chunks}
            self.conditions[record_id] = Condition(
                record_id=record_id,
                condition=first.condition,
                specialty=first.specialty,
                icd10=first.icd10,
                page=first.page,
                chunk_types=[t for t in _TYPE_ORDER if t in present],
            )

    # -- routing ----------------------------------------------------------
    def index_text(self) -> str:
        """Every condition as one line. ~5k characters for the whole corpus,
        so the router sees the complete catalogue rather than a shortlist
        someone else pre-filtered."""
        lines = sorted(
            (c.index_line() for c in self.conditions.values()),
            key=str.lower,
        )
        return "\n".join(lines)

    # -- analysis ---------------------------------------------------------
    def dossier(self, record_id: str) -> str:
        """Every chunk for one condition, headers intact, ordered clinically."""
        chunks = self._by_record.get(record_id)
        if not chunks:
            raise KeyError(record_id)
        ordered = sorted(
            chunks,
            key=lambda c: (
                _TYPE_ORDER.index(c.chunk_type) if c.chunk_type in _TYPE_ORDER else len(_TYPE_ORDER),
                c.chunk_id,
            ),
        )
        return "\n\n".join(c.text for c in ordered)

    def dossier_chunks(self, record_id: str) -> list[Chunk]:
        return list(self._by_record.get(record_id, []))

    def citations(self, record_ids: list[str]) -> list[dict[str, Any]]:
        """Provenance for whatever the model was shown -- so a caller can
        verify a claim without the reasoning itself being templated."""
        out: list[dict[str, Any]] = []
        for record_id in record_ids:
            for chunk in self.dossier_chunks(record_id):
                out.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "condition": chunk.condition,
                        "specialty": chunk.specialty,
                        "chunk_type": chunk.chunk_type,
                        "page": chunk.page,
                        "source": "ICMR STW 2019 Vol 1",
                    }
                )
        return out

    def resolve(self, record_ids: list[str]) -> list[str]:
        """Keep only ids that exist, preserving the model's ordering."""
        seen: set[str] = set()
        out: list[str] = []
        for rid in record_ids:
            if rid in self.conditions and rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out
