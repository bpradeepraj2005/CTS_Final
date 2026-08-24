"""Create citation-preserving semantic chunks from the structured corpus."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .corpus_loader import load_records
except ImportError:  # Supports direct execution: python prior_auth/chunker.py
    from corpus_loader import load_records


@dataclass
class Chunk:
    chunk_id: str
    parent_record_id: str
    chunk_type: str
    text: str
    text_for_display: str
    condition: str
    specialty: str
    page: int
    icd10: list[str]


SECTION_TYPES = {
    "clinical_criteria": "criteria",
    "severity_scores": "score",
    "investigations": "investigation",
    "care_level": "care_level",
    "referral_criteria": "referral",
    "admission_criteria": "admission",
    "discharge_criteria": "discharge",
    "procedures": "procedure",
    "drugs": "drug",
    "exclusions": "exclusion",
    "explicit_non_indications": "exclusion",
    "absolute_contraindications_thrombolysis": "exclusion",
    "contraindications": "exclusion",
    "donts": "exclusion",
    "prior_auth_notes": "payer_note",
}


def _render(value: Any, indent: int = 0) -> str:
    """Render nested JSON deterministically while keeping lists intact."""
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = key.replace("_", " ").capitalize()
            rendered = _render(item, indent + 1)
            if "\n" in rendered:
                lines.append(f"{prefix}{label}:\n{rendered}")
            else:
                lines.append(f"{prefix}{label}: {rendered}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            rendered = _render(item, indent + 1)
            if "\n" in rendered:
                lines.append(f"{prefix}-\n{rendered}")
            else:
                lines.append(f"{prefix}- {rendered}")
        return "\n".join(lines)
    if value is None:
        return "not specified"
    return str(value)


def build_chunks(source: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in load_records(source):
        record_id = record["id"]
        condition = record["condition"]
        specialty = record["specialty"]
        page = record["source"]["page"]
        icd10 = [str(code) for code in record.get("icd10", [])]
        codes = ", ".join(icd10) or "not specified"

        overview = record.get("definition")
        if overview:
            chunks.append(_make_chunk(record, "overview", 0, str(overview)))

        for field, chunk_type in SECTION_TYPES.items():
            value = record.get(field)
            if value in (None, [], {}):
                continue
            chunks.append(_make_chunk(record, chunk_type, 0, f"{field}:\n{_render(value)}"))

        # Preserve any structured sections not covered by the named semantic types.
        known = {"id", "condition", "specialty", "icd10", "source", "definition", *SECTION_TYPES}
        for field, value in record.items():
            if field in known or value in (None, [], {}):
                continue
            inferred_type = "exclusion" if "contraind" in field or "exclusion" in field else "context"
            chunks.append(_make_chunk(record, inferred_type, 0, f"{field}:\n{_render(value)}"))

    return chunks


def _make_chunk(record: dict[str, Any], chunk_type: str, index: int, prose: str) -> Chunk:
    condition = record["condition"]
    specialty = record["specialty"]
    page = record["source"]["page"]
    icd10 = [str(code) for code in record.get("icd10", [])]
    codes = ", ".join(icd10) or "not specified"
    header = f"[{condition} | {specialty} | ICD-10 {codes} | {chunk_type} | ICMR STW 2019 Vol 1 p.{page}]"
    return Chunk(
        chunk_id=f"{record['id']}::{chunk_type}::{index}",
        parent_record_id=record["id"],
        chunk_type=chunk_type,
        text=f"{header}\n{prose}",
        text_for_display=prose,
        condition=condition,
        specialty=specialty,
        page=page,
        icd10=icd10,
    )


def write_chunks(source: Path, destination: Path) -> list[Chunk]:
    chunks = build_chunks(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([asdict(chunk) for chunk in chunks], indent=2), encoding="utf-8")
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build semantic chunks from the ICMR JSON corpus.")
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent / "icmr_stw_vol1_rules.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "chunks.json")
    args = parser.parse_args()
    chunks = write_chunks(args.source, args.output)
    print(f"Wrote {len(chunks)} chunks to {args.output}")