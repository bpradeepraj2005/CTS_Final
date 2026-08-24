"""
Load and validate the ICMR Standard Treatment Workflows rule corpus.

Fails loudly on any record missing provenance (id, condition, specialty,
source.page) -- a chunk without a page number cannot be cited and must not
enter the index.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("id", "condition", "specialty")

# Comparator phrases -> canonical operator, used to keep extraction prompts
# and rendered prose consistent about vocabulary.
COMPARATOR_CANON = {
    "greater than or equal to": "gte",
    "more than or equal to": "gte",
    "at least": "gte",
    "less than or equal to": "lte",
    "at most": "lte",
    "greater than": "gt",
    "more than": "gt",
    "over": "gt",
    "less than": "lt",
    "under": "lt",
    "equal to": "eq",
    "equals": "eq",
}

_COMPARATOR_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(COMPARATOR_CANON, key=len, reverse=True)),
    re.IGNORECASE,
)


def normalize_comparators(text: str) -> str:
    """Rewrite comparator phrases to a canonical word so prompts/renderers
    see one consistent vocabulary. Never used to alter citation_text -- raw
    wording is always preserved alongside this."""

    def _sub(m: re.Match) -> str:
        return COMPARATOR_CANON[m.group(0).lower()]

    return _COMPARATOR_RE.sub(_sub, text)


class CorpusValidationError(Exception):
    pass


def load_corpus(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise CorpusValidationError(f"{path}: no 'records' list found")

    for rec in records:
        missing = [f for f in REQUIRED_FIELDS if not rec.get(f)]
        page = rec.get("source", {}).get("page") if isinstance(rec.get("source"), dict) else None
        if missing or page is None:
            missing_desc = missing + (["source.page"] if page is None else [])
            raise CorpusValidationError(
                f"Record {rec.get('id', '<unknown>')} missing required provenance: {missing_desc}"
            )

    return data


def load_records(path: Path) -> list[dict[str, Any]]:
    return load_corpus(path)["records"]


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "icmr_stw_vol1_rules.json"
    corpus = load_corpus(src)
    records = corpus["records"]
    print(f"Loaded {len(records)} records from {src.name}")
    print(f"Corpus: {corpus.get('corpus')!r} edition={corpus.get('edition')!r} volume={corpus.get('volume')!r}")
    specialties = sorted(set(r["specialty"] for r in records))
    print(f"Specialties ({len(specialties)}): {', '.join(specialties)}")
