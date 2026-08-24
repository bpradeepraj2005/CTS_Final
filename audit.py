"""Phase J -- append-only audit log. Every decision this system makes is
written as one JSON line, immutable by convention (the file is append-only
and nothing in this codebase opens it for writing except `write_record`).

This buys the two properties the spec calls out: replay a historical
decision exactly by pinning rule_table_version + prompt_version (here,
"prompt_version" is the fact-schema/scoring-model version, since there is
no LLM prompt in this build), and show a disputing provider the exact page
and rule that drove their decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path(__file__).parent / "data" / "audit_log.jsonl"


def write_record(record: dict[str, Any], path: Path = DEFAULT_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path = DEFAULT_LOG) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
