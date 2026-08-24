"""Runnable entry point: evaluate a prior-authorization request against the
STW criteria and print the pass/fail/score summary. One output format --
no separate JSON dump, no raw top-k retrieved-chunk listing.

Run with a JSON file:      python -m prior_auth.app request.json
Run interactively:         python -m prior_auth.app

The full decision packet (rule evaluations, citations, etc.) is still
written to the audit log by decision_engine.adjudicate for anyone who
needs it; this CLI only ever shows the human-readable summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .decision_engine import Corpus, adjudicate
    from .interactive import build_request_interactively
    from .summary import render_json_report
except ImportError:
    from decision_engine import Corpus, adjudicate
    from interactive import build_request_interactively
    from summary import render_json_report

DEFAULT_SOURCE = Path(__file__).parent.parent / "icmr_stw_vol1_rules.json"


def _load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("Request JSON must contain an object.")
    return value


def _attach_internal_procedure_code(request: dict[str, Any], procedure_codes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    service = request.get("requested_service") or {}
    if service.get("code"):
        return request
    service_text = str(service.get("text") or "").lower()
    for name, entry in procedure_codes.items():
        if name in service_text or service_text in name:
            service["code"] = entry["code"]
            break
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description="Prior Authorization Triage & Policy Companion")
    parser.add_argument("request", type=Path, nargs="?",
                         help="JSON prior-authorization request to evaluate. Omit to enter one interactively.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source file not found: {args.source}")

    corpus = Corpus(args.source)
    request = _load_request(args.request) if args.request else build_request_interactively(list(corpus.records_by_id.values()))
    request = _attach_internal_procedure_code(request, corpus.procedure_codes)
    packet = adjudicate(request, corpus)
    print(render_json_report(packet, request))


if __name__ == "__main__":
    raise SystemExit(main())
