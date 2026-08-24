"""Phase K -- decision-accuracy and reproducibility evaluation against the
golden set. Not retrieval-recall metrics (this build has no embedding
index to benchmark -- see README) but every decision-layer metric the spec
asks for that this offline build can actually measure: exact-match
decision accuracy, false-approval rate (weighted heaviest), false-denial
rate, mean RFI item count, and score reproducibility (must be 100% --
identical input must always produce an identical score, since the rule
engine does the arithmetic, not an LLM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .decision_engine import Corpus, adjudicate
except ImportError:
    from decision_engine import Corpus, adjudicate

DEFAULT_GOLDEN_SET = Path(__file__).parent / "data" / "golden_set.jsonl"


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(path: Path = DEFAULT_GOLDEN_SET) -> dict[str, Any]:
    corpus = Corpus()
    cases = load_golden_set(path)

    results = []
    reproducibility_failures = []
    for case in cases:
        packet1 = adjudicate(case["request"], corpus)
        packet2 = adjudicate(case["request"], corpus)
        if (packet1.get("score"), packet1.get("decision")) != (packet2.get("score"), packet2.get("decision")):
            reproducibility_failures.append(case["id"])
        results.append({
            "id": case["id"], "description": case["description"],
            "expected": case["expected_decision"], "actual": packet1["decision"],
            "match": packet1["decision"] == case["expected_decision"],
            "score": packet1.get("score"), "completeness": packet1.get("completeness"),
            "rfi_items": len(packet1.get("requested_information", [])),
        })

    n = len(results)
    correct = sum(1 for r in results if r["match"])

    # False approval: system approved something the golden label says should
    # have been denied or pended -- the costliest error, weighted heaviest.
    false_approvals = [r for r in results if r["actual"] == "approve" and r["expected"] in ("deny", "pend", "not_covered")]
    # False denial: system denied something the golden label says should not
    # have been denied.
    false_denials = [r for r in results if r["actual"] == "deny" and r["expected"] != "deny"]

    rfi_cases = [r for r in results if r["actual"] == "request_more_information"]
    mean_rfi_items = round(sum(r["rfi_items"] for r in rfi_cases) / len(rfi_cases), 2) if rfi_cases else 0.0

    summary = {
        "cases": n,
        "decision_accuracy": round(correct / n, 3) if n else 0.0,
        "false_approval_rate": round(len(false_approvals) / n, 3) if n else 0.0,
        "false_denial_rate": round(len(false_denials) / n, 3) if n else 0.0,
        "mean_rfi_items": mean_rfi_items,
        "score_reproducibility": round(1 - len(reproducibility_failures) / n, 3) if n else 1.0,
        "reproducibility_failures": reproducibility_failures,
        "mismatches": [r for r in results if not r["match"]],
    }
    return {"summary": summary, "results": results}


def main() -> None:
    report = run()
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    if summary["mismatches"]:
        print(f"\n{len(summary['mismatches'])} mismatch(es):")
        for m in summary["mismatches"]:
            print(f"  {m['id']}: expected {m['expected']!r}, got {m['actual']!r} -- {m['description']}")


if __name__ == "__main__":
    main()
