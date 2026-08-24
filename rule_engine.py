"""The deterministic criterion engine (prior_auth_criterion_scoring.md
sections 1-3). Loads the criteria table from YAML (data/rules.yaml,
hand-curated; and data/rules_generated.yaml, auto-extracted -- see
rule_extractor.py) and evaluates it against a facts dict. This is the ONLY
component allowed to produce PASS / FAIL / INSUFFICIENT / NOT_APPLICABLE
verdicts. No LLM is in this path, so identical facts always produce
identical verdicts (score reproducibility, section 5 of the scoring doc).

## Criterion schema (each entry in the YAML `criteria:` list)

    criterion_id, source_record_id, source_page, condition
    applies_to: {procedure_codes: [...], icd10_any: [...]}
    type: gateway | exclusion | mandatory | supporting | contextual
    evaluator: deterministic | llm_offline_stub
        (no live LLM is available in this build -- see the module docstring
        in scoring_model.py for how `contextual` criteria are still
        evaluated without one)
    weight: int
    applicable_if: OPTIONAL predicate tree; False -> verdict is
        NOT_APPLICABLE, None -> verdict is INSUFFICIENT (can't tell if this
        criterion even applies), absent -> always applicable
    check: a predicate tree of {all: [...]} / {any: [...]} over leaf
        predicates {field, op, value}. op in: gt, gte, lt, lte, eq, ne, in,
        not_in, contains_any, not_contains_any, exists
    text: the criterion's prose, quoted from the source page -- doubles as
        both the citation and the plain-language explanation regardless of
        verdict
    version, review_status: unverified | human_verified

## Three-valued logic

A leaf predicate evaluates to True, False, or None (unknown) -- None
whenever `facts.get(field)` is missing. `all`/`any` combine with Kleene
logic: unknown only wins when it isn't already overridden by a decisive
False (for `all`) or True (for `any`). `check` result True/False/None maps
directly to verdict PASS/FAIL/INSUFFICIENT -- this is what correctly
implements "the data needed isn't in the request" without ever silently
treating it as a FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------- predicate evaluation

def _contains_any(haystack: Any, needles: list[str]) -> bool | None:
    if haystack is None:
        return None
    text = str(haystack).lower()
    return any(str(n).lower() in text for n in needles)


_OPS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "exists": lambda a, b: a is not None,
}


def _eval_leaf(pred: dict[str, Any], facts: dict[str, Any]) -> bool | None:
    field = pred["field"]
    op = pred["op"]
    value = pred.get("value")
    actual = facts.get(field)

    if op == "contains_any":
        return _contains_any(actual, value)
    if op == "not_contains_any":
        result = _contains_any(actual, value)
        return None if result is None else not result
    if op == "exists":
        return actual is not None
    if actual is None:
        return None
    try:
        return bool(_OPS[op](actual, value))
    except TypeError:
        return None


def _eval_node(node: dict[str, Any], facts: dict[str, Any]) -> bool | None:
    if "field" in node:
        return _eval_leaf(node, facts)
    if "all" in node:
        results = [_eval_node(child, facts) for child in node["all"]]
        if any(r is False for r in results):
            return False
        if any(r is None for r in results):
            return None
        return True
    if "any" in node:
        results = [_eval_node(child, facts) for child in node["any"]]
        if any(r is True for r in results):
            return True
        if any(r is None for r in results):
            return None
        return False
    raise ValueError(f"Malformed logic node: {node!r}")


def unresolved_fields(node: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    """Fields still capable of changing this node's outcome, for RFI
    reporting. Short-circuit-aware: if this node already resolved to
    True/False, nothing under it is "still needed", even if some leaf
    inside it happens to be null -- e.g. in `any: [ne(category, X),
    gte(size, N)]`, once the category clause alone proves the node True,
    `size` was never actually load-bearing for this request and must not
    be reported as a missing/blocking field."""
    if _eval_node(node, facts) is not None:
        return []
    if "field" in node:
        return [] if node["op"] == "exists" else [node["field"]]
    out: list[str] = []
    for key in ("all", "any"):
        if key in node:
            for child in node[key]:
                out.extend(unresolved_fields(child, facts))
    return out


# ---------------------------------------------------------------- criteria table

@dataclass
class Criterion:
    criterion_id: str
    source_record_id: str
    source_page: int
    condition: str
    applies_to: dict[str, Any]
    type: str  # gateway | exclusion | mandatory | supporting | contextual
    evaluator: str  # deterministic | llm_offline_stub
    weight: int
    check: dict[str, Any]
    text: str
    version: str
    review_status: str
    applicable_if: dict[str, Any] | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Criterion":
        return Criterion(
            criterion_id=d["criterion_id"], source_record_id=d["source_record_id"],
            source_page=d["source_page"], condition=d["condition"], applies_to=d.get("applies_to") or {},
            type=d["type"], evaluator=d.get("evaluator", "deterministic"), weight=int(d["weight"]),
            check=d["check"], text=d.get("text", ""), version=d.get("version", "1.0"),
            review_status=d.get("review_status", "unverified"), applicable_if=d.get("applicable_if"),
        )


def load_criteria(*paths: Path) -> list[Criterion]:
    criteria: list[Criterion] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("criteria") or []:
            criterion = Criterion.from_dict(entry)
            if criterion.criterion_id in seen_ids:
                raise ValueError(f"Duplicate criterion_id {criterion.criterion_id} in {path}")
            seen_ids.add(criterion.criterion_id)
            criteria.append(criterion)
    return criteria


def default_criteria_paths() -> list[Path]:
    return [DATA_DIR / "rules.yaml", DATA_DIR / "rules_generated.yaml"]


def criterion_applies(criterion: Criterion, facts: dict[str, Any]) -> bool:
    """OR the procedure-code filter with the ICD filter -- never AND."""
    applies_to = criterion.applies_to
    proc_codes = applies_to.get("procedure_codes") or []
    icd_any = applies_to.get("icd10_any") or []
    if not proc_codes and not icd_any:
        return False

    requested_code = ((facts.get("requested_service") or {}).get("code"))
    requested_icd = (facts.get("diagnosis") or {}).get("icd10") or []

    # A recognized requested procedure narrows procedure-specific criteria.
    # Falling back to diagnosis matching in that case incorrectly evaluates
    # rules for other procedures in the same condition, such as RFA or sinus
    # conversion criteria on a cardioversion request.
    if proc_codes and requested_code:
        return requested_code in proc_codes

    code_match = bool(proc_codes) and requested_code in proc_codes
    icd_match = bool(icd_any) and any(code in icd_any for code in requested_icd)
    return code_match or icd_match


# ---------------------------------------------------------------- evaluation

@dataclass
class CriterionVerdict:
    criterion_id: str
    condition: str
    type: str
    weight: int
    verdict: str  # "pass" | "fail" | "insufficient" | "not_applicable"
    confidence: float | None  # only meaningful for contextual PASS verdicts
    explanation: str
    missing_fields: list[str]
    citation: dict[str, Any]
    review_status: str


def evaluate_criterion(criterion: Criterion, facts: dict[str, Any]) -> CriterionVerdict:
    citation = {"page": criterion.source_page, "condition": criterion.condition, "quote": criterion.text}

    if criterion.applicable_if is not None:
        applicable = _eval_node(criterion.applicable_if, facts)
        if applicable is False:
            return CriterionVerdict(criterion.criterion_id, criterion.condition, criterion.type, criterion.weight,
                                     "not_applicable", None, criterion.text, [], citation, criterion.review_status)
        if applicable is None:
            missing = unresolved_fields(criterion.applicable_if, facts)
            return CriterionVerdict(criterion.criterion_id, criterion.condition, criterion.type, criterion.weight,
                                     "insufficient", None, criterion.text, missing, citation, criterion.review_status)

    result = _eval_node(criterion.check, facts)
    confidence = None
    if result is True:
        verdict = "pass"
        if criterion.type == "contextual":
            confidence = facts.get(f"{criterion.criterion_id}_confidence", 1.0)
    elif result is False:
        verdict = "fail"
    else:
        verdict = "insufficient"

    missing = unresolved_fields(criterion.check, facts) if verdict == "insufficient" else []
    return CriterionVerdict(criterion.criterion_id, criterion.condition, criterion.type, criterion.weight,
                             verdict, confidence, criterion.text, missing, citation, criterion.review_status)


def evaluate_applicable_criteria(criteria: list[Criterion], facts: dict[str, Any]) -> list[CriterionVerdict]:
    applicable = [c for c in criteria if criterion_applies(c, facts)]
    return [evaluate_criterion(c, facts) for c in applicable]
