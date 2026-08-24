"""Implements prior_auth_criterion_scoring.md: criteria in, verdicts out,
verdicts become two numbers that are never blended -- `score` (how well the
request meets criteria) and `completeness` (how much of the picture the
evaluator actually had). A score of 90 from 3 of 9 criteria is not the same
claim as 90 from 9 of 9, and this module never lets the two collapse into
one.

## Why CONTEXTUAL criteria don't call an LLM here

The spec's `evaluator: llm` criteria (e.g. "failed medical management")
need a judgment call no comparison operator can make, backed by a quoted
evidence excerpt and a confidence score. This build has no LLM API
available, so those criteria are evaluated the same way every other
criterion is: deterministically, off a fact the caller supplies directly.
Concretely, a contextual criterion's PASS confidence is read from an
optional `<criterion_id>_confidence` fact (defaulting to 1.0 when the
caller asserts the finding as fact rather than a hedged judgment) -- this
is exactly the `{verdict, confidence}` contract the LLM evaluator prompt in
section 10 would need to emit, so wiring in a real LLM call later means
adding a call that produces this same shape, not redesigning the model.
The four hard rules on the LLM evaluator (must quote evidence, confidence
<0.5 forces INSUFFICIENT, never invent a value, confidence scales credit)
still apply to a human filling in that fact by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:
    from .rule_engine import CriterionVerdict
except ImportError:
    from rule_engine import CriterionVerdict

DATA_DIR = Path(__file__).parent / "data"


def _load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((DATA_DIR / name).read_text(encoding="utf-8"))


DECISION_TABLE = _load_yaml("decision_matrix.yaml")

SCOREABLE_TYPES = ("mandatory", "supporting", "contextual")


def is_genuinely_failed(criterion_type: str, verdict: str) -> bool:
    """True if this verdict counts against the patient. `exclusion`
    criteria are authored with inverted polarity (see rules.yaml): verdict
    FAIL means the excluded scenario does NOT apply (good), and PASS means
    it DOES (bad) -- the opposite of every other type. Anything presenting
    verdicts to a human (rationale text, the plain-text summary, the
    reviewer UI) must go through this rather than testing `verdict ==
    "fail"` directly, or an exclusion that correctly did not fire will
    read as a failure."""
    if criterion_type == "exclusion":
        return verdict == "pass"
    return verdict == "fail"


def _credit(v: CriterionVerdict) -> float:
    if v.verdict != "pass":
        return 0.0
    if v.type == "contextual" and v.confidence is not None:
        return max(0.0, min(1.0, v.confidence))
    return 1.0


# ---------------------------------------------------------------- score + completeness

def compute_score(verdicts: list[CriterionVerdict]) -> dict[str, Any]:
    gateways = [v for v in verdicts if v.type == "gateway"]
    if any(v.verdict == "fail" for v in gateways):
        return _terminal(verdicts, score=0.0, reason="gateway_failed",
                          rationale="A gateway criterion failed: this request does not match the guideline this corpus covers.")
    if any(v.verdict == "insufficient" for v in gateways):
        return _terminal(verdicts, score=None, reason="gateway_insufficient",
                          rationale="A gateway criterion (diagnosis/procedure match) could not be evaluated.")

    exclusions = [v for v in verdicts if v.type == "exclusion"]
    if any(v.verdict == "pass" for v in exclusions):
        return _terminal(verdicts, score=0.0, reason="exclusion_matched",
                          rationale="The request matches a guideline-listed exclusion (non-indication).")
    exclusion_insufficient = any(v.verdict == "insufficient" for v in exclusions)

    scoreable = [v for v in verdicts if v.type in SCOREABLE_TYPES]
    evaluable = [v for v in scoreable if v.verdict in ("pass", "fail")]

    caps_applied: list[str] = []
    if not evaluable:
        score = None
    else:
        earned = sum(v.weight * _credit(v) for v in evaluable)
        available = sum(v.weight for v in evaluable)
        score = 100 * earned / available if available else None

    if score is not None:
        mandatory_fails = sum(1 for v in scoreable if v.type == "mandatory" and v.verdict == "fail")
        if mandatory_fails >= 2:
            score = min(score, 20.0)
            caps_applied.append("2_or_more_mandatory_fail_cap_20")
        elif mandatory_fails == 1:
            score = min(score, 35.0)
            caps_applied.append("mandatory_fail_cap_35")
        if exclusion_insufficient:
            score = min(score, 60.0)
            caps_applied.append("exclusion_insufficient_cap_60")
        score = round(score, 1)

    completeness = _completeness(verdicts)
    tally = {
        "pass": sum(1 for v in verdicts if v.verdict == "pass"),
        "fail": sum(1 for v in verdicts if v.verdict == "fail"),
        "insufficient": sum(1 for v in verdicts if v.verdict == "insufficient"),
        "not_applicable": sum(1 for v in verdicts if v.verdict == "not_applicable"),
    }
    return {"score": score, "completeness": completeness, "reason": None, "caps_applied": caps_applied,
            "tally": tally, "criteria_evaluated": len(verdicts)}


def _completeness(verdicts: list[CriterionVerdict]) -> float:
    """Gateway criteria are preconditions, not part of "the clinical
    picture" -- excluded from this denominator, matching the spec's own
    worked example (4 evaluated / 6 applicable, out of 7 total criteria
    including 1 gateway)."""
    population = [v for v in verdicts if v.type != "gateway" and v.verdict != "not_applicable"]
    if not population:
        return 100.0
    evaluated = sum(1 for v in population if v.verdict in ("pass", "fail"))
    return round(100 * evaluated / len(population), 1)


def _terminal(verdicts: list[CriterionVerdict], score: float | None, reason: str, rationale: str) -> dict[str, Any]:
    tally = {
        "pass": sum(1 for v in verdicts if v.verdict == "pass"),
        "fail": sum(1 for v in verdicts if v.verdict == "fail"),
        "insufficient": sum(1 for v in verdicts if v.verdict == "insufficient"),
        "not_applicable": sum(1 for v in verdicts if v.verdict == "not_applicable"),
    }
    return {"score": score, "completeness": _completeness(verdicts), "reason": reason, "caps_applied": [],
            "tally": tally, "criteria_evaluated": len(verdicts), "rationale": rationale}


def top_score_drivers(verdicts: list[CriterionVerdict], limit: int = 5) -> list[dict[str, Any]]:
    """Leave-one-out over the criteria that actually fed the score."""
    scoreable = [v for v in verdicts if v.type in SCOREABLE_TYPES]
    evaluable = [v for v in scoreable if v.verdict in ("pass", "fail")]
    if len(evaluable) < 2:
        return []
    baseline = compute_score(verdicts)["score"]
    if baseline is None:
        return []
    drivers = []
    for i, v in enumerate(evaluable):
        remainder_verdicts = [x for x in verdicts if x is not v]
        without = compute_score(remainder_verdicts)["score"]
        if without is None:
            continue
        delta = round(baseline - without, 1)
        if abs(delta) < 0.5:
            continue
        drivers.append({"factor": v.explanation, "criterion_id": v.criterion_id, "contribution": delta})
    drivers.sort(key=lambda d: -abs(d["contribution"]))
    return drivers[:limit]


# ---------------------------------------------------------------- Section 8: what to ask for

def classify_gaps(verdicts: list[CriterionVerdict]) -> dict[str, list[CriterionVerdict]]:
    blocking, material, minor = [], [], []
    for v in verdicts:
        if v.verdict != "insufficient":
            continue
        if v.type in ("mandatory", "gateway"):
            blocking.append(v)
        elif v.weight >= 3:
            material.append(v)
        else:
            minor.append(v)
    return {"blocking": blocking, "material": material, "minor": minor}


def requested_information(gaps: dict[str, list[CriterionVerdict]], cap: int = 3) -> list[dict[str, Any]]:
    ranked = gaps["blocking"] + gaps["material"]
    out = []
    for v in ranked[:cap]:
        # A single criterion can have more than one unresolved field (e.g.
        # a closed indication list where several items are simply unstated
        # rather than known-false) -- list all of them, not just the first,
        # or the RFI silently drops fields the request genuinely still needs.
        item = ", ".join(v.missing_fields) if v.missing_fields else v.criterion_id
        out.append({
            "item": item,
            "why": v.explanation,
            "page": v.citation.get("page"),
            "criticality": "blocking" if v in gaps["blocking"] else "material",
        })
    return out


# ---------------------------------------------------------------- Section 7: decision

def decide(score: float | None, completeness: float, reason: str | None, has_blocking_gap: bool) -> str:
    if reason in ("gateway_failed", "exclusion_matched"):
        return "deny"
    if score is None:
        return "request_more_information"
    # A score of exactly 0 that did NOT come from a gateway/exclusion
    # match (i.e. every evaluable criterion simply failed) is a low score,
    # not an exclusion -- falls through to the lowest band like any other
    # score, per "never deny on a low score."
    return _band_decision(score, completeness, has_blocking_gap)


def _band_decision(score: float, completeness: float, has_blocking_gap: bool) -> str:
    for row in DECISION_TABLE["score_bands"]:
        if row["min"] <= score <= row["max"]:
            if completeness >= row["completeness_threshold"] and not has_blocking_gap:
                return row["positive_decision"]
            return "request_more_information"
    return "request_more_information"


def enforce_invariants(decision: str, completeness: float, has_blocking_gap: bool, reason: str | None) -> str:
    """Belt-and-suspenders re-check of section 7's three invariants,
    independent of the table lookup above."""
    if decision == "approve" and has_blocking_gap:
        decision = "request_more_information"
    if decision == "approve" and completeness < 80:
        decision = "request_more_information"
    if decision == "deny" and reason not in ("gateway_failed", "exclusion_matched"):
        decision = "pend"
    return decision
