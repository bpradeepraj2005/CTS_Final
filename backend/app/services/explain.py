"""
Local explanations for the approval-likelihood score.

The previous implementation attributed the policy-fit regressor by single-feature
ablation: re-score the case ~40 times, once per feature replaced by a corpus
reference value, and take the difference. That is not available against a remote
model -- forty ablations would be forty network calls at ~54 seconds each.

It is also no longer necessary. Approval likelihood is now a weighted share of
guideline criteria (see ml.approval_likelihood), so the attribution is exact
rather than approximated: each criterion's contribution is its own weight, signed
by whether the case met it. No interactions to decompose, no reference profile,
no caveat about first-order approximation.

The output keeps the shape the attribution rail reads -- base_score,
contributions[], method, caveat, levers -- so the reviewer UI is untouched.
"""
from .prior_auth_client import ModelUnavailable  # noqa: F401  (re-exported)

_PASS = ("PASS", "PASSED", "MET")
_FAIL = ("FAIL", "FAILED", "NOT_MET")
_MISSING = ("MISSING", "INSUFFICIENT", "UNKNOWN", "NOT_EVALUABLE")

# Unmet criteria a provider can usually act on, shown first in the panel.
# Mirrors the curability classes Model 2 uses, but we only need the binary here.
_ACTIONABLE_HINTS = (
    "not performed",
    "not done",
    "not submitted",
    "not attached",
    "not available",
    "not recorded",
    "not documented",
    "not stated",
    "missing",
    "investigation",
    "laborator",
    "x-ray",
    "ultrasound",
    "imaging",
    "serum",
    "electrolyte",
    "culture",
    "report",
    "consent",
    "form",
    "referral",
)

_OBLIGATION_WEIGHTS = {
    "mandatory": 5.0,
    "indication": 4.0,
    "implied": 3.0,
    "desirable": 2.0,
    "supporting": 2.0,
}


def _status(rule: dict) -> str:
    return str(rule.get("status", "")).upper()


def _weight(rule: dict) -> float:
    obligation = (rule.get("obligation") or "").lower()
    for key, value in _OBLIGATION_WEIGHTS.items():
        if key in obligation:
            return value
    return 1.0


def _label(rule: dict, index: int) -> str:
    """A readable name for the criterion. Guideline rule text is a sentence, so
    take the leading clause and fall back to a number."""
    text = (rule.get("rule") or "").strip()
    if not text:
        return f"Criterion {index}"
    for sep in (":", " -- ", ". "):
        if sep in text:
            head = text.split(sep, 1)[0].strip()
            if 8 <= len(head) <= 70:
                return head
    return text[:70].rstrip(" .,;") + ("..." if len(text) > 70 else "")


def _actionable(rule: dict) -> bool:
    blob = " ".join(
        str(rule.get(k) or "")
        for k in ("rule", "patient_evidence", "explanation")
    ).lower()
    return any(hint in blob for hint in _ACTIONABLE_HINTS)


def _observed(rule: dict) -> str:
    evidence = (rule.get("patient_evidence") or "").strip()
    if evidence:
        return evidence[:110] + ("..." if len(evidence) > 110 else "")
    status = _status(rule)
    if status in _MISSING:
        return "not evaluable from the submission"
    return "met" if status in _PASS else "not met"


def explain_policy_fit(report: dict, top_n: int = 10) -> dict:
    """Per-criterion contributions to the approval-likelihood score.

    A satisfied criterion contributes its normalised weight upward; an unmet one
    contributes it downward. The contributions sum to base_score minus the score
    a case would get if it met nothing, which is what the rail renders.
    """
    rules = report.get("rules") or []
    evaluable = [r for r in rules if _status(r) in _PASS + _FAIL]
    unresolved = [r for r in rules if _status(r) in _MISSING]

    if not evaluable and not unresolved:
        return {
            "base_score": 0.5,
            "reference_score": 0.5,
            "method": "no guideline criteria were evaluable for this case",
            "caveat": (
                "The service matched a guideline record but could not evaluate any "
                "criterion against the submission. Routed to a human."
            ),
            "contributions": [],
            "levers": [],
        }

    total = sum(_weight(r) for r in evaluable) or 1.0
    earned = sum(_weight(r) for r in evaluable if _status(r) in _PASS)
    uncapped = round(earned / total, 4)

    # The score in the rail header has to be the score the engine actually
    # decided on, cap included -- otherwise a reviewer sees 0.50 here and 0.20 on
    # the request. approval_likelihood owns that rule; do not restate it.
    from .ml import approval_likelihood

    base = approval_likelihood(report)
    capped = base < uncapped

    contributions = []
    for i, rule in enumerate(evaluable + unresolved, start=1):
        status = _status(rule)
        share = round(_weight(rule) / total, 5)
        met = status in _PASS
        # Unresolved criteria do not move the score, but a reviewer needs to see
        # them, so they are listed at zero.
        contribution = 0.0 if status in _MISSING else (share if met else -share)
        contributions.append(
            {
                "feature": f"GL-{i:02d}",
                "label": _label(rule, i),
                "value": _observed(rule),
                "reference": (rule.get("obligation") or "criterion").lower(),
                "contribution": round(contribution, 5),
                "direction": "supports" if contribution > 0 else "weakens",
                "actionable": (not met) and _actionable(rule),
                "page": rule.get("page"),
                "citation": rule.get("kb_section"),
                "why_it_matters": rule.get("why_it_matters"),
            }
        )

    contributions.sort(key=lambda c: -abs(c["contribution"]))
    negatives = [c for c in contributions if c["contribution"] < 0]

    caveat = (
        "Each criterion contributes its own obligation weight, so the "
        "contributions are exact rather than approximated."
    )
    if unresolved:
        caveat += (
            f" {len(unresolved)} criteria could not be evaluated from the "
            f"submission and contribute nothing."
        )
    if capped:
        caveat += (
            f" The weighted share of satisfied criteria is {uncapped:.2f}, but "
            f"unmet mandatory criteria cap the score at {base:.2f}, so the "
            f"contributions below sum to more than the score shown."
        )

    return {
        "base_score": base,
        "uncapped_score": uncapped,
        # What the case would score meeting nothing -- the rail's left anchor.
        "reference_score": 0.0,
        "method": (
            "exact per-criterion attribution against the matched guideline record"
        ),
        "caveat": caveat,
        "contributions": contributions[:top_n],
        "levers": [c for c in negatives if c["actionable"]][:4],
    }


def guideline_criteria(report: dict) -> list[dict]:
    """The Report's rules mapped into decision-ledger rows.

    Appended to the ledger after the necessity score is computed, so these rows
    are shown to a reviewer without changing what the engine decided. Weights are
    rescaled to the 0-0.25 range the ledger's meters expect.
    """
    rules = report.get("rules") or []
    if not rules:
        return []

    heaviest = max(_weight(r) for r in rules) or 1.0
    rows = []
    for i, rule in enumerate(rules, start=1):
        status = _status(rule)
        page = rule.get("page")
        expected = (rule.get("obligation") or "criterion").lower()
        rows.append(
            {
                "code": f"GL-{i:02d}",
                "label": _label(rule, i),
                "passed": status in _PASS,
                "observed": _observed(rule),
                "expected": f"{expected}{f' (p.{page})' if page else ''}",
                "weight": round(0.25 * _weight(rule) / heaviest, 3),
                "blocking": False,
                "guideline": True,
            }
        )
    return rows
