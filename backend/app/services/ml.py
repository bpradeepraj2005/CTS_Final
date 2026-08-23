"""
Production serving layer for the prior-authorization models.

Model 1 is the guideline reasoning service on Render (see prior_auth_client).
Model 2 is the appeal-propensity regressor bundle in
ml/models/appeal_propensity.joblib, trained by ml/train_appeal.py, combined with
the rule-based gap split in curability.py. It runs only on the denial path.

ml/models/policy_fit.joblib is no longer loaded -- Model 1 replaced it. The file
can stay on disk for reference; nothing reads it.

Everything a scoring function needs now comes out of one Report. The pipeline
fetches it once per request and threads it through, so a single adjudication
makes a single call to the service.
"""
from ..config import (
    MODEL2_ENABLED,
    MODEL2_REAPPEAL_PERCENTILE,
    PRIOR_AUTH_URL,
)
from . import model2
from .prior_auth_client import (  # noqa: F401  (ModelUnavailable is re-exported)
    ModelUnavailable,
    analyse,
    build_case_text,
    health,
)

# Obligation weights, shared with model2._criteria_satisfaction so that the
# approval likelihood shown to a reviewer and the criteria satisfaction reported
# by Model 2 cannot drift apart.
_OBLIGATION_WEIGHTS = {
    "mandatory": 5.0,
    "indication": 4.0,
    "implied": 3.0,
    "desirable": 2.0,
    "supporting": 2.0,
}

_PASS = ("PASS", "PASSED", "MET")
_FAIL = ("FAIL", "FAILED", "NOT_MET")
_MISSING = ("MISSING", "INSUFFICIENT", "UNKNOWN", "NOT_EVALUABLE")

_SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1, "info": 0}
_ROUTE_CONFIDENCE = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _status(rule: dict) -> str:
    return str(rule.get("status", "")).upper()


def _weight(rule: dict) -> float:
    obligation = (rule.get("obligation") or "").lower()
    for key, value in _OBLIGATION_WEIGHTS.items():
        if key in obligation:
            return value
    return 1.0


# ---------------------------------------------------------------------------
# Model 1 -- approval likelihood
# ---------------------------------------------------------------------------


def approval_likelihood(report: dict) -> float:
    """Weighted share of guideline criteria the case satisfies, as 0..1.

    This is what replaces the policy-fit regressor. It feeds the FIT criterion in
    necessity_engine and the auto-approve / auto-deny thresholds in config, so it
    has to stay on the same 0..1 scale the regressor used.

    Only PASS and FAIL count toward the ratio -- a criterion the service could not
    evaluate is not evidence either way. Unmet mandatory criteria then cap the
    result, because a case that fails a hard requirement should not be rescued by
    passing a pile of soft ones.
    """
    rules = report.get("rules") or []
    evaluable = [r for r in rules if _status(r) in _PASS + _FAIL]
    if not evaluable:
        # Nothing scoreable. Sit at the midpoint so the router sends it to a human
        # rather than auto-deciding on no evidence.
        return 0.5

    total = sum(_weight(r) for r in evaluable)
    earned = sum(_weight(r) for r in evaluable if _status(r) in _PASS)
    score = earned / total if total else 0.5

    mandatory_unmet = sum(
        1
        for r in rules
        if "mandatory" in (r.get("obligation") or "").lower()
        and _status(r) in _FAIL + _MISSING
    )
    if mandatory_unmet >= 2:
        score = min(score, 0.20)
    elif mandatory_unmet == 1:
        score = min(score, 0.35)

    return round(max(0.0, min(1.0, score)), 4)


def complexity_score(report: dict) -> float:
    """How much work this case is for a reviewer, as 0..1.

    Not a model output -- a deterministic read of the Report. Breadth of the rule
    set, how much the service could not resolve, how severe the flags are, and how
    sure it was that it pulled the right guideline.
    """
    rules = report.get("rules") or []
    flagged = report.get("flagged") or []
    routing = report.get("routing") or {}
    conditions = report.get("conditions") or []

    breadth = min(len(rules) / 12.0, 1.0)

    unresolved = [r for r in rules if _status(r) in _MISSING]
    unresolved_ratio = len(unresolved) / len(rules) if rules else 0.0

    severities = [
        _SEVERITY_RANK.get(str(f.get("severity", "")).lower(), 0) for f in flagged
    ]
    severity = (max(severities) / 3.0) if severities else 0.0

    retrieval_doubt = 1.0 - _ROUTE_CONFIDENCE.get(
        str(routing.get("confidence", "")).lower(), 0.6
    )
    if not routing.get("matched", True):
        retrieval_doubt = 1.0

    ambiguity = 1.0 if len(conditions) > 1 else 0.0

    missing_mandatory = 1.0 if (report.get("missing_mandatory") or []) else 0.0

    score = (
        0.25 * breadth
        + 0.25 * unresolved_ratio
        + 0.20 * severity
        + 0.15 * retrieval_doubt
        + 0.10 * missing_mandatory
        + 0.05 * ambiguity
    )
    return round(min(1.0, score), 4)


def predict_policy_fit(features: dict, document_text: str | None = None) -> float:
    """Approval likelihood for one case. Fetches the Report if it has to.

    Kept for callers that only hold `features`. The pipeline uses
    `approval_likelihood(report)` directly so it can reuse one Report across
    scoring, explanation and complexity.
    """
    return approval_likelihood(analyse(features, document_text))


# ---------------------------------------------------------------------------
# Model 2 -- appeal / supporting-material assessment
# ---------------------------------------------------------------------------


def predict_appeal(
    report: dict, features: dict, context: dict | None = None
) -> dict:
    """Appeal outlook for one denied case, shaped for the reviewer UI."""
    return model2.assess(report, features, context)["appeal_prediction"]


def empty_appeal(reason: str = "Not assessed -- request was not denied") -> dict:
    return model2.empty_prediction(reason)


# ---------------------------------------------------------------------------
# Readiness and provenance
# ---------------------------------------------------------------------------


def models_ready() -> dict:
    """Shape preserved: main.py and the dashboard both read these two keys."""
    service = health()
    return {
        "policy_fit": bool(service.get("reachable")),
        "appeal_propensity": model2.ready(),
    }


def service_health() -> dict:
    return health()


def metrics_card() -> dict:
    """Provenance for the model card.

    Model 1 is retrieval plus reasoning over a cited corpus, so it has no R2 or
    MAE to report and none is invented for it -- what is reported is provenance:
    which guideline, which version, which corpus.

    Model 2 does have held-out metrics now, read from appeal_metrics.json and
    passed through verbatim. They are poor: ROC-AUC ~0.54 against 0.500 for
    random guessing. That is reported as-is rather than softened.
    """
    service = health()
    m2 = model2.info()

    corpus = service.get("corpus") or {}
    versions = service.get("versions") or {}
    llm = service.get("llm") or {}

    return {
        "available": True,
        "reports_accuracy": False,
        "model_1": {
            "name": "Guideline reasoning service",
            "kind": "retrieval + LLM reasoning over a versioned guideline corpus",
            "endpoint": f"{PRIOR_AUTH_URL}/analyze",
            "reachable": bool(service.get("reachable")),
            "guideline_version": versions.get("guideline"),
            "rule_table_version": versions.get("rule_table"),
            "prompt_version": versions.get("prompt"),
            "reasoning_model": llm.get("model"),
            "conditions_indexed": corpus.get("records"),
            "criteria_indexed": corpus.get("criteria"),
            "procedure_codes": corpus.get("procedure_codes"),
            "chunks": corpus.get("chunks"),
            "notes": [
                "Every verdict carries a page citation into the source guideline, "
                "so a reviewer can check the reasoning against the text.",
                "Approval likelihood is a weighted share of satisfied criteria, "
                "not a learned score. Unmet mandatory criteria cap it.",
                "Runs on a free instance that spins down when idle. A cold "
                "request can take 50-90 seconds.",
            ],
        },
        "model_2": {
            "name": "Supporting-material assessment",
            "kind": "HistGradientBoostingRegressor over the submitted case fields",
            "enabled": MODEL2_ENABLED,
            "runs_on": "denied requests only",
            "reappeal_percentile_threshold": MODEL2_REAPPEAL_PERCENTILE,
            **m2,
            "notes": [
                "Splits unmet criteria into gaps a provider can close with "
                "documentation and gaps no document will fix. That split is "
                "rule-based and does not depend on the model.",
                "Fixable gaps route to human review; only-hard gaps auto-deny.",
                "The regressor barely separates cases on held-out data. The "
                "corpus records case features and appeal outcomes but almost no "
                "relationship between them.",
            ],
        },
        "notes": [
            "The approve, deny and route-to-human decision is still made by the "
            "deterministic rules engine, not by a classifier. A decision that "
            "affects treatment has to be reconstructable criterion by criterion.",
            "Model 1 grounds that engine in a cited guideline. Model 2 only "
            "decides whether a denial is worth a human's time.",
        ],
    }
