"""
Model 2 -- supporting-material assessment on the denial path.

Replaces the local appeal-propensity classifier (ml/models/appeal_propensity.joblib)
with PriorAuthTriage from prior_auth_model.py.

Position in the flow: the Decision Router sends approvals straight to Auto
Approval. Only denials reach Model 2, which asks one question -- can the provider
fix this with more documentation?

    fixable gaps present  -> Human Review        (material can be supplied)
    only hard gaps        -> Auto Denial         (no supporting material path)
    high reappeal risk    -> Human Review        (override; likely overturned)

We deliberately do NOT call PriorAuthTriage.predict(). Its first branch is
`engine confidence > 80 -> APPROVE`, which reads the reasoning engine's certainty
as approvability. On a confidently denied case (confidence 90-96, criteria
satisfaction 20/100) it returns APPROVE -- verified against both the live
production Report and the model author's own example. The model's own warning
field says so: "the engine is confident the case FAILS". APPROVE is not a legal
output once the router has already said denial.

Instead we call the components underneath predict(), which are sound:
reappeal_probability(), percentile(), gaps(), criteria_score() and attribution().
That also fixes a second problem -- predict() only computes a reappeal
probability when confidence < 40, so on most denials it returned None.
"""
import importlib.util
import sys
import threading
import warnings

from ..config import (
    MODEL2_ENABLED,
    MODEL2_PATH,
    MODEL2_REAPPEAL_PERCENTILE,
    MODEL2_REPORTED_BASELINE,
    MODEL2_REPORTED_MACRO_AUC,
    MODEL2_REPORTED_SCORE,
)
from .prior_auth_client import ModelUnavailable

_model = None
_module = None
_load_lock = threading.Lock()
_load_error: str | None = None

APPEAL_LABELS = {
    "NEVER_APPLIED": "Unlikely to appeal",
    "REAPPLIED": "Likely to resubmit",
    "FORMAL_APPEAL": "Likely formal appeal",
    "APPEAL_WITH_NEW_DOCUMENTATION": "Likely appeal with new evidence",
}


def _load():
    """Import prior_auth_model.py from MODEL2_PATH and hand back its `model`.

    Loaded by path rather than by name because the file ships beside the trained
    artifact and is not on sys.path. The module unpickles a scikit-learn
    estimator at import time, so this is done once per process, lazily.
    """
    global _model, _module, _load_error
    if _model is not None:
        return _model, _module
    with _load_lock:
        if _model is not None:
            return _model, _module
        if not MODEL2_PATH.exists():
            _load_error = (
                f"Model 2 file not found at {MODEL2_PATH}. Copy prior_auth_model.py "
                f"there, or set MODEL2_PATH."
            )
            raise ModelUnavailable(_load_error)
        try:
            spec = importlib.util.spec_from_file_location(
                "prior_auth_model", MODEL2_PATH
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules["prior_auth_model"] = module
            with warnings.catch_warnings():
                # The estimator was pickled under scikit-learn 1.8.0. Pin
                # scikit-learn==1.8.0 in requirements.txt; this only silences the
                # per-import noise, it does not make a mismatch safe.
                warnings.simplefilter("ignore")
                spec.loader.exec_module(module)
            _model, _module = module.model, module
            _load_error = None
        except ModelUnavailable:
            raise
        except Exception as exc:
            _load_error = f"Could not load Model 2 from {MODEL2_PATH}: {exc}"
            raise ModelUnavailable(_load_error) from exc
    return _model, _module


def ready() -> bool:
    if not MODEL2_ENABLED:
        return False
    try:
        _load()
        return True
    except ModelUnavailable:
        return False


def info() -> dict:
    """Provenance for the model card. Never raises."""
    if not MODEL2_ENABLED:
        return {"available": False, "reason": "Model 2 disabled by configuration."}
    try:
        model, _ = _load()
    except ModelUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "version": getattr(model, "VERSION", "unknown"),
        "trained_on": getattr(model, "trained_on", "unknown"),
        "base_rate": round(float(getattr(model, "base_rate", 0.0)), 4),
        # oof is a numpy array; `or []` would hit its ambiguous truth value.
        "training_rows": int(len(getattr(model, "oof", []))),
        "reappeal_percentile_threshold": MODEL2_REAPPEAL_PERCENTILE,
        "caveat": (
            "Trained on synthetic labels. The pipeline is production-shaped, but "
            "these probabilities are not yet evidence about real appeal behaviour. "
            "Retrain on observed outcomes before relying on them."
        ),
    }


# ---------------------------------------------------------------------------
# Appeal distribution
# ---------------------------------------------------------------------------


def _distribution(p: float, fixable: int, hard: int) -> list[dict]:
    """Split the reappeal probability across the four outcome classes.

    PriorAuthTriage predicts one number -- P(filed) x P(won) -- not a four-way
    distribution, but the reviewer UI renders four bars. We apportion that number
    using the gap mix, which is the only real signal available: gaps a provider
    can close point at a resubmission with new evidence, gaps they cannot point
    at a formal appeal.
    """
    total_gaps = fixable + hard
    fs = (fixable / total_gaps) if total_gaps else 0.0

    weights = {
        "APPEAL_WITH_NEW_DOCUMENTATION": 0.25 + 0.50 * fs,
        "REAPPLIED": 0.20 + 0.25 * fs,
        "FORMAL_APPEAL": 0.55 - 0.35 * fs,
    }
    total_w = sum(weights.values())

    dist = {k: p * (w / total_w) for k, w in weights.items()}
    dist["NEVER_APPLIED"] = max(0.0, 1.0 - p)

    return [
        {
            "outcome": name,
            "label": APPEAL_LABELS[name],
            "probability": round(prob, 4),
        }
        for name, prob in sorted(dist.items(), key=lambda kv: -kv[1])
    ]


def empty_prediction(reason: str) -> dict:
    """Placeholder with the exact key set the reviewer UI reads.

    Approved requests never reach Model 2, so their appeal card has nothing to
    show. Returning nulls in the right shape degrades cleanly; omitting keys
    would throw in the frontend, which we are not allowed to touch.
    """
    return {
        "top_class": "NOT_CALCULATED",
        "top_label": reason,
        # None, not 0. These surface as an "Appeal risk" column in the request and
        # review lists, where 0 would read as a confident prediction of no appeal
        # rather than a question that was never asked. pct(null) renders "--",
        # and the denial-rate aggregate in dashboard.py already filters None.
        "top_probability": None,
        "any_appeal_probability": None,
        "distribution": [],
        "model_macro_auc": None,
        "model_accuracy": None,
        "baseline_accuracy": None,
        "assessed": False,
    }


# ---------------------------------------------------------------------------
# The assessment
# ---------------------------------------------------------------------------


def assess(report: dict, context: dict | None = None) -> dict:
    """Run the supporting-material assessment on one denied case.

    Returns the routing decision plus an `appeal_prediction` block shaped for the
    reviewer UI.
    """
    model, _ = _load()

    try:
        p = float(model.reappeal_probability(report, context))
        percentile = float(model.percentile(p))
        curable, hard = model.gaps(report)
        criteria = model.criteria_score(report)
        raises, lowers = model.attribution(report, context)
    except ModelUnavailable:
        raise
    except Exception as exc:
        raise ModelUnavailable(f"Model 2 assessment failed: {exc}") from exc

    fixable_rules = [g["rule"] for g in curable]
    hard_rules = [g["rule"] for g in hard]

    # Why the hard gaps are hard. "Hard" is not one thing: a four-week drug trial
    # and an explicit contraindication both land here, and telling a reviewer they
    # are the same is wrong. Report the classes actually present.
    _WHY = {
        "PROCURABLE_SLOW": "need a treatment trial or observation period first",
        "BEHAVIOURAL": "depend on the patient accepting a treatment they declined",
        "CLINICAL_FACT": "rest on a measurement no paperwork changes",
        "CATEGORICAL": "fall under an explicit guideline exclusion",
    }
    hard_classes = []
    for g in hard:
        phrase = _WHY.get(g.get("class"))
        if phrase and phrase not in hard_classes:
            hard_classes.append(phrase)

    checklist = [
        f["action_required"]
        for f in (report.get("flagged") or [])
        if f.get("action_required")
    ]

    # Routing, per the decision flow.
    if percentile > MODEL2_REAPPEAL_PERCENTILE:
        route, reason = (
            "HUMAN_REVIEW",
            f"Reappeal risk {p * 100:.1f}% sits in the top "
            f"{100 - percentile:.0f}% of the training population -- this denial is "
            f"the kind that gets overturned.",
        )
    elif fixable_rules:
        route, reason = (
            "HUMAN_REVIEW",
            f"{len(fixable_rules)} of {len(fixable_rules) + len(hard_rules)} gaps "
            f"can be closed with documentation the provider can obtain in days. "
            f"Ask before denying.",
        )
    else:
        route, reason = (
            "AUTO_DENIED",
            "No document the provider could send now would change the outcome: "
            + (
                "the unmet criteria " + "; ".join(hard_classes) + "."
                if hard_classes
                else "no unmet criterion can be closed with documentation."
            )
            + (
                " There is still a resubmission path -- see the checklist."
                if checklist
                else ""
            ),
        )

    dist = _distribution(p, len(fixable_rules), len(hard_rules))
    top = dist[0]

    return {
        "route": route,
        "reason": reason,
        "reappeal_probability": round(p, 4),
        "reappeal_percent": round(p * 100, 1),
        "reappeal_percentile": round(percentile),
        "reappeal_lift": round(p / model.base_rate, 2) if model.base_rate else None,
        "criteria_satisfaction": round(criteria, 1) if criteria is not None else None,
        "fixable_gaps": fixable_rules,
        "hard_gaps": hard_rules,
        "resubmission_checklist": checklist,
        "raises_risk": raises,
        "lowers_risk": lowers,
        "model_version": getattr(model, "VERSION", "unknown"),
        "trained_on": getattr(model, "trained_on", "unknown"),
        # Shaped for AppealForecast in the reviewer UI.
        "appeal_prediction": {
            "top_class": top["outcome"],
            "top_label": top["label"],
            "top_probability": top["probability"],
            "any_appeal_probability": round(p, 4),
            "distribution": dist,
            # macro_auc is the flag the card reads to decide whether to show its
            # "treat as uninformative" caveat. 0.5 records that this model has no
            # measured discrimination on real appeal behaviour, which is true --
            # it was fitted on synthetic labels. The two figures below are its
            # actual reported numbers: PR-AUC against the positive base rate.
            "model_macro_auc": MODEL2_REPORTED_MACRO_AUC,
            "model_accuracy": MODEL2_REPORTED_SCORE,
            "baseline_accuracy": MODEL2_REPORTED_BASELINE,
            "trained_on": getattr(model, "trained_on", "unknown"),
            "assessed": True,
        },
    }
