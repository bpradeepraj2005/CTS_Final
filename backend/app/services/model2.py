"""
Model 2 -- supporting-material assessment on the denial path.

Two independent halves, and the split matters:

  * `curability.gaps()` sorts the Report's unmet criteria into gaps a provider can
    close with paperwork and gaps no document will fix. Rule-based, auditable,
    and what actually decides routing.
  * a HistGradientBoostingRegressor bundle, trained by ml/train_appeal.py on
    appeals_prediction_transformed.csv, scores how likely the denial is to be
    challenged and how that appeal would arrive.

Position in the flow: the Decision Router sends approvals straight to Auto
Approval. Only denials reach Model 2, which asks one question -- can the provider
fix this with more documentation?

    fixable gaps present  -> Human Review        (material can be supplied)
    only hard gaps        -> Auto Denial         (no supporting material path)
    high reappeal risk    -> Human Review        (override; likely overturned)

Read the caveat on `info()` before trusting the percentages. The regressor scores
ROC-AUC 0.537 on held-out data against 0.500 for random guessing, so it barely
separates cases at all. The routing above is unaffected: it turns on the
rule-based gap split, and the percentile override is the only place the model can
change an outcome.
"""
import json
import threading

import joblib
import pandas as pd

from ..config import (
    MODEL2_ENABLED,
    MODEL2_METRICS_PATH,
    MODEL2_PATH,
    MODEL2_REAPPEAL_PERCENTILE,
)
from . import curability
from .prior_auth_client import ModelUnavailable

_bundle = None
_metrics: dict | None = None
_load_lock = threading.Lock()

APPEAL_LABELS = {
    "NEVER_APPLIED": "Unlikely to appeal",
    "REAPPLIED": "Likely to resubmit",
    "FORMAL_APPEAL": "Likely formal appeal",
    "APPEAL_WITH_NEW_DOCUMENTATION": "Likely appeal with new evidence",
}


def _load():
    """Load the joblib bundle once per process, lazily."""
    global _bundle
    if _bundle is not None:
        return _bundle
    with _load_lock:
        if _bundle is not None:
            return _bundle
        if not MODEL2_PATH.exists():
            raise ModelUnavailable(
                f"Appeal model is missing at {MODEL2_PATH}. Run: "
                f"python ml/train_appeal.py --csv data/appeals_prediction_transformed.csv"
            )
        try:
            bundle = joblib.load(MODEL2_PATH)
        except Exception as exc:
            raise ModelUnavailable(f"Could not load appeal model: {exc}") from exc
        if "propensity" not in bundle:
            raise ModelUnavailable(
                f"{MODEL2_PATH} is not an appeal-regressor bundle (no 'propensity' "
                f"key). Retrain with ml/train_appeal.py."
            )
        _bundle = bundle
    return _bundle


def _held_out_metrics() -> dict:
    global _metrics
    if _metrics is None:
        try:
            _metrics = json.loads(MODEL2_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _metrics = {}
    return _metrics


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
        bundle = _load()
    except ModelUnavailable as exc:
        return {"available": False, "reason": str(exc)}

    card = _held_out_metrics()
    prop = card.get("propensity", {})
    auc = prop.get("roc_auc")

    return {
        "available": True,
        "version": bundle.get("version", "unknown"),
        "model": "HistGradientBoostingRegressor",
        "trained_on": bundle.get("trained_on", "unknown"),
        "base_rate": round(float(bundle.get("base_rate", 0.0)), 4),
        "training_rows": card.get("dataset_rows"),
        "source_csv": card.get("source_csv"),
        "features_used": len(bundle.get("features") or []),
        "reappeal_percentile_threshold": MODEL2_REAPPEAL_PERCENTILE,
        "roc_auc": auc,
        "pr_auc": prop.get("pr_auc"),
        "r2": prop.get("r2"),
        "brier": prop.get("brier"),
        "distribution_argmax_accuracy": prop.get("distribution_argmax_accuracy"),
        "majority_class_baseline": prop.get("majority_class_baseline"),
        "per_outcome": card.get("per_outcome", {}),
        "caveat": (
            f"Held-out ROC-AUC is {auc} against 0.500 for random guessing, so this "
            f"model barely separates cases. The four-way distribution beats its "
            f"majority-class baseline by "
            f"{prop.get('distribution_lift_over_baseline')} of accuracy. The "
            f"features in this corpus carry almost no information about whether a "
            f"denial gets appealed -- treat the percentages as ranking hints, not "
            f"rates. Routing does not depend on them."
            if auc is not None
            else "Held-out metrics were not found; run ml/train_appeal.py."
        ),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _frame(features: dict, columns: list) -> pd.DataFrame:
    """Build the one-row frame the regressors were fitted on."""
    from ml.feature_schema import CATEGORICAL, derive_features

    full = derive_features(features)
    row = {c: full.get(c) for c in columns}
    frame = pd.DataFrame([row], columns=columns)
    for c in columns:
        if c in CATEGORICAL:
            frame[c] = frame[c].fillna("Unknown").astype("category")
        else:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def _clip(v: float) -> float:
    """Least-squares regression on a 0/1 target can land outside the unit
    interval. Clip rather than pretend it cannot happen."""
    return float(max(0.0, min(1.0, v)))


def _percentile(bundle, p: float) -> float:
    """Where this score falls in the held-out distribution."""
    ref = bundle.get("reference")
    if ref is None or len(ref) == 0:
        return 50.0
    import numpy as np

    return float((np.asarray(ref) < p).mean() * 100)


def _distribution(bundle, frame) -> list[dict]:
    """Learned four-way distribution, one regressor per outcome, normalised."""
    outcomes = bundle.get("outcomes") or {}
    if not outcomes:
        return []
    raw = {name: _clip(m.predict(frame)[0]) for name, m in outcomes.items()}
    total = sum(raw.values())
    if total <= 0:
        return []
    return [
        {
            "outcome": name,
            "label": APPEAL_LABELS.get(name, name),
            "probability": round(value / total, 4),
        }
        for name, value in sorted(raw.items(), key=lambda kv: -kv[1])
    ]


def empty_prediction(reason: str) -> dict:
    """Placeholder with the exact key set the reviewer UI reads.

    Approved requests never reach Model 2, so their appeal card has nothing to
    show. None rather than 0: these surface as an "Appeal risk" column in the
    request and review lists, where 0 would read as a confident prediction of no
    appeal rather than a question that was never asked.
    """
    return {
        "top_class": "NOT_CALCULATED",
        "top_label": reason,
        "top_probability": None,
        "any_appeal_probability": None,
        "distribution": [],
        "model_macro_auc": None,
        "model_accuracy": None,
        "baseline_accuracy": None,
        "assessed": False,
    }


def assess(report: dict, features: dict, context: dict | None = None) -> dict:
    """Run the supporting-material assessment on one denied case.

    `report` is Model 1's output and drives the gap split. `features` is the
    submitted case, which is what the regressor was fitted on.
    """
    bundle = _load()
    card = _held_out_metrics().get("propensity", {})

    try:
        frame = _frame(features, bundle["features"])
        p = _clip(bundle["propensity"].predict(frame)[0])
        percentile = _percentile(bundle, p)
        dist = _distribution(bundle, frame)
    except Exception as exc:
        raise ModelUnavailable(f"Appeal scoring failed: {exc}") from exc

    fixable, hard = curability.gaps(report)
    fixable_rules = [g["rule"] for g in fixable]
    hard_rules = [g["rule"] for g in hard]

    # Why the hard gaps are hard. "Hard" is not one thing: a four-week drug trial
    # and an explicit contraindication both land here, and telling a reviewer they
    # are the same is wrong.
    hard_classes = []
    for g in hard:
        phrase = curability.WHY_HARD.get(g["class"])
        if phrase and phrase not in hard_classes:
            hard_classes.append(phrase)

    checklist = [
        f["action_required"]
        for f in (report.get("flagged") or [])
        if f.get("action_required")
    ]

    # Routing, per the decision flow.
    if percentile > MODEL2_REAPPEAL_PERCENTILE:
        route = "HUMAN_REVIEW"
        reason = (
            f"Reappeal risk {p * 100:.1f}% sits in the top "
            f"{100 - percentile:.0f}% of the training population -- this denial is "
            f"the kind that gets overturned."
        )
    elif fixable_rules:
        route = "HUMAN_REVIEW"
        reason = (
            f"{len(fixable_rules)} of {len(fixable_rules) + len(hard_rules)} gaps "
            f"can be closed with documentation the provider can obtain in days. "
            f"Ask before denying."
        )
    else:
        route = "AUTO_DENIED"
        reason = "No document the provider could send now would change the outcome: " + (
            "the unmet criteria " + "; ".join(hard_classes) + "."
            if hard_classes
            else "no unmet criterion can be closed with documentation."
        )
        if checklist:
            reason += " There is still a resubmission path -- see the checklist."

    top = dist[0] if dist else None
    base_rate = float(bundle.get("base_rate") or 0.0)

    return {
        "route": route,
        "reason": reason,
        "reappeal_probability": round(p, 4),
        "reappeal_percent": round(p * 100, 1),
        "reappeal_percentile": round(percentile),
        "reappeal_lift": round(p / base_rate, 2) if base_rate else None,
        "criteria_satisfaction": _criteria_satisfaction(report),
        "fixable_gaps": fixable_rules,
        "hard_gaps": hard_rules,
        "resubmission_checklist": checklist,
        "curability_index": round(curability.index([g["class"] for g in fixable + hard]), 4),
        "model_version": bundle.get("version", "unknown"),
        "trained_on": bundle.get("trained_on", "unknown"),
        # Shaped for AppealForecast in the reviewer UI.
        "appeal_prediction": {
            "top_class": top["outcome"] if top else "NOT_CALCULATED",
            "top_label": top["label"] if top else "No outcome distribution available",
            "top_probability": top["probability"] if top else None,
            "any_appeal_probability": round(p, 4),
            "distribution": dist,
            # macro_auc is the flag the appeal card reads to decide whether to
            # show its "treat as uninformative" caveat. This is the real held-out
            # ROC-AUC, and at ~0.54 it correctly trips that warning.
            "model_macro_auc": card.get("roc_auc"),
            "model_accuracy": card.get("distribution_argmax_accuracy"),
            "baseline_accuracy": card.get("majority_class_baseline"),
            "trained_on": bundle.get("trained_on", "unknown"),
            "assessed": True,
        },
    }


def _criteria_satisfaction(report: dict) -> float | None:
    """Weighted share of guideline criteria met, 0..100, on the same obligation
    weights ml.approval_likelihood uses."""
    from .ml import approval_likelihood

    rules = report.get("rules") or []
    if not rules:
        return None
    return round(approval_likelihood(report) * 100, 1)
