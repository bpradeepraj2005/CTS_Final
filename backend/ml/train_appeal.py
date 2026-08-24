"""
Trains Model 2 -- the appeal-propensity regressor.

    python ml/train_appeal.py --csv data/appeals_prediction_transformed.csv

Outputs to ml/models/:
    appeal_propensity.joblib   HistGradientBoostingRegressor bundle
    appeal_metrics.json        held-out metrics, surfaced on the model card

Why regressors and not the old classifier
-----------------------------------------
Five HistGradientBoostingRegressors are fitted:

  * `propensity` regresses the 0/1 indicator "this denial was challenged". Its
    prediction is the appeal probability the flow routes on.
  * one per appeal outcome, each regressing that outcome's 0/1 indicator, so the
    four-way distribution the reviewer sees is learned rather than apportioned
    by a rule of thumb.

Held-out predictions from the propensity model are stored in the bundle as
`reference`. A percentile threshold is meaningless without the distribution it
was measured against, so the artifact has to carry it.

The corpus is denial-only (every row is final_decision = DENIED), which is
exactly where Model 2 sits in the flow. That is a fit, not a limitation -- but it
does mean the model knows nothing about approved cases and must never be asked
about one.

Nothing here is fabricated. Whatever the held-out numbers are, they are what
appeal_metrics.json reports and what the app displays.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_schema import (  # noqa: E402
    APPEAL_CLASSES,
    APPEAL_REGRESSOR_FEATURES,
    CATEGORICAL,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
NO_APPEAL = "NEVER_APPLIED"


def as_frame(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    X = df[columns].copy()
    for c in columns:
        if c in CATEGORICAL:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def fit_one(Xtr, ytr, seed=42):
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.5,
        early_stopping=True,
        random_state=seed,
    ).fit(Xtr, ytr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/appeals_prediction_transformed.csv")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    missing = [
        c
        for c in set(APPEAL_REGRESSOR_FEATURES) | {"appeal_status"}
        if c not in df.columns
    ]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")

    if "final_decision" in df.columns:
        decisions = set(df["final_decision"].unique())
        if decisions != {"DENIED"}:
            print(f"  note: corpus contains non-denied rows: {decisions}")

    X = as_frame(df, APPEAL_REGRESSOR_FEATURES)
    status = df["appeal_status"].astype(str)
    y_any = (status != NO_APPEAL).astype(float)

    Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
        X, y_any, status, test_size=0.2, random_state=42, stratify=status
    )
    print(f"  train {len(Xtr):,}   test {len(Xte):,}")
    print(f"  appeal base rate: {y_any.mean():.4f}")

    # --- propensity -------------------------------------------------------
    print("\nFitting appeal-propensity regressor...")
    propensity = fit_one(Xtr, ytr)
    raw = propensity.predict(Xte)
    pred = np.clip(raw, 0.0, 1.0)

    metrics = {
        "task": "regression on the 0/1 appeal indicator",
        "target": "appeal_status != NEVER_APPLIED",
        "model": "HistGradientBoostingRegressor",
        "r2": round(float(r2_score(yte, pred)), 4),
        "mae": round(float(mean_absolute_error(yte, pred)), 4),
        "brier": round(float(brier_score_loss(yte, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte, pred)), 4),
        "pr_auc": round(float(average_precision_score(yte, pred)), 4),
        "base_rate": round(float(y_any.mean()), 4),
        "predictions_out_of_unit_range": int(((raw < 0) | (raw > 1)).sum()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }
    print(json.dumps(metrics, indent=2))

    # --- per-outcome ------------------------------------------------------
    print("\nFitting one regressor per appeal outcome...")
    outcomes, outcome_metrics = {}, {}
    for cls in APPEAL_CLASSES:
        if cls not in set(status):
            print(f"  {cls:32} absent from corpus, skipped")
            continue
        m = fit_one(Xtr, (str_tr == cls).astype(float))
        p = np.clip(m.predict(Xte), 0.0, 1.0)
        truth = (str_te == cls).astype(float)
        outcomes[cls] = m
        outcome_metrics[cls] = {
            "r2": round(float(r2_score(truth, p)), 4),
            "mae": round(float(mean_absolute_error(truth, p)), 4),
            "roc_auc": round(float(roc_auc_score(truth, p)), 4),
            "prevalence": round(float(truth.mean()), 4),
        }
        print(
            f"  {cls:32} r2={outcome_metrics[cls]['r2']:+.4f}  "
            f"auc={outcome_metrics[cls]['roc_auc']:.4f}"
        )

    # Argmax over the four regressors, scored as if it were a classifier, so the
    # distribution is not taken on trust.
    if outcomes:
        names = list(outcomes)
        stacked = np.column_stack(
            [np.clip(outcomes[c].predict(Xte), 0, 1) for c in names]
        )
        argmax = pd.Series([names[i] for i in stacked.argmax(axis=1)], index=str_te.index)
        acc = float((argmax == str_te).mean())
        majority = float(str_te.value_counts(normalize=True).max())
        metrics["distribution_argmax_accuracy"] = round(acc, 4)
        metrics["majority_class_baseline"] = round(majority, 4)
        metrics["distribution_lift_over_baseline"] = round(acc - majority, 4)
        print(
            f"\n  argmax accuracy {acc:.4f} vs majority baseline {majority:.4f} "
            f"({acc - majority:+.4f})"
        )

    joblib.dump(
        {
            "propensity": propensity,
            "outcomes": outcomes,
            "features": APPEAL_REGRESSOR_FEATURES,
            "categorical": CATEGORICAL,
            "classes": list(outcomes),
            "no_appeal_class": NO_APPEAL,
            # Held-out predictions: the distribution percentile thresholds are
            # measured against. Without it a percentile means nothing.
            "reference": np.sort(pred).astype("float32"),
            "base_rate": float(y_any.mean()),
            "trained_on": f"{Path(args.csv).name} (n={len(df):,}, denial-only)",
            "version": "appeal-propensity-hgbr-v1",
        },
        MODELS_DIR / "appeal_propensity.joblib",
    )

    card = {
        "dataset_rows": int(len(df)),
        "source_csv": Path(args.csv).name,
        "corpus": "denied requests only",
        "features_used": len(APPEAL_REGRESSOR_FEATURES),
        "excluded_features": ["policy_fit_score", "clinical_evidence_score"],
        "propensity": metrics,
        "per_outcome": outcome_metrics,
        "notes": [
            "Metrics are computed on a held-out 20% split and reported verbatim.",
            "policy_fit_score and clinical_evidence_score are excluded: the live "
            "pipeline no longer produces them in the units this CSV recorded.",
            "The corpus is denial-only, which matches where Model 2 runs. It "
            "knows nothing about approved requests.",
        ],
    }
    (MODELS_DIR / "appeal_metrics.json").write_text(json.dumps(card, indent=2))
    print(f"\nWrote {MODELS_DIR / 'appeal_propensity.joblib'}")
    print(f"Wrote {MODELS_DIR / 'appeal_metrics.json'}")


if __name__ == "__main__":
    main()
