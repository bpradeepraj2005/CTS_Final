"""
Single source of truth for the feature contract.

Both the training scripts (ml/train_*.py) and the serving layer
(app/services/ml.py) import from here so a model can never be fed a
frame whose columns drifted from the ones it was fitted on.
"""

# Raw clinical / administrative fields that arrive from a PA packet.
RAW_FIELDS = [
    "age", "sex", "bmi", "diagnosis", "diagnosis_code", "disease_severity",
    "symptom_burden_0_10", "symptom_duration_months", "comorbidities",
    "requested_treatment", "dose_category", "frequency", "route",
    "requested_duration_months", "request_reason",
    "previous_treatment_count", "previous_failed_count",
    "previous_partial_response_count", "previous_adverse_effect_count",
    "longest_previous_treatment_weeks",
    "doctor_note_present", "lab_results_present", "imaging_present",
    "medication_history_present", "documentation_complete",
    "provider_specialty", "provider_state", "provider_type",
    "payer", "member_eligible", "treatment_covered",
]

# Fields computed deterministically from RAW_FIELDS (see derive_features).
DERIVED_FIELDS = [
    "bmi_category", "age_group", "total_prior_events",
    "treatment_failure_ratio", "documentation_flag_sum",
    "documentation_score", "documentation_completeness_pct",
]

DOC_FLAGS = [
    "doctor_note_present", "lab_results_present",
    "imaging_present", "medication_history_present",
]

# --- Model input columns -----------------------------------------------------

# Policy-fit regressor: everything except the score it is predicting.
POLICY_FIT_FEATURES = RAW_FIELDS + DERIVED_FIELDS

# Appeal-propensity classifier: same inputs plus the two payer scores that
# exist at the moment the denial is issued.
APPEAL_FEATURES = (
    RAW_FIELDS + DERIVED_FIELDS + ["clinical_evidence_score", "policy_fit_score"]
)

CATEGORICAL = [
    "sex", "diagnosis", "diagnosis_code", "disease_severity", "comorbidities",
    "requested_treatment", "dose_category", "frequency", "route",
    "request_reason", "provider_specialty", "provider_state", "provider_type",
    "payer", "bmi_category", "age_group",
]

APPEAL_CLASSES = [
    "NEVER_APPLIED",
    "REAPPLIED",
    "FORMAL_APPEAL",
    "APPEAL_WITH_NEW_DOCUMENTATION",
]

APPEAL_LABELS = {
    "NEVER_APPLIED": "Unlikely to appeal",
    "REAPPLIED": "Likely to resubmit",
    "FORMAL_APPEAL": "Likely formal appeal",
    "APPEAL_WITH_NEW_DOCUMENTATION": "Likely appeal with new evidence",
}


def bmi_category(bmi: float) -> str:
    if bmi is None:
        return "Normal"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def age_group(age) -> str:
    if age is None:
        return "30-44"
    age = float(age)
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60-74"


def derive_features(row: dict) -> dict:
    """Add the derived columns to a raw feature dict. Pure and deterministic."""
    out = dict(row)
    out["bmi_category"] = bmi_category(_f(row.get("bmi")))
    out["age_group"] = age_group(_f(row.get("age")))

    prior = [
        _i(row.get("previous_treatment_count")),
        _i(row.get("previous_failed_count")),
        _i(row.get("previous_partial_response_count")),
        _i(row.get("previous_adverse_effect_count")),
    ]
    out["total_prior_events"] = sum(prior)

    tried = _i(row.get("previous_treatment_count"))
    failed = _i(row.get("previous_failed_count"))
    out["treatment_failure_ratio"] = round(failed / tried, 4) if tried else 0.0

    flag_sum = sum(_i(row.get(f)) for f in DOC_FLAGS)
    out["documentation_flag_sum"] = flag_sum
    out["documentation_score"] = round(flag_sum / len(DOC_FLAGS), 4)
    out["documentation_completeness_pct"] = out["documentation_score"]
    return out


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default

# --- Appeal regressor -------------------------------------------------------

# Model 2 is served on the denial branch, where the only inputs available are
# the submitted case fields. Deliberately excludes policy_fit_score and
# clinical_evidence_score: both exist in the training CSV, but neither is
# produced by the live pipeline in the same units any more. Model 1's approval
# likelihood is guideline-derived and capped on unmet mandatory criteria, so it
# occupies a different range than the CSV column of the same name. Training on a
# feature whose meaning shifts between fit and serve is how a model quietly gets
# worse in production.
APPEAL_REGRESSOR_FEATURES = RAW_FIELDS + DERIVED_FIELDS
