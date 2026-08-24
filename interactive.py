"""Single-prompt input for a complete pasted prior-authorization application."""

from __future__ import annotations

import re
from typing import Any


def _coerce(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in ("true", "yes", "y"):
        return True
    if lowered in ("false", "no", "n"):
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _find(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _number(pattern: str, text: str) -> int | float | None:
    raw = _find(pattern, text)
    return _coerce(raw) if raw is not None else None


def _best_record(text: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    best = None
    best_score = 0
    for record in records:
        condition_words = set(re.findall(r"[a-z0-9]+", record["condition"].lower()))
        score = len(words & condition_words)
        if score > best_score:
            best, best_score = record, score
    return best


def _build_request(text: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    # Pasted reports often concatenate labels (for example, Age: 71Gender:
    # Male). Insert boundaries before known labels before applying extraction.
    labels = [
        "Patient ID", "Claim ID", "Submission Date", "Patient Name", "Age", "Gender", "Sex",
        "Primary Diagnosis", "Diagnosis", "Requested Procedure",
        "Requested Intervention", "Care Level", "Blood Pressure", "Heart Rate",
        "Previous Treatment", "Clinical Notes", "Reason for Request",
    ]
    for label in labels:
        text = re.sub(rf"(?<!\n)\s*({re.escape(label)}\s*:)", r"\n\1", text, flags=re.IGNORECASE)

    record = _best_record(text, records)
    age = _number(r"(?:patient\s+)?age\s*(?:is|=|:)\s*(\d+)", text)
    sex = _find(r"(?:patient\s+)?(?:sex|gender)\s*(?:is|=|:)\s*(male|female|m|f)\b", text)
    icd10 = re.findall(r"\b[A-Z]\d{2}(?:\.\d+)?\b", text.upper())
    diagnosis = _find(r"(?:(?:primary\s+)?diagnosis|diagnosed\s+with)\s*(?:is|=|:)\s*([^\n;]+)", text)
    service = _find(r"requested\s+(?:service|procedure|intervention)\s*(?:is|=|:)\s*([^\n;]+)", text)
    if record:
        diagnosis = diagnosis or record["condition"]
        service = service or record["condition"]

    findings: list[dict[str, Any]] = []
    systolic_bp = _number(r"(?:blood\s+pressure|bp)\s*(?:is|=|:)\s*(\d+)\s*/\s*\d+", text)
    heart_rate = _number(r"(?:heart\s+rate|hr|pulse)\s*(?:is|=|:)\s*(\d+)", text)
    if systolic_bp is not None:
        findings.append({"parameter": "systolic_bp", "value": systolic_bp, "confidence": "high", "provenance": "pasted_application"})
    if heart_rate is not None:
        findings.append({"parameter": "heart_rate", "value": heart_rate, "confidence": "high", "provenance": "pasted_application"})
    numeric_fields = {
        "uterine_size_weeks": r"uterine\s+size\s*(?:is|=|:)\s*([\d.]+)",
        "submucous_myoma_size_cm": r"(?:submucous|submucus)\s+(?:myoma|fibroid)\s*(?:size\s*)?(?:is|=|:)\s*([\d.]+)\s*cm",
        "stone_size_mm": r"stone\s+size\s*(?:is|=|:)\s*([\d.]+)\s*mm",
        "egfr": r"e?gfr\s*(?:is|=|:)\s*([\d.]+)",
        "hours_from_onset": r"(?:hours?|time)\s+from\s+onset\s*(?:is|=|:)\s*([\d.]+)",
        "platelet_count": r"platelet(?:\s+count)?\s*(?:is|=|:)\s*([\d.]+)",
        "episodes_last_12_months": r"(?:episodes|attacks)\s+(?:in\s+the\s+)?last\s+12\s+months\s*(?:is|are|=|:)\s*(\d+)",
        "centor_score": r"centor\s+score\s*(?:is|=|:)\s*(\d+)",
        "failed_antipsychotic_trials": r"failed\s+antipsychotic\s+trials?\s*(?:is|are|=|:)\s*(\d+)",
        "fever_duration_hours": r"fever\s+duration\s*(?:is|=|:)\s*([\d.]+)\s*(?:hours?|hrs?)",
        "met_trial_duration_weeks": r"(?:met|medical\s+expulsive\s+therapy)\s+trial\s*(?:duration\s*)?(?:is|=|:)\s*([\d.]+)\s*weeks?",
    }
    for parameter, pattern in numeric_fields.items():
        value = _number(pattern, text)
        if value is not None:
            findings.append({"parameter": parameter, "value": value, "confidence": "high", "provenance": "pasted_application"})

    boolean_fields = {
        "second_opinion_documentation": r"second\s+opinion(?:\s+documentation)?\s*(?:is|=|:)\s*(true|false|yes|no)",
        "ct_or_mri_done": r"(?:ct|mri)(?:\s+scan)?\s+(?:done|completed)\s*(?:is|=|:)\s*(true|false|yes|no)",
        "infection_present": r"infection\s+(?:is\s+)?present\s*(?:is|=|:)\s*(true|false|yes|no)",
        "obstruction_present": r"obstruction\s+(?:is\s+)?present\s*(?:is|=|:)\s*(true|false|yes|no)",
        "active_bleeding": r"active\s+bleeding\s*(?:is|=|:)\s*(true|false|yes|no)",
        "prolonged_shock": r"prolonged\s+shock\s*(?:is|=|:)\s*(true|false|yes|no)",
        "organic_cause_excluded": r"organic\s+cause\s+(?:is\s+)?excluded\s*(?:is|=|:)\s*(true|false|yes|no)",
        "deranged_renal_function": r"deranged\s+renal\s+function\s*(?:is|=|:)\s*(true|false|yes|no)",
    }
    for parameter, pattern in boolean_fields.items():
        raw = _find(pattern, text)
        if raw is not None:
            findings.append({"parameter": parameter, "value": _coerce(raw), "confidence": "high", "provenance": "pasted_application"})

    # Closed-list indications are accepted only when the pasted application
    # explicitly mentions the indication, or explicitly says it is absent.
    indication_phrases = {
        "fluid_overload": r"fluid\s+overload",
        "pericarditis": r"pericarditis",
        "hyperkalemia": r"hyperkal(?:emia|aemia)",
        "severe_metabolic_acidosis": r"severe\s+metabolic\s+acidosis",
        "encephalopathy": r"encephalopathy",
        "severe_uraemia": r"severe\s+(?:uraemia|uremia)",
        "need_dialysis_access_for_fluids_or_blood": r"(?:create|make)\s+space\s+for\s+(?:fluids?|blood\s+products?)",
    }
    for parameter, phrase in indication_phrases.items():
        if not re.search(phrase, text, re.IGNORECASE):
            continue
        negative = re.search(r"(?:no|without|absent|denies)\s+(?:[\w-]+\s+){0,3}" + phrase, text, re.IGNORECASE)
        findings.append({"parameter": parameter, "value": negative is None, "confidence": "high", "provenance": "pasted_application"})

    # Convert explicit AF facts into the exact indication wording used by the
    # generated criteria. This is normalization, not a clinical guess.
    signals: list[str] = []
    if (systolic_bp is not None and systolic_bp < 90) or re.search(r"hypotension|hemodynamic(?:ally)?\s+unstable", text, re.IGNORECASE):
        signals.append("Hemodynamic instability")
    if heart_rate is not None and heart_rate > 130:
        signals.append("Very rapid HR greater than 130/min not controlled")
    # Broadly detect a rate-control trial failure even when the submitter
    # phrases it as "despite 4 weeks of metoprolol 50 mg BID" rather than
    # the literal token "rate control failed". If the ventricular rate is
    # elevated and there is evidence of a beta-blocker trial or an explicit
    # failure/"despite" phrase, mark the rate-control-failure signal.
    if heart_rate is not None and heart_rate > 110:
        rc_phrase = re.search(r"rate[- ]control.*(?:inadequate|failed|unsuccessful)|(?:inadequate|failed|unsuccessful).*rate[- ]control", text, re.IGNORECASE)
        met_therapy = re.search(r"\bmetoprolol\b|\bbeta[- ]?blocker\b|\bbisoprolol\b|\batenolol\b", text, re.IGNORECASE)
        met_duration = re.search(r"(\d+)\s*weeks?\s*(?:of|on)\s*(?:metoprolol|beta[- ]?blocker)", text, re.IGNORECASE)
        despite_phrase = re.search(r"despite|not controlled|uncontrolled|remains.*\b\d+\b|still.*\b\d+\b", text, re.IGNORECASE)
        if rc_phrase or (met_therapy and (despite_phrase or met_duration)):
            signals.append("HR remains greater than 110/min after rate control attempt")

    return {
        "request_id": _find(r"(?:request\s+id|case\s+id|claim\s+id|patient\s+id)\s*(?:is|=|:)\s*([^\n;]+)", text) or "PA-PASTED-0001",
        "patient_name": _find(r"patient\s+name\s*(?:is|=|:)\s*([^\n;]+)", text) or "",
        "patient": {key: value for key, value in (("age", age), ("sex", sex.upper() if sex else None)) if value is not None},
        "diagnosis": {"icd10": icd10, "text": diagnosis or ""},
        "requested_service": {"code": "", "text": service or "", "facility_level": _find(r"facility\s+level\s*(?:is|=|:)\s*([^\n;]+)", text) or ""},
        "clinical_findings": findings,
        "clinical_notes": text + ("\nExplicit extracted STW signals: " + "; ".join(signals) if signals else ""),
        "eligibility": {
            "enrollment_active_on_service_date": True, "benefit_covers_service": True,
            "service_not_in_plan_exclusions": True, "waiting_period_satisfied": True,
            "annual_or_lifetime_limit_available": True, "provider_empanelled": True,
            "prior_auth_actually_required": True, "no_duplicate_active_authorization": True,
        },
    }


def build_request_interactively(records: list[dict[str, Any]]) -> dict[str, Any]:
    print("Paste the complete patient/prior-authorization application in one input.")
    print("When finished, type END on a new line and press Enter.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        raise SystemExit("No application text was provided.")
    return _build_request(text, records)
