"""Phase F1/F3 -- turn an incoming request into the facts dict the rule
engine, eligibility gate, and scoring model all read from.

The spec's F1 phase is an LLM call that extracts structured facts from free
text. This build has no LLM API available, so the request format documented
here **is** the extracted-facts schema directly -- callers submit the
structured artifact, not a free-text note. This is a deliberate scope
boundary (see README), not a shortcut: it keeps the invariant the whole
architecture depends on -- "never let an unverified value become a
denial" -- because there is no inference step that could hallucinate one.

Request shape:
{
  "request_id": "PA-2026-0084412",
  "patient": {"age": 34, "sex": "F"},
  "diagnosis": {"icd10": ["D25.9"], "text": "symptomatic uterine fibroid"},
  "requested_service": {"code": "PA-INT-OBG001", "text": "total abdominal hysterectomy",
                         "setting": "inpatient", "facility_level": "district"},
  "clinical_findings": [
    {"parameter": "uterine_size_weeks", "value": 10, "unit": "weeks", "confidence": "high"},
    {"parameter": "second_opinion_documentation", "value": false, "confidence": "high"}
  ],
  "prior_therapies": [{"therapy": "tranexamic acid", "duration_months": 2, "outcome": "partial response"}],
  "documentation_present": ["ultrasound_report", "hemogram"],
  "documentation_absent": ["second_opinion"],
  "documents": [ { "doc_type": ..., "present": true, "legible_or_parseable": true,
                   "within_validity_window": true, "contains_required_fields": [...],
                   "provenance": "signed_report" } ],   # optional; richer than documentation_present/absent
  "eligibility": { "enrollment_active_on_service_date": true, ... }   # optional; see eligibility.py
}

`clinical_findings` carries numeric, boolean, and categorical values alike --
whatever a rule's `logic` references by field name. A field with
`"confidence": "low"` is treated as absent everywhere downstream (spec F3:
"treat low-confidence extractions as absent").
"""

from __future__ import annotations

import re
from typing import Any
try:
    from . import documentation as _documentation
except ImportError:
    import documentation as _documentation

try:
    from .facility import rank_from_level
except ImportError:
    from facility import rank_from_level


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_facts(request: dict[str, Any]) -> dict[str, Any]:
    """Flatten a request into the facts dict consumed by rule_engine, and
    keep the extraction-confidence map so the scoring model can compute
    evidence quality and information sufficiency."""
    patient = request.get("patient") or {}
    diagnosis = request.get("diagnosis") or {}
    requested_service = request.get("requested_service") or {}

    facts: dict[str, Any] = {}
    confidence: dict[str, str] = {}
    provenance: dict[str, str] = {}

    for key in ("age", "sex", "pregnancy_status"):
        if patient.get(key) is not None:
            facts[key] = patient[key]
            confidence[key] = "high"

    for finding in request.get("clinical_findings") or []:
        param = finding.get("parameter")
        if not param:
            continue
        field_confidence = finding.get("confidence", "high")
        confidence[param] = field_confidence
        facts[param] = finding.get("value") if field_confidence != "low" else None
        provenance[param] = finding.get("provenance", "structured_report")

    for therapy in request.get("prior_therapies") or []:
        name = therapy.get("therapy")
        if not name:
            continue
        base = _slug(name)
        if therapy.get("duration_months") is not None:
            facts.setdefault(f"{base}_duration_months", therapy["duration_months"])
        if therapy.get("outcome") is not None:
            facts.setdefault(f"{base}_outcome", therapy["outcome"])

    facts["diagnosis"] = diagnosis
    facts["requested_service"] = requested_service
    facts["facility_level_rank"] = rank_from_level(requested_service.get("facility_level"))
    confidence["facility_level_rank"] = "high" if facts["facility_level_rank"] is not None else "low"

    if "indication_text" not in facts:
        base_ind = " ".join(
            str(v) for v in (diagnosis.get("text"), requested_service.get("text"), request.get("clinical_notes")) if v
        ) or ""
        # Synthesize STW normalized indication phrases from structured findings
        signals: list[str] = []
        hr = facts.get("heart_rate")
        if hr is not None and hr > 130:
            signals.append("Very rapid HR greater than 130/min not controlled")
        if hr is not None and hr > 110:
            # detect a prior trial of rate control from prior_therapies or facts
            met_trial = False
            # prior_therapies are not carried directly into facts; detect common keys
            if facts.get("met_trial_duration_weeks"):
                met_trial = True
            if facts.get("metoprolol") is True or facts.get("beta_blocker_trial") is True:
                met_trial = True
            if met_trial:
                signals.append("HR remains greater than 110/min after rate control attempt")
        indication_text_final = (base_ind + " " + "; ".join(signals)).strip()
        facts["indication_text"] = indication_text_final or None
        confidence["indication_text"] = "medium" if facts["indication_text"] else "low"
    # Compute HAS-BLED when not explicitly provided in the incoming facts.
    # This is a best-effort calculation from available structured findings.
    if facts.get("has_bled_score") is None:
        hb = 0
        age = patient.get("age")
        if age is not None and age > 65:
            hb += 1
        # Hypertension: explicit flag or elevated systolic BP
        if facts.get("hypertension") is True or (facts.get("systolic_bp") is not None and facts.get("systolic_bp") >= 140):
            hb += 1
        # Abnormal renal/liver function: deranged_renal_function flag or low egfr
        if facts.get("deranged_renal_function") is True or (facts.get("egfr") is not None and isinstance(facts.get("egfr"), (int, float)) and facts.get("egfr") < 60):
            hb += 1
        # Stroke history
        if facts.get("stroke") is True or facts.get("prior_stroke") is True:
            hb += 1
        # History of bleeding / predisposition
        if facts.get("history_of_bleeding") is True or facts.get("bleeding_tendency") is True:
            hb += 1
        # Labile INR (best-effort): if caller provided an explicit flag
        if facts.get("labile_inr") is True:
            hb += 1
        # Drugs/alcohol
        if facts.get("drugs_aspirin") is True or facts.get("concomitant_antiplatelet") is True or facts.get("excess_alcohol") is True:
            hb += 1
        facts["has_bled_score"] = hb
        confidence["has_bled_score"] = "calculated" if hb >= 0 else "low"

    # Normalize richer `documents[]` entries into the flat documentation_present
    # list so downstream checks reliably see attachments declared as present.
    docs_present = set(request.get("documentation_present") or [])
    for doc in request.get("documents") or []:
        if not doc.get("present"):
            continue
        # Prefer an explicit doc_type if supplied
        doc_type = doc.get("doc_type")
        if doc_type:
            docs_present.add(_slug(str(doc_type)))
            continue
        # Otherwise attempt to match free-text content against known synonyms
        text = str(doc.get("text") or "").lower()
        for canonical, syns in _documentation.SYNONYMS.items():
            for syn in syns:
                if syn in text:
                    docs_present.add(canonical)
                    break
    # Overwrite the returned documentation_present to the normalized list
    documentation_present_normalized = list(docs_present)
    return {
        "facts": facts,
        "extraction_confidence": confidence,
        "field_provenance": provenance,
        "documentation_present": documentation_present_normalized,
        "documentation_absent": list(request.get("documentation_absent") or []),
        "documents": list(request.get("documents") or []),
        "eligibility": request.get("eligibility") or {},
    }
