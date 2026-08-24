"""Plain-text summary rendering: which STW criteria the patient passed,
which failed, which are still unknown, and the bottom-line score used to
accept/reject/pend the request. This is a presentation layer only -- every
number and verdict here comes straight out of the decision packet computed
by decision_engine.adjudicate; nothing is recomputed or judged here.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from .scoring_model import is_genuinely_failed
except ImportError:
    from scoring_model import is_genuinely_failed

_DECISION_HEADLINE = {
    "approve": "PASSES STW CRITERIA -- Approve",
    "pend": "DOES NOT CLEARLY PASS -- Pend for nurse review",
    "request_more_information": "INCOMPLETE -- Request more information before a decision can be made",
    "deny": "FAILS STW CRITERIA -- Deny",
    "not_covered": "NOT EVALUATED -- Not covered by policy (eligibility failed before any clinical check ran)",
    "no_authorization_needed": "NOT EVALUATED -- No prior authorization is required for this service",
}

def _display_bucket(c: dict[str, Any]) -> str:
    """Which section a criterion should read under -- delegates the
    exclusion-polarity flip to scoring_model.is_genuinely_failed so this
    file and the rationale text in decision_engine.py can never drift
    apart on what counts as a "failure" for exclusion-type criteria."""
    verdict = c["verdict"]
    if verdict in ("insufficient", "not_applicable"):
        return verdict
    return "fail" if is_genuinely_failed(c["type"], verdict) else "pass"


def _criterion_line(c: dict[str, Any]) -> str:
    conf = f" (confidence {c['confidence']:.2f})" if c.get("confidence") is not None else ""
    page = c.get("citation", {}).get("page")
    note = " [exclusion criterion -- matched, i.e. this IS why it fails]" if c["type"] == "exclusion" and c["verdict"] == "pass" else ""
    return f"  [{c['criterion_id']}] ({c['type']}, weight {c['weight']}){conf} p.{page}: {c['text']}{note}"


def render_text(packet: dict[str, Any]) -> str:
    decision = packet.get("decision", "unknown")
    status = "APPROVE" if decision == "approve" else "REJECT" if decision in ("deny", "not_covered") else "MORE INFORMATION NEEDED"
    lines: list[str] = [f"Request: {packet.get('request_id')}", ""]

    score = packet.get("score")
    completeness = packet.get("completeness")
    if score is not None:
        lines.append(f"CLINICAL RULE SCORE: {score}/100")
        lines.append(f"EVIDENCE COMPLETENESS: {completeness}%")
        lines.append(f"DECISION CONFIDENCE: {packet.get('confidence_score', completeness)}/100")
    lines.append("")

    criteria = packet.get("criteria", [])
    passed = [c for c in criteria if _display_bucket(c) == "pass"]
    failed = [c for c in criteria if _display_bucket(c) == "fail"]
    unknown = [c for c in criteria if _display_bucket(c) == "insufficient"]

    if passed:
        lines.append(f"PASSED ({len(passed)}):")
        lines.extend(_criterion_line(c) for c in passed)
        lines.append("")
    if failed:
        lines.append(f"FAILED ({len(failed)}):")
        lines.extend(_criterion_line(c) for c in failed)
        lines.append("")
    if unknown:
        lines.append(f"UNKNOWN -- missing data ({len(unknown)}):")
        lines.extend(_criterion_line(c) for c in unknown)
        lines.append("")

    requested = packet.get("requested_information", [])
    if requested:
        lines.append("INFORMATION NEEDED:")
        for item in requested:
            lines.append(f"  - {item['item']} ({item['criticality']}) -- {item['why']}")
        lines.append("")

    lines.append(f"RATIONALE: {packet.get('clinical_rationale')}")
    lines.append("")
    lines.append("FINAL DECISION:")
    lines.append(status)
    return "\n".join(lines)


def render_json_report(packet: dict[str, Any], request: dict[str, Any]) -> str:
    """Render the audit-safe decision packet in the requested report shape."""
    decision = packet.get("decision", "unknown")
    final_decision = "APPROVE" if decision == "approve" else "REJECT" if decision in ("deny", "not_covered") else "MORE INFORMATION NEEDED"
    patient = request.get("patient") or {}
    diagnosis = request.get("diagnosis") or {}
    service = request.get("requested_service") or {}

    checks = []
    for criterion in packet.get("criteria", []):
        verdict = criterion.get("verdict")
        # Determine display bucket (pass/fail/insufficient) from verdict
        display = _display_bucket(criterion)
        adverse = bool(criterion.get("adverse"))
        # Compute compliance flag respecting adverse polarity
        if display in ("insufficient", "not_applicable"):
            flag = "MISSING_DOCUMENTATION"
        else:
            if adverse:
                # For adverse rules, PASS means an adverse feature -> non-compliant
                flag = "NON_COMPLIANT" if display == "pass" else "COMPLIANT"
            else:
                flag = "COMPLIANT" if display == "pass" else "NON_COMPLIANT"

        # Build a readable evidence string from the evidence map if available
        evidence = criterion.get("evidence") or {}
        if evidence:
            ev_parts = [f"{k}: {v!s}" for k, v in evidence.items() if v is not None]
            extracted = ", ".join(ev_parts) if ev_parts else (", ".join(criterion.get("missing_fields") or []) or criterion.get("text"))
        else:
            extracted = criterion.get("text") if display == "pass" else (", ".join(criterion.get("missing_fields") or []) or criterion.get("text"))

        checks.append({
            "rule_name": criterion.get("criterion_id"),
            "guideline_requirement": criterion.get("text"),
            "extracted_patient_value": extracted,
            "compliance_flag": flag,
            "stw_citation": f"{criterion.get('citation', {}).get('condition', '')} STW, Page {criterion.get('citation', {}).get('page')}"
        })

    documentation = packet.get("pillars", {}).get("documentation", {})
    missing_items = [item.get("item") for item in packet.get("requested_information", [])]
    missing_items.extend(documentation.get("required_missing", []))
    missing_items = list(dict.fromkeys(item for item in missing_items if item))

    service_text = service.get("text", "") or ""
    # Avoid returning an excessively long / pasted-clinical-text string for
    # `requested_procedure`. Prefer the first sentence or first line when
    # the supplied text looks like a pasted justification.
    if len(service_text) > 200 and ("\n" in service_text or "Provider Justification" in service_text):
        first_sentence = re.split(r'(?<=[.!?])\s+', service_text.strip())[0]
        service_text = first_sentence if len(first_sentence) > 10 else service_text.splitlines()[0][:200]

    return json.dumps({
        "claim_id": request.get("request_id", packet.get("request_id")),
        "patient_name": request.get("patient_name") or patient.get("name", ""),
        "icd_code_detected": (diagnosis.get("icd10") or [""])[0],
        "condition_detected": diagnosis.get("text", ""),
        "requested_procedure": service_text,
        "rule_checks": checks,
        "missing_items": missing_items,
        "flagged_non_compliant_items": [c["rule_name"] for c in checks if c["compliance_flag"] == "NON_COMPLIANT"],
        "required_documents_for_resubmission": missing_items,
        "ai_clinical_explanation": packet.get("clinical_rationale", ""),
        "confidence_score": packet.get("confidence_score", 0.0),
        "final_decision": final_decision,
    }, indent=2)
