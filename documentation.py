"""Pillar 3 -- Documentation Completeness (scoring model section 5).

Derives the checklist from the STW investigation tiers of the matched
condition record: essential -> required, desirable -> supporting,
optional -> no penalty (never scored). Feeds Axis 2 (confidence) only --
never the necessity score. A present document can still be worthless, so
when the request supplies the richer `documents[]` form (doc_type, present,
legible_or_parseable, within_validity_window, contains_required_fields,
provenance) that is honored; the flat documentation_present/absent lists
are treated as "present and presumed current" when that richer form is
absent, which is a scope simplification documented in the README.
"""

from __future__ import annotations

import re
from typing import Any

# Heuristic synonyms between common request-side document identifiers and
# STW investigation-list wording. A production system would replace this
# with a controlled document-type taxonomy (the same caution the RAG spec
# gives about drug-name normalization in Phase A2 applies here).
SYNONYMS = {
    "ultrasound_report": ["ultrasonography", "usg", "ultrasound"],
    "hemogram": ["complete blood count", "hemogram", "cbc"],
    "second_opinion": ["second opinion"],
    "second_opinion_note": ["second opinion"],
    "echo_report": ["echocardiography", "2d echocardiography", "echo"],
    "renal_function_test": ["renal function test", "creatinine", "electrolytes"],
    "ecg": ["electrocardiogram", "ecg", "12 lead ecg"],
    "ct_head_report": ["ct scan head", "noncontrast brain ct", "ct scan"],
    "mri_report": ["mri", "brain mri"],
    "endometrial_sampling_report": ["endometrial"],
    "chest_xray": ["x-ray chest", "chest x-ray", "plain x-ray chest"],
    "urine_analysis": ["urine routine", "urine analysis", "urinalysis"],
    "npn_test": ["ns1 antigen", "ns1"],
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _matches(doc_id: str, investigation_item: str) -> bool:
    doc_slug = _slug(doc_id)
    item_lower = investigation_item.lower()
    if doc_slug in _slug(investigation_item) or _slug(investigation_item) in doc_slug:
        return True
    for synonym in SYNONYMS.get(doc_id, []):
        if synonym in item_lower:
            return True
    return False


def _document_status(doc_id: str, present_ids: set[str], documents: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return (is_present_and_valid, is_expired). Falls back to flat presence
    when no richer `documents[]` entry exists for this doc_id."""
    for doc in documents:
        if _slug(str(doc.get("doc_type", ""))) == _slug(doc_id):
            present = bool(doc.get("present"))
            valid = present and doc.get("legible_or_parseable", True) and doc.get("within_validity_window", True)
            expired = present and not doc.get("within_validity_window", True)
            return valid, expired
    return doc_id in present_ids, False


def evaluate_documentation(record: dict[str, Any] | None, documentation_present: list[str],
                            documentation_absent: list[str], documents: list[dict[str, Any]]) -> dict[str, Any]:
    investigations = (record or {}).get("investigations") or {}
    essential = investigations.get("essential") or []
    desirable = investigations.get("desirable") or []
    present_ids = {str(p) for p in documentation_present}
    known_ids = present_ids | {str(a) for a in documentation_absent}

    def _resolve_tier(tier_items: list[str]) -> tuple[list[str], list[str], list[str]]:
        matched_present, matched_missing, matched_expired = [], [], []
        for item in tier_items:
            doc_id = next((d for d in known_ids if _matches(d, item)), None)
            if doc_id is None:
                matched_missing.append(item)
                continue
            valid, expired = _document_status(doc_id, present_ids, documents)
            if expired:
                matched_expired.append(item)
                matched_missing.append(item)
            elif valid:
                matched_present.append(item)
            else:
                matched_missing.append(item)
        return matched_present, matched_missing, matched_expired

    req_present, req_missing, req_expired = _resolve_tier(essential)
    sup_present, sup_missing, _ = _resolve_tier(desirable)

    required_total = len(essential) or 1
    supporting_total = len(desirable) or 1
    dcs = 100 * (0.8 * len(req_present) / required_total + 0.2 * len(sup_present) / supporting_total) \
        if essential or desirable else 100.0

    return {
        "score": round(dcs, 1),
        "required_present": req_present,
        "required_missing": req_missing,
        "expired": req_expired,
        "supporting_present": sup_present,
        "supporting_missing": sup_missing,
    }
