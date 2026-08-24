"""Orchestrates every layer into one decision packet: eligibility gate
first (short-circuits on hard failure), then the criterion engine, then
prior_auth_criterion_scoring.md's score/completeness model, then retrieval
(for citations only -- it never influences the decision), then the audit
write.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import audit, eligibility, scoring_model
    from .chunker import Chunk, build_chunks
    from .corpus_loader import load_corpus
    from .documentation import evaluate_documentation
    from .fact_schema import build_facts
    from .retriever import BM25, retrieve
    from .rule_engine import Criterion, default_criteria_paths, evaluate_applicable_criteria, load_criteria
except ImportError:
    import audit
    import eligibility
    import scoring_model
    from chunker import Chunk, build_chunks
    from corpus_loader import load_corpus
    from documentation import evaluate_documentation
    from fact_schema import build_facts
    from retriever import BM25, retrieve
    from rule_engine import Criterion, default_criteria_paths, evaluate_applicable_criteria, load_criteria

DEFAULT_SOURCE = Path(__file__).parent.parent / "icmr_stw_vol1_rules.json"
DEFAULT_CARE_LEVEL_PATH = Path(__file__).parent / "data" / "care_level_requirements.json"
RULE_TABLE_VERSION = "2.0.0"
PROMPT_VERSION = "pa-decide-offline-v2"
GUIDELINE_VERSION = "ICMR-STW-2019-V1"


class Corpus:
    """Everything loaded once and reused across requests: records, chunks,
    the BM25 index, the criteria table, and the facility-level requirements
    lookup (Pillar 1 only -- not part of the criterion model)."""

    def __init__(self, source: Path = DEFAULT_SOURCE, criteria_paths: list[Path] | None = None,
                 care_level_path: Path = DEFAULT_CARE_LEVEL_PATH):
        raw = load_corpus(source)
        self.records_by_id = {r["id"]: r for r in raw["records"]}
        self.chunks: list[Chunk] = build_chunks(source)
        self.bm25 = BM25(self.chunks)
        self.criteria: list[Criterion] = load_criteria(*(criteria_paths or default_criteria_paths()))
        self.care_level_requirements: dict[str, int] = (
            json.loads(care_level_path.read_text(encoding="utf-8")) if care_level_path.exists() else {}
        )
        procedure_path = Path(__file__).parent / "data" / "procedure_codes.json"
        self.procedure_codes: dict[str, dict[str, Any]] = (
            json.loads(procedure_path.read_text(encoding="utf-8")) if procedure_path.exists() else {}
        )


def _query_text(facts: dict[str, Any]) -> str:
    diagnosis = facts.get("diagnosis") or {}
    service = facts.get("requested_service") or {}
    parts = [diagnosis.get("text"), service.get("text"), facts.get("indication_text")]
    return " ".join(str(p) for p in parts if p)


def _terminal_packet(request_id: str, decision: str, reason: str, eligibility_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "decision": decision,
        "score": None,
        "confidence_score": 100.0 if decision in ("not_covered", "no_authorization_needed") else 0.0,
        "completeness": None,
        "pillars": {"eligibility": eligibility_result},
        "criteria": [],
        "tally": {},
        "clinical_rationale": reason,
        "citations": [],
        "requested_information": [],
        "top_score_drivers": [],
        "guideline_version": GUIDELINE_VERSION,
        "rule_table_version": RULE_TABLE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def adjudicate(request: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    request_id = request.get("request_id", "PA-UNSPECIFIED")
    parsed = build_facts(request)
    facts = parsed["facts"]

    requested_code = (facts.get("requested_service") or {}).get("code")
    required_facility_rank = corpus.care_level_requirements.get(requested_code)

    elig = eligibility.evaluate_eligibility(parsed["eligibility"], facts.get("facility_level_rank"), required_facility_rank)
    if elig["status"] == "not_required":
        packet = _terminal_packet(request_id, "no_authorization_needed", elig["reason"], elig)
        audit.write_record({**packet, "request": request})
        return packet
    if elig["status"] == "not_covered":
        packet = _terminal_packet(request_id, "not_covered", elig["reason"], elig)
        audit.write_record({**packet, "request": request})
        return packet

    verdicts = evaluate_applicable_criteria(corpus.criteria, facts)
    # Build a map of criterion_id -> Criterion for enrichment (polarity, fields)
    crit_map = {c.criterion_id: c for c in corpus.criteria}
    applicable_ids = {v.criterion_id for v in verdicts}
    applicable_criteria = [c for c in corpus.criteria if c.criterion_id in applicable_ids]

    record = None
    if applicable_criteria:
        top_record_id = max({c.source_record_id for c in applicable_criteria},
                             key=lambda rid: sum(1 for c in applicable_criteria if c.source_record_id == rid))
        record = corpus.records_by_id.get(top_record_id)

    doc = evaluate_documentation(record, parsed["documentation_present"], parsed["documentation_absent"], parsed["documents"])

    result = scoring_model.compute_score(verdicts)
    if not verdicts:
        # Empty criteria means this case cannot be evaluated at application level.
        result["completeness"] = 0.0
    gaps = scoring_model.classify_gaps(verdicts)
    has_blocking_gap = bool(gaps["blocking"])
    # `decide()` uses the true null score internally -- score=None is the
    # signal for "nothing evaluable, route to request_more_information"
    # regardless of the band table. The packet's displayed score is always
    # a bounded 0-100 number: null becomes 0 (no criterion has yet been
    # confirmed met), never an undefined value.
    decision = scoring_model.decide(result["score"], result["completeness"], result["reason"], has_blocking_gap)
    decision = scoring_model.enforce_invariants(decision, result["completeness"], has_blocking_gap, result["reason"])
    display_score = result["score"] if result["score"] is not None else 0.0

    drivers = scoring_model.top_score_drivers(verdicts)
    requested_info = scoring_model.requested_information(gaps)

    primary_condition = record["condition"] if record else (verdicts[0].condition if verdicts else None)
    rule_citations = [v.citation for v in verdicts]
    results = retrieve(_query_text(facts), corpus.chunks, corpus.bm25, rule_citations, primary_condition)
    citations = [{"chunk_id": r.chunk.chunk_id, "condition": r.chunk.condition, "page": r.chunk.page,
                  "chunk_type": r.chunk.chunk_type, "quote": r.chunk.text_for_display, "retrieval": r.reason}
                 for r in results]

    rationale = result.get("rationale") or _build_rationale(decision, result, verdicts, requested_info, display_score)

    packet = {
        "request_id": request_id,
        "decision": decision,
        "score": display_score,
        "confidence_score": _confidence_score(decision, display_score, result["completeness"], result["reason"], len(verdicts)),
        "completeness": result["completeness"],
        "score_reason": result["reason"],
        "caps_applied": result["caps_applied"],
        "tally": result["tally"],
        "pillars": {
            "eligibility": elig,
            "documentation": doc,
        },
        "criteria": [],
        "requested_information": requested_info,
        "top_score_drivers": drivers,
        "clinical_rationale": rationale,
        "citations": citations,
        "guideline_version": GUIDELINE_VERSION,
        "rule_table_version": RULE_TABLE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Enrich criteria with evidence and polarity (adverse/favorable) so
    # presentation layers can correctly mark compliance.
    def _collect_fields(node: dict) -> list[str]:
        out: list[str] = []
        if not node:
            return out
        if "field" in node:
            out.append(node["field"])
            return out
        for key in ("all", "any"):
            if key in node:
                for child in node[key]:
                    out.extend(_collect_fields(child))
        return out

    crit_items = []
    for v in verdicts:
        c = crit_map.get(v.criterion_id)
        fields = _collect_fields(c.check) if c is not None else []
        # Adverse polarity detection: HAS-BLED / score-threshold rules indicate
        # that a PASS means an adverse feature. Use heuristic based on field
        # names or rule text.
        adverse = False
        if c is not None:
            text_up = (c.text or "").upper()
            if any("HAS-BLED" in text_up for _ in (1,)):
                adverse = True
            if any("has_bled" in f.lower() or "bleed" in f.lower() or "score" in f.lower() for f in fields):
                adverse = adverse or True

        evidence = {f: facts.get(f) for f in fields}
        crit_items.append({
            "criterion_id": v.criterion_id,
            "type": v.type,
            "weight": v.weight,
            "verdict": v.verdict,
            "confidence": v.confidence,
            "text": v.explanation,
            "missing_fields": v.missing_fields,
            "citation": v.citation,
            "review_status": v.review_status,
            "adverse": adverse,
            "evidence": evidence,
        })

    packet["criteria"] = crit_items
    audit.write_record({**packet, "request": request})
    return packet


def _confidence_score(decision: str, score: float, completeness: float, reason: str | None,
                      criterion_count: int) -> float:
    """Evidence confidence, not a probability of clinical benefit."""
    if criterion_count == 0:
        return 0.0
    if reason in ("gateway_failed", "exclusion_matched"):
        return round(completeness, 1)
    if decision == "approve":
        return round(min(score, completeness), 1)
    return round(completeness, 1)


def _build_rationale(decision: str, result: dict[str, Any], verdicts, requested_info: list[dict[str, Any]],
                      display_score: float) -> str:
    """Every sentence here maps to a computed value already in the packet --
    there is no LLM synthesis step to introduce an unsupported claim."""
    parts = [f"Decision: {decision.replace('_', ' ')}."]
    parts.append(f"Score {display_score}/100, completeness {result['completeness']}%.")
    if not verdicts:
        parts.append("No applicable STW rule was found for the submitted diagnosis and requested service; manual review or a diabetes rule set is required.")
    if result["score"] is None:
        parts.append("No applicable criterion could yet be evaluated to a definite outcome.")
    if result["caps_applied"]:
        parts.append(f"Caps applied: {', '.join(result['caps_applied']).replace('_', ' ')}.")
    failed = [v for v in verdicts if scoring_model.is_genuinely_failed(v.type, v.verdict)]
    if failed:
        parts.append("Not met: " + "; ".join(v.explanation for v in failed[:3]))
    if requested_info:
        parts.append("Outstanding: " + "; ".join(item["item"] for item in requested_info))
    return " ".join(parts)
