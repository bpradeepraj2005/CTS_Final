"""Structural criteria extraction over the full corpus, for
prior_auth_criterion_scoring.md's five-type model.

`data/rules.yaml` hand-curates ~10 conditions with controlled vocabulary
fields (e.g. `indication_category`), which is what lets those criteria be
`mandatory`/`exclusion` -- a closed enum comparison is exact, so a failure
is trustworthy enough to cap or zero the score.

The other ~43 records in the corpus were not hand-modeled into a controlled
vocabulary. For those, this module extracts two safe things automatically:

1. `care_level_requirements.json` -- every `procedures[].required_level` in
   the corpus (all 53 records), as a procedure-code -> minimum-facility-rank
   lookup. This is NOT a criterion in the new model at all: facility-level
   gating belongs to Pillar 1 (eligibility.py's
   `facility_level_authorized_for_service` check), not the clinical
   criteria table.
2. `type: supporting` criteria matched against a free-text `indication_text`
   fact via substring search -- exactly the "fuzzy string matching on free
   text" the spec warns is what you get without a code mapping. These are
   deliberately never emitted as `type: exclusion` or `type: mandatory`:
   EXCLUSION now has hard terminal-zero power and MANDATORY caps the score,
   and a keyword hit is not trustworthy enough to carry either. As
   `supporting`, a false match can only nudge the score down by staying in
   the denominator at zero credit -- it can never single-handedly deny or
   cap.

Run this to regenerate `data/rules_generated.yaml` and
`data/care_level_requirements.json` whenever the corpus or the curated
criteria set changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .corpus_loader import load_records
    from .facility import rank_from_text
except ImportError:
    from corpus_loader import load_records
    from facility import rank_from_text

CURATED_RECORD_IDS = {
    "stw-v1-obg-hysterectomy", "stw-v1-ent-pharyngitis", "stw-v1-uro-stones",
    "stw-v1-nephro-ckd", "stw-v1-neuro-stroke", "stw-v1-psych-psychosis",
    "stw-v1-paed-dengue", "stw-v1-paed-fever", "stw-v1-cardio-heart-failure",
    "stw-v1-uro-aur", "stw-v1-nephro-aki",
}

EXCLUSION_FIELDS = (
    "exclusions", "explicit_non_indications", "contraindications",
    "absolute_contraindications_thrombolysis", "donts",
)


def _procedure_code_lookup(codes_path: Path) -> dict[str, str]:
    raw = json.loads(codes_path.read_text(encoding="utf-8"))
    return {v["display"]: v["code"] for v in raw.values()}


def build_care_level_requirements(records: list[dict[str, Any]], code_by_name: dict[str, str]) -> dict[str, int]:
    """procedure_code -> minimum facility rank (0=primary, 1=district/secondary, 2=tertiary)."""
    out: dict[str, int] = {}
    for record in records:
        for proc in record.get("procedures") or []:
            name = proc.get("name")
            code = code_by_name.get(name or "")
            rank = rank_from_text(proc.get("required_level"))
            if code and rank is not None:
                out[code] = rank if code not in out else min(out[code], rank)
    return out


def _score_threshold_criteria(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort single-clause extraction from severity_scores[].thresholds[].rule.

    Skipped when the rule text encodes more than one numeric clause (e.g. a
    sex-conditional threshold) -- a regex cannot safely disambiguate which
    clause governs which population, and a wrong silent guess there is
    worse than no criterion at all.
    """
    pattern = re.compile(
        r"\b(greater than or equal to|more than or equal to|at least|greater than|more than|over)\s+(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    out: list[dict[str, Any]] = []
    for record in records:
        if record["id"] in CURATED_RECORD_IDS:
            continue
        for score in record.get("severity_scores") or []:
            score_name = score.get("name")
            if not score_name:
                continue
            field = re.sub(r"[^a-z0-9]+", "_", score_name.lower()).strip("_") + "_score"
            for threshold in score.get("thresholds") or []:
                rule_text = threshold.get("rule") or ""
                matches = pattern.findall(rule_text)
                if len(matches) != 1:
                    continue
                comparator_phrase, value_str = matches[0]
                op = "gte" if "equal" in comparator_phrase.lower() else "gt"
                criterion_id = f"RG-SCORE-{len(out) + 1:04d}"
                out.append({
                    "criterion_id": criterion_id,
                    "source_record_id": record["id"],
                    "source_page": record["source"]["page"],
                    "condition": record["condition"],
                    "applies_to": {"procedure_codes": [], "icd10_any": list(record.get("icd10") or [])},
                    "type": "supporting",
                    "evaluator": "deterministic",
                    "weight": 3,
                    "check": {"all": [{"field": field, "op": op, "value": float(value_str)}]},
                    "text": f"{score_name}: {rule_text} -> {threshold.get('outcome') or threshold.get('interpretation') or threshold.get('action') or ''}".strip(),
                    "version": "1.0",
                    "review_status": "unverified",
                })
    return out


def _keyword_criteria(records: list[dict[str, Any]], code_by_name: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        if record["id"] in CURATED_RECORD_IDS:
            continue

        exclusion_phrases: list[str] = []
        for field in EXCLUSION_FIELDS:
            value = record.get(field)
            if isinstance(value, list):
                exclusion_phrases.extend(str(v) for v in value if isinstance(v, (str, int, float)))
        if exclusion_phrases:
            criterion_id = f"RG-EXCL-{len(out) + 1:04d}"
            out.append({
                "criterion_id": criterion_id,
                "source_record_id": record["id"],
                "source_page": record["source"]["page"],
                "condition": record["condition"],
                "applies_to": {"procedure_codes": [], "icd10_any": list(record.get("icd10") or [])},
                "type": "supporting",  # never `exclusion`: keyword match only, no terminal power
                "evaluator": "deterministic",
                "weight": 3,
                "check": {"all": [{"field": "indication_text", "op": "not_contains_any", "value": exclusion_phrases}]},
                "text": "Not a listed non-indication: " + "; ".join(exclusion_phrases[:6]),
                "version": "1.0",
                "review_status": "unverified",
            })

        for proc in record.get("procedures") or []:
            indications = proc.get("indications")
            if not indications:
                continue
            code = code_by_name.get(proc.get("name", ""))
            criterion_id = f"RG-INDIC-{len(out) + 1:04d}"
            out.append({
                "criterion_id": criterion_id,
                "source_record_id": record["id"],
                "source_page": record["source"]["page"],
                "condition": record["condition"],
                "applies_to": {"procedure_codes": [code] if code else [], "icd10_any": list(record.get("icd10") or [])},
                "type": "supporting",  # never `mandatory`: keyword match only, cannot cap the score
                "evaluator": "deterministic",
                "weight": 3,
                "check": {"all": [{"field": "indication_text", "op": "contains_any", "value": [str(i) for i in indications]}]},
                "text": "Matches a listed indication: " + "; ".join(str(i) for i in indications[:6]),
                "version": "1.0",
                "review_status": "unverified",
            })
    return out


def build_generated_criteria(source: Path, procedure_codes_path: Path) -> list[dict[str, Any]]:
    records = load_records(source)
    code_by_name = _procedure_code_lookup(procedure_codes_path)
    criteria = _score_threshold_criteria(records)
    criteria += _keyword_criteria(records, code_by_name)
    return criteria


def write_generated_files(source: Path, procedure_codes_path: Path, criteria_dest: Path, care_level_dest: Path) -> None:
    import yaml

    records = load_records(source)
    code_by_name = _procedure_code_lookup(procedure_codes_path)

    criteria = build_generated_criteria(source, procedure_codes_path)
    header = (
        "# AUTO-GENERATED by rule_extractor.py -- do not hand-edit.\n"
        "# Keyword-matched (type: supporting) criteria for the records not covered by\n"
        "# the hand-curated data/rules.yaml. Deliberately never `exclusion` or\n"
        "# `mandatory`: a free-text keyword hit can nudge the score, never deny or cap it.\n\n"
    )
    criteria_dest.parent.mkdir(parents=True, exist_ok=True)
    criteria_dest.write_text(header + yaml.safe_dump({"criteria": criteria}, sort_keys=False, allow_unicode=True), encoding="utf-8")

    care_level = build_care_level_requirements(records, code_by_name)
    care_level_dest.write_text(json.dumps(care_level, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-generate structural criteria and facility requirements from the corpus.")
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent / "icmr_stw_vol1_rules.json")
    parser.add_argument("--procedure-codes", type=Path, default=Path(__file__).parent / "data" / "procedure_codes.json")
    parser.add_argument("--criteria-output", type=Path, default=Path(__file__).parent / "data" / "rules_generated.yaml")
    parser.add_argument("--care-level-output", type=Path, default=Path(__file__).parent / "data" / "care_level_requirements.json")
    args = parser.parse_args()
    write_generated_files(args.source, args.procedure_codes, args.criteria_output, args.care_level_output)
    print(f"Wrote generated criteria to {args.criteria_output} and facility requirements to {args.care_level_output}")
