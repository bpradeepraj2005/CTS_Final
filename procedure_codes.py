"""Build the procedure-code mapping described in Phase A3 of the RAG spec.

The spec asks for PM-JAY HBP package codes or CPT/HCPCS. Neither is available
offline in this environment as a licensed, verifiable reference table, and
inventing "official" codes would misrepresent fabricated values as regulatory
data. Instead this module assigns a stable **internal** code to every
procedure named anywhere in the corpus (`PA-INT-<SPECIALTY><seq>`), which is
exactly what Phase A3 needs functionally: an incoming request should be able
to arrive as a *code*, not a condition name, and rules/chunks should be
addressable by that code. A payer deploying this for real would swap this
table for a signed PM-JAY HBP 2.0 or CPT/HCPCS crosswalk without touching any
downstream code -- every consumer only ever sees `{system, code, display}`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .corpus_loader import load_records
except ImportError:
    from corpus_loader import load_records

SPECIALTY_ABBR = {
    "Cardiology": "CARD",
    "Nephrology": "NEPH",
    "Urology": "URO",
    "Obstetrics and Gynaecology": "OBG",
    "Pulmonology": "PULM",
    "Neurology": "NEURO",
    "ENT": "ENT",
    "Paediatrics": "PAED",
    "Psychiatry": "PSY",
}

# Procedures that are the *subject* of a record rather than an entry in that
# record's procedures[] array (e.g. hysterectomy is the record; its own
# `routes[]` describes surgical approach, not indication-gated sub-procedures).
# These need to be registered explicitly so a request naming them resolves.
IMPLIED_SUBJECT_PROCEDURES = {
    "stw-v1-obg-hysterectomy": ["Total abdominal hysterectomy", "Vaginal hysterectomy", "Laparoscopic hysterectomy"],
    "stw-v1-obg-dc": ["Dilatation and curettage"],
    "stw-v1-ent-pharyngitis": ["Tonsillectomy"],
    "stw-v1-uro-stones": ["Surgical stone removal (PCNL / URS / ESWL)"],
    "stw-v1-nephro-uti": ["Cystoscopy", "Urodynamic study"],
    # Distinct from "Dialysis (HD or PD)" (AKI's own procedures[] entry,
    # which keeps that name/code). Same physical procedure, different
    # medical-necessity criteria: AKI dialysis is gated by the corpus's
    # 7-item closed indication list, CKD dialysis initiation by eGFR.
    # Reusing one code for both let a request for one silently get
    # evaluated against the other's criteria -- see C-NEPH-CKD-003 /
    # C-NEPH-AKI-001 in rules.yaml.
    "stw-v1-nephro-ckd": ["Dialysis initiation for chronic kidney disease"],
}


def _slug(name: str) -> str:
    return " ".join(name.replace("(", " ").replace(")", " ").split()).strip().lower()


def build_procedure_codes(source: Path) -> dict[str, dict[str, Any]]:
    """Return {normalized_procedure_name: {system, code, display, record_ids, specialty}}."""
    entries: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = {}

    def _register(name: str, record_id: str, specialty: str) -> None:
        key = _slug(name)
        if key not in entries:
            abbr = SPECIALTY_ABBR.get(specialty, "GEN")
            counters[abbr] = counters.get(abbr, 0) + 1
            entries[key] = {
                "system": "PA-INT-1.0",
                "code": f"PA-INT-{abbr}{counters[abbr]:03d}",
                "display": name,
                "specialty": specialty,
                "record_ids": [],
            }
        if record_id not in entries[key]["record_ids"]:
            entries[key]["record_ids"].append(record_id)

    for record in load_records(source):
        record_id = record["id"]
        specialty = record["specialty"]
        for proc in record.get("procedures") or []:
            name = proc.get("name")
            if name:
                _register(name, record_id, specialty)
        for name in IMPLIED_SUBJECT_PROCEDURES.get(record_id, []):
            _register(name, record_id, specialty)

    return entries


def write_procedure_codes(source: Path, destination: Path) -> dict[str, dict[str, Any]]:
    entries = build_procedure_codes(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the internal procedure-code table.")
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent / "icmr_stw_vol1_rules.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "procedure_codes.json")
    args = parser.parse_args()
    entries = write_procedure_codes(args.source, args.output)
    print(f"Wrote {len(entries)} procedure codes to {args.output}")
