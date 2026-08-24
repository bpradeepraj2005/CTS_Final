"""FastAPI service wrapper for the Prior Authorization Triage engine.

This module is deployment glue only. It loads the decision engine once at
start-up and exposes it over HTTP so another backend can call it. Nothing in
`prior_auth/` is modified or reimplemented here: every decision comes from
`decision_engine.adjudicate` and every response body from
`summary.render_json_report`, exactly what the CLI produces.

Local:  uvicorn main:app --reload --port 7860
Docker: the Dockerfile runs the same command on $PORT (Render sets 10000).
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from prior_auth import audit
from prior_auth.decision_engine import (
    GUIDELINE_VERSION,
    PROMPT_VERSION,
    RULE_TABLE_VERSION,
    Corpus,
    adjudicate,
)
# Reused rather than re-derived, so the HTTP path resolves an internal
# procedure code exactly the way `python -m prior_auth.app` does.
from prior_auth.app import _attach_internal_procedure_code
from prior_auth.summary import render_json_report, render_text
from reasoner import analyst, gemini
from reasoner.catalog import Catalog

BASE_DIR = Path(__file__).parent
EXAMPLES_DIR = BASE_DIR / "prior_auth" / "examples"

# Set API_TOKEN in the host's environment to require
# `Authorization: Bearer <token>` on every data endpoint. Left unset, the API
# is open to anyone who knows the URL.
API_TOKEN = os.getenv("API_TOKEN", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


# --------------------------------------------------------------------------
# Audit log location
#
# `audit.write_record` binds its default path at import time, so the only way
# to redirect it without editing the model is to pass the path in explicitly.
# The shim below does that, which lets a deployment point PA_AUDIT_LOG at
# persistent storage (a mounted disk) and keeps a read-only container
# filesystem from failing requests.
# --------------------------------------------------------------------------
_audit_log_path: Path | None = None
_audit_warning: str | None = None
_original_write_record = audit.write_record


def _resolve_audit_log() -> tuple[Path | None, str | None]:
    """First writable candidate wins: $PA_AUDIT_LOG, the package default, temp."""
    candidates: list[Path] = []
    configured = os.getenv("PA_AUDIT_LOG", "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.append(audit.DEFAULT_LOG)
    candidates.append(Path(tempfile.gettempdir()) / "prior_auth_audit_log.jsonl")

    failures: list[str] = []
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8"):
                pass
            return path, None
        except OSError as exc:
            failures.append(f"{path} ({exc.strerror or exc})")
    return None, "no writable location for the audit log: " + "; ".join(failures)


def _audit_write_record(record: dict[str, Any], path: Path | None = None) -> None:
    target = path or _audit_log_path
    if target is None:
        return  # degraded, and reported by /health -- never fail the decision
    _original_write_record(record, target)


audit.write_record = _audit_write_record


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _audit_log_path, _audit_warning
    _audit_log_path, _audit_warning = _resolve_audit_log()
    if _audit_warning:
        print(f"[warn] {_audit_warning}", file=sys.stderr)
    # ~0.15s: 53 records -> 455 chunks -> BM25 index -> 68 criteria. Done once
    # here so the first request does not pay for it.
    app.state.corpus = Corpus()
    print(
        f"[startup] corpus ready: {len(app.state.corpus.records_by_id)} records, "
        f"{len(app.state.corpus.chunks)} chunks, {len(app.state.corpus.criteria)} criteria",
        file=sys.stderr,
    )
    # Header index + per-condition dossiers for the LLM reasoning path. Pure
    # data assembly -- no network call, so a missing GEMINI_API_KEY only fails
    # at request time on /analyze, never at boot.
    app.state.catalog = Catalog()
    print(
        f"[startup] catalog ready: {len(app.state.catalog.conditions)} conditions "
        f"| gemini configured: {gemini.configured()}",
        file=sys.stderr,
    )
    yield


app = FastAPI(
    title="Prior Authorization Triage & Policy Companion",
    description=(
        "Deterministic prior-authorization triage against the ICMR Standard "
        "Treatment Workflows (Vol. 1, 2019). Submit a structured request, get "
        "back a scored, cited, audit-logged decision. No LLM is in the "
        "decision path."
    ),
    version=RULE_TABLE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False, description="Only enforced when API_TOKEN is set.")


def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if not API_TOKEN:
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_corpus(request: Request) -> Corpus:
    corpus = getattr(request.app.state, "corpus", None)
    if corpus is None:
        raise HTTPException(status_code=503, detail="Corpus is still loading.")
    return corpus


# --------------------------------------------------------------------------
# Request schema
#
# Mirrors the schema documented in prior_auth/fact_schema.py. Every field is
# optional and `extra="allow"` keeps unknown keys, so the model -- not this
# wrapper -- decides what is missing. Dumped with exclude_unset=True so the
# payload reaches `adjudicate` exactly as the caller sent it: no defaults
# injected, no nulls dropped.
# --------------------------------------------------------------------------
EXAMPLE_REQUEST: dict[str, Any] = {
    "request_id": "PA-2026-90152",
    "patient": {"age": 54, "sex": "M"},
    "diagnosis": {"icd10": ["N17.9"], "text": "Acute Kidney Injury following viral gastroenteritis"},
    "requested_service": {
        "code": "PA-INT-NEPH001",
        "text": "Outpatient Hemodialysis Evaluation & Procedure",
        "setting": "outpatient",
        "facility_level": "tertiary",
    },
    "clinical_findings": [
        {"parameter": "hyperkalemia", "value": False, "provenance": "structured_report"},
        {"parameter": "creatinine_mg_dl", "value": 2.4, "unit": "mg/dl"},
        {"parameter": "urine_output_ml_per_day", "value": 700, "unit": "ml/day"},
    ],
    "prior_therapies": [{"therapy": "IV normal saline rehydration", "outcome": "clinically improved"}],
    "documentation_present": ["renal_function_test", "ecg"],
    "documentation_absent": [],
    "eligibility": {
        "enrollment_active_on_service_date": True,
        "benefit_covers_service": True,
        "service_not_in_plan_exclusions": True,
        "waiting_period_satisfied": True,
        "annual_or_lifetime_limit_available": True,
        "provider_empanelled": True,
        "prior_auth_actually_required": True,
        "no_duplicate_active_authorization": True,
    },
}


class PriorAuthRequest(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_extra={"example": EXAMPLE_REQUEST})

    request_id: str | None = Field(None, description="Your claim identifier; echoed back as claim_id.")
    patient: dict[str, Any] | None = Field(None, description='e.g. {"age": 54, "sex": "M"}')
    patient_name: str | None = None
    diagnosis: dict[str, Any] | None = Field(None, description='{"icd10": ["N17.9"], "text": "..."}')
    requested_service: dict[str, Any] | None = Field(
        None,
        description='{"code", "text", "setting", "facility_level"} -- code is resolved from text when omitted.',
    )
    clinical_findings: list[dict[str, Any]] | None = Field(
        None,
        description='[{"parameter", "value", "unit", "confidence", "provenance"}]; confidence "low" is treated as absent.',
    )
    prior_therapies: list[dict[str, Any]] | None = None
    documentation_present: list[str] | None = None
    documentation_absent: list[str] | None = None
    documents: list[dict[str, Any]] | None = Field(
        None, description="Richer alternative to documentation_present/absent."
    )
    eligibility: dict[str, Any] | None = Field(
        None, description="Eligibility gate; see prior_auth/eligibility.py."
    )


def _adjudicate(payload: PriorAuthRequest, corpus: Corpus) -> tuple[dict[str, Any], dict[str, Any]]:
    request = payload.model_dump(exclude_unset=True)
    request = _attach_internal_procedure_code(request, corpus.procedure_codes)
    try:
        packet = adjudicate(request, corpus)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        # Almost always a malformed payload (wrong nesting, wrong value type).
        # The traceback still reaches the service logs, so a genuine engine bug
        # is not hidden by the 422.
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request could not be evaluated: {type(exc).__name__}: {exc}",
        ) from exc
    return request, packet


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.post(
    "/adjudicate",
    dependencies=[Depends(require_token)],
    tags=["adjudication"],
    summary="Adjudicate a prior-authorization request",
)
def adjudicate_endpoint(payload: PriorAuthRequest, corpus: Corpus = Depends(get_corpus)) -> dict[str, Any]:
    """The decision report -- identical to what `python -m prior_auth.app` prints.

    `final_decision` is one of APPROVE / REJECT / MORE INFORMATION NEEDED.
    """
    request, packet = _adjudicate(payload, corpus)
    return json.loads(render_json_report(packet, request))


@app.post(
    "/adjudicate/full",
    dependencies=[Depends(require_token)],
    tags=["adjudication"],
    summary="Adjudicate and return the complete decision packet",
)
def adjudicate_full_endpoint(payload: PriorAuthRequest, corpus: Corpus = Depends(get_corpus)) -> dict[str, Any]:
    """Report plus the raw packet (per-criterion verdicts, pillars, score
    drivers, retrieved citations) and the plain-text summary."""
    request, packet = _adjudicate(payload, corpus)
    return {
        "report": json.loads(render_json_report(packet, request)),
        "packet": packet,
        "text_summary": render_text(packet),
    }


class CaseRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case": (
                    "52-year-old man, 30 pack-year smoker, 4 days of worsening breathlessness "
                    "and purulent sputum on a background of known COPD. Two similar admissions "
                    "in the last 12 months. RR 28, SpO2 86% on room air, using accessory muscles. "
                    "ABG pH 7.31, pCO2 58. Chest X-ray no consolidation. Provider requests ICU "
                    "admission with non-invasive ventilation, IV antibiotics and nebulised "
                    "bronchodilators."
                )
            }
        }
    )

    case: str = Field(
        ...,
        min_length=20,
        max_length=20000,
        description="The case in free text: clinical note, referral letter, or authorisation "
        "request. No particular structure is required or expected.",
    )
    temperature: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Reasoning temperature. Defaults to 0.4.",
    )


def get_catalog(request: Request) -> Catalog:
    catalog = getattr(request.app.state, "catalog", None)
    if catalog is None:
        raise HTTPException(status_code=503, detail="Catalog is still loading.")
    return catalog


@app.post(
    "/analyze",
    dependencies=[Depends(require_token)],
    tags=["analysis"],
    summary="Reason over a free-text case against the matched guideline record",
)
def analyze_endpoint(payload: CaseRequest, catalog: Catalog = Depends(get_catalog)) -> dict[str, Any]:
    """Two LLM passes and no rule table.

    The model reads the context-header index for all 53 conditions and routes
    the case, then receives the matched record's COMPLETE dossier -- every
    procedure, drug, investigation, referral rule, care level, exclusion and
    workflow the STW documents -- and reasons over it in prose.

    `analysis` is free-form by design; the structure follows the medicine of
    the case rather than a fixed template. Machine-readable provenance
    (`conditions`, `citations`) sits alongside it, not inside it.
    """
    try:
        result = analyst.analyse(
            payload.case,
            catalog,
            **({"temperature": payload.temperature} if payload.temperature is not None else {}),
        )
    except gemini.LLMUnavailable as exc:
        # Never fabricate an analysis no model produced.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    Report = {
        "conditions": result.conditions,
        "condition_assessment": result.condition_assessment,
        "rules": result.reported_rules(),
        "tally": result.tally(),
        "background_rules": len(result.background_rules()),
        "not_applicable_count": result.not_applicable_count(),
        "flagged": result.flagged,
        "missing_mandatory": result.missing_mandatory(),
        "confidence": result.confidence,
        "overall_explanation": result.overall_explanation,
        "routing": {
            "reasoning": result.routing_reasoning,
            "confidence": result.routing_confidence,
            "matched": result.matched,
        },
        "citations": result.citations,
        "usage": result.usage,
        "guideline_version": GUIDELINE_VERSION,
    }
    return Report


@app.get("/procedure-codes", dependencies=[Depends(require_token)], tags=["reference"])
def procedure_codes_endpoint(
    q: str | None = Query(None, description="Case-insensitive substring filter on name, display, or code."),
    corpus: Corpus = Depends(get_corpus),
) -> dict[str, Any]:
    """The internal PA-INT-* procedure table, for populating `requested_service.code`."""
    items = [{"name": name, **entry} for name, entry in corpus.procedure_codes.items()]
    if q:
        needle = q.lower()
        items = [
            i
            for i in items
            if needle in i["name"].lower()
            or needle in str(i.get("display", "")).lower()
            or needle in str(i.get("code", "")).lower()
        ]
    return {"count": len(items), "items": items}


@app.get("/conditions", dependencies=[Depends(require_token)], tags=["reference"])
def conditions_endpoint(corpus: Corpus = Depends(get_corpus)) -> dict[str, Any]:
    """Every condition the corpus covers, so a caller can check coverage first."""
    items = [
        {
            "id": record["id"],
            "condition": record["condition"],
            "specialty": record.get("specialty"),
            "icd10": record.get("icd10", []),
            "page": (record.get("source") or {}).get("page"),
        }
        for record in corpus.records_by_id.values()
    ]
    items.sort(key=lambda i: (i["specialty"] or "", i["condition"]))
    return {"count": len(items), "items": items}


@app.get("/example-requests", dependencies=[Depends(require_token)], tags=["reference"])
def example_requests_endpoint() -> dict[str, Any]:
    """The bundled sample payloads, ready to POST to /adjudicate."""
    examples: dict[str, Any] = {}
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        try:
            examples[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return {"count": len(examples), "examples": examples}


@app.get("/health", tags=["service"])
def health_endpoint(request: Request) -> dict[str, Any]:
    corpus = getattr(request.app.state, "corpus", None)
    return {
        "status": "ok" if corpus is not None else "loading",
        "corpus": None
        if corpus is None
        else {
            "records": len(corpus.records_by_id),
            "chunks": len(corpus.chunks),
            "criteria": len(corpus.criteria),
            "procedure_codes": len(corpus.procedure_codes),
        },
        "versions": {
            "guideline": GUIDELINE_VERSION,
            "rule_table": RULE_TABLE_VERSION,
            "prompt": PROMPT_VERSION,
        },
        "audit_log": str(_audit_log_path) if _audit_log_path else None,
        "audit_log_warning": _audit_warning,
        "auth_required": bool(API_TOKEN),
        "llm": {
            "configured": gemini.configured(),
            "model": gemini.chat_model(),
            "conditions_indexed": len(catalog.conditions) if (
                catalog := getattr(request.app.state, "catalog", None)
            ) else 0,
        },
    }


_INDEX = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prior Authorization Triage API</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 46rem; margin: 3rem auto; padding: 0 1.25rem; }
  h1 { font-size: 1.5rem; margin-bottom: .25rem; }
  p.sub { opacity: .7; margin-top: 0; }
  table { border-collapse: collapse; width: 100%; margin: 1.5rem 0; }
  td, th { text-align: left; padding: .5rem .6rem;
           border-bottom: 1px solid rgba(128,128,128,.3); vertical-align: top; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
  pre { background: rgba(128,128,128,.12); padding: 1rem; border-radius: 8px; overflow-x: auto; }
</style>
<h1>Prior Authorization Triage &amp; Policy Companion</h1>
<p class="sub">Deterministic triage against the ICMR Standard Treatment Workflows
(Vol.&nbsp;1, 2019). No LLM in the decision path.</p>
<table>
  <tr><th>Endpoint</th><th>What it does</th></tr>
  <tr><td><code>POST /adjudicate</code></td><td>Decision report for one request</td></tr>
  <tr><td><code>POST /analyze</code></td><td><strong>LLM reasoning over a free-text case</strong> &mdash; no rule table</td></tr>
  <tr><td><code>POST /adjudicate/full</code></td><td>Report + full packet + text summary</td></tr>
  <tr><td><code>GET /procedure-codes</code></td><td>Internal PA-INT-* code table</td></tr>
  <tr><td><code>GET /conditions</code></td><td>Conditions covered by the corpus</td></tr>
  <tr><td><code>GET /example-requests</code></td><td>Sample payloads</td></tr>
  <tr><td><code>GET /health</code></td><td>Readiness and versions</td></tr>
</table>
<p>Interactive docs: <a href="docs">/docs</a> &middot;
   OpenAPI: <a href="openapi.json">/openapi.json</a></p>
"""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def index() -> str:
    return _INDEX
