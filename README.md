# Prior Authorization Triage & Policy Companion

A FastAPI service that checks prior-authorization cases against the ICMR Standard Treatment
Workflows (Vol. 1, 2019). The corpus is validated and split into 455 citation-preserving
chunks, each stamped with a context header naming its condition, specialty, ICD-10 codes,
section type and source page.

Two paths, with different trade-offs:

- **[`POST /analyze`](#post-analyze--llm-reasoning)** — the LLM route, and the one to use for
  real cases. It routes a free-text case by context header, then extracts the requirements
  the knowledge base states for that condition and checks the patient against each one,
  returning a pass/fail finding per rule with the clinical consequence of every gap. No rule
  table is involved and no clinical rule is hardcoded — the rules come from the KB at request
  time. Requires `GEMINI_API_KEY`.
- **[`POST /adjudicate`](#post-adjudicate--deterministic-rules)** — the original deterministic
  engine: 68 hand-written criteria, scored and banded, no API key needed. Fully reproducible,
  but it covers only 24 of the 53 conditions, so it can only pend on the other 29.

Both cite every claim to a page, and both write to an append-only audit log.

## API

Base URL once deployed: `https://<your-service>.onrender.com`

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/analyze` | **LLM reasoning over a free-text case — no rule table** |
| `POST` | `/adjudicate` | Decision report for one request |
| `POST` | `/adjudicate/full` | Report + full decision packet + plain-text summary |
| `GET` | `/procedure-codes` | Internal `PA-INT-*` code table (`?q=` filters) |
| `GET` | `/conditions` | Conditions the corpus covers |
| `GET` | `/example-requests` | Bundled sample payloads |
| `GET` | `/health` | Readiness, corpus size, rule-table version |
| `GET` | `/docs` | Interactive OpenAPI docs |

## `POST /analyze` — LLM reasoning

The path to use for real cases. No rule table is involved, and the input is free text.

Why: the corpus holds 53 conditions whose treatments, procedures, drugs, workflows and
frequency periods combine far past what a rule table can enumerate — 29 of those 53
conditions have **zero** executable rules, so the deterministic path can only pend on them.
The LLM reads the guideline text itself instead.

**Two passes** ([reasoner/analyst.py](reasoner/analyst.py)):

1. **Route** — the model reads the context-header index for all 53 conditions
   (`record_id | Condition | Specialty | ICD-10 | page | covers: …`, stamped onto every chunk
   by [chunker.py:112](prior_auth/chunker.py#L112)) and decides which record(s) the case concerns.
2. **Check** — it receives those records' **complete** knowledge-base content, extracts every
   requirement the guideline actually states — thresholds, investigations, indications,
   contraindications, procedures, drugs with doses and frequency periods, care levels,
   referral and admission triggers — and checks the patient input against each one.

Nothing in the code enumerates a clinical rule. The rules come out of the knowledge base at
request time, so a richly specified condition yields many findings and a sparse one yields
few, with no code change. The largest dossier is ~1,800 tokens, so nothing is top-k sampled.

```json
{"case": "34-year-old woman, para 1. Ultrasound shows a 10-week-size uterus with a small intramural fibroid. Mild spotting only. No prior medical therapy. No second gynaecologist opinion on file. Provider requests total abdominal hysterectomy."}
```

Response — one finding per rule the KB states:

```json
{
  "conditions": [{"condition": "Uterine Fibroids and Polyps", "page": 40}, {"condition": "Hysterectomy for Benign Gynaecological Conditions", "page": 38}],
  "condition_assessment": "…what the case is, in the model's words…",
  "tally": {"FAIL": 4, "PASS": 1, "NOT_IN_INPUT": 3},
  "findings": [
    {
      "requirement": "Hysterectomy is explicitly non-indicated for fibroids that are small (<5 cm) or with uterine size less than 12 weeks",
      "kb_section": "exclusion",
      "page": 38,
      "status": "FAIL",
      "patient_evidence": "10-week-size uterus with a small intramural fibroid, mild spotting only",
      "explanation": "Uterine size 10 weeks is below the 12-week threshold, placing this case on the explicit non-indication list",
      "why_it_matters": "The case meets explicit exclusion criteria, making hysterectomy inappropriate"
    }
  ],
  "summary": "…overall reading, prose…",
  "routing": {"reasoning": "…", "confidence": "high", "matched": true},
  "citations": [...]
}
```

**Status values** — `PASS` (input satisfies it, with the patient value quoted), `FAIL` (input
contradicts it or breaches a threshold), `NOT_IN_INPUT` (the KB requires it, the case is
silent — missing evidence, not a failure), `NOT_RELEVANT` (stated but inapplicable to this
patient).

`why_it_matters` is the explainable-AI field: for every `FAIL` and `NOT_IN_INPUT` the model
explains the clinical consequence — what could go wrong, what decision it blocks, what must
be obtained first.

When no record matches, `matched` is `false` and the model explains in its own words what
the case appears to be and why the corpus does not cover it. There is no canned message.

Requires `GEMINI_API_KEY`. Without it `/analyze` returns **503** rather than fabricating an
analysis no model produced. Typical call: ~2–4s routing, ~10–17s reasoning.

## `POST /adjudicate` — deterministic rules

Retained for the 24 conditions that have executable rules, and for cases where a fixed,
reproducible verdict matters more than coverage.

### Request body

The request format **is** the extracted-facts schema — callers submit the structured
artifact, not a free-text note. Full field documentation lives in
[prior_auth/fact_schema.py](prior_auth/fact_schema.py).

```json
{
  "request_id": "PA-2026-90152",
  "patient": {"age": 54, "sex": "M"},
  "diagnosis": {"icd10": ["N17.9"], "text": "Acute Kidney Injury"},
  "requested_service": {
    "code": "PA-INT-NEPH001",
    "text": "Outpatient Hemodialysis Evaluation & Procedure",
    "setting": "outpatient",
    "facility_level": "tertiary"
  },
  "clinical_findings": [
    {"parameter": "hyperkalemia", "value": false, "provenance": "structured_report"},
    {"parameter": "creatinine_mg_dl", "value": 2.4, "unit": "mg/dl"}
  ],
  "documentation_present": ["renal_function_test", "ecg"],
  "eligibility": {"enrollment_active_on_service_date": true, "benefit_covers_service": true}
}
```

`requested_service.code` is optional — it is resolved from `requested_service.text` against
the procedure table when omitted. A `clinical_findings` entry with `"confidence": "low"` is
treated as absent everywhere downstream.

### Response

```json
{
  "claim_id": "PA-2026-90152",
  "icd_code_detected": "N17.9",
  "condition_detected": "Acute Kidney Injury",
  "requested_procedure": "Outpatient Hemodialysis Evaluation & Procedure",
  "rule_checks": [
    {
      "rule_name": "...",
      "guideline_requirement": "...",
      "extracted_patient_value": "...",
      "compliance_flag": "COMPLIANT | NON_COMPLIANT | MISSING_DOCUMENTATION",
      "stw_citation": "Acute Kidney Injury STW, Page 41"
    }
  ],
  "missing_items": [],
  "flagged_non_compliant_items": [],
  "required_documents_for_resubmission": [],
  "ai_clinical_explanation": "...",
  "confidence_score": 100.0,
  "final_decision": "APPROVE | REJECT | MORE INFORMATION NEEDED"
}
```

`POST /adjudicate/full` returns the same object under `report`, plus `packet` (per-criterion
verdicts, eligibility and documentation pillars, score drivers, retrieved citations) and
`text_summary`.

## Calling it from another backend

```python
import httpx, os

API_URL = "https://<your-service>.onrender.com"
client = httpx.AsyncClient(
    headers={"Authorization": f"Bearer {os.environ['PA_API_TOKEN']}"},
    timeout=httpx.Timeout(90.0, connect=10.0),
)

@app.post("/check-authorization")
async def check_authorization(request: dict):
    response = await client.post(f"{API_URL}/adjudicate", json=request)
    response.raise_for_status()
    return response.json()
```

Reuse one client for the process, and keep the read timeout generous: a free-tier instance
sleeps after ~15 minutes idle and takes about a minute to wake. A runnable version is in
[client_example.py](client_example.py).

Or with curl:

```bash
curl -X POST "$API_URL/adjudicate" \
  -H "Content-Type: application/json" \
  -d @prior_auth/examples/robert_martinez.json
```

## Deploying to Render

The repo carries a [render.yaml](render.yaml) Blueprint, so the service configures itself.

1. Push this repo to your own GitHub account.
2. In Render: **New → Blueprint**, pick the repo, and apply. Render reads `render.yaml`,
   builds the [Dockerfile](Dockerfile), and creates a free web service.
3. Watch the deploy log. When it prints
   `[startup] corpus ready: 53 records, 455 chunks, 68 criteria`, hit `/health`.

The container binds `$PORT` (Render sets `10000`) and falls back to `7860` locally, so the
same image runs in both places.

### Free-tier behaviour

- Sleeps after 15 minutes without traffic; the next request waits ~1 minute for wake-up.
- 750 instance-hours per month covers one always-on service.
- **No persistent disk.** The audit log is wiped on every restart and deploy — `render.yaml`
  points `PA_AUDIT_LOG` at `/tmp/audit_log.jsonl` to make that explicit. Attach a paid disk
  and repoint it there if you need decisions to survive restarts.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | **Required for `/analyze`.** Set it as a Render secret — never commit it. `/adjudicate` does not need it. |
| `GEMINI_CHAT_MODEL` | `models/gemini-3.6-flash` | Reasoning model. |
| `API_TOKEN` | unset | When set, every data endpoint requires `Authorization: Bearer <token>`. The service URL is public and guessable, so set this. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Irrelevant for server-to-server calls. |
| `PA_AUDIT_LOG` | `prior_auth/data/audit_log.jsonl` | Where the append-only audit log is written. |

Any Docker host works the same way — the image has nothing Render-specific in it.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

Then open <http://localhost:7860/docs>.

The CLI still works unchanged:

```bash
python -m prior_auth.app prior_auth/examples/robert_martinez.json   # from a JSON file
python -m prior_auth.app                                            # paste a request interactively
```

## Scope

The current implementation covers corpus loading, contextual chunking, lexical retrieval,
citations, and deterministic triage. Embedding retrieval, a versioned editable rule table,
and a reviewer UI are the next production phases described in
[prior_auth_rag_spec.md](prior_auth_rag_spec.md).

The `rag/` directory is a separate Gemini-backed retrieval experiment. It is not imported by
the API and is excluded from the container image.
