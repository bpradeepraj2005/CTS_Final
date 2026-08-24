# Prior Authorization Triage & Policy Companion

This prototype runs against `icmr_stw_vol1_rules.json` without an API key. It validates the corpus, creates 455 citation-preserving semantic chunks, performs local lexical retrieval, and produces conservative deterministic triage packets.

## Run a query

From the workspace root:

```powershell
python -m prior_auth.app --query "When is dialysis indicated in acute kidney injury?"
```

## Paste text interactively

Run without `--query` or `--request`:

```powershell
python -m prior_auth.app
```

Paste your text, then type `END` on its own line and press Enter.

## Triage a request

Create a JSON file such as `request.json`:

```json
{
  "condition": "Acute Kidney Injury",
  "requested_service": "dialysis",
  "clinical_findings": ["hyperkalemia", "reduced urine output"],
  "clinical_notes": "Urgent renal review requested."
}
```

Run:

```powershell
python -m prior_auth.app --request request.json
```

Decisions are intentionally conservative: `deny` requires a listed exclusion match, missing facts produce `request_more_information`, and requests without a human-verified executable approval rule produce `pend`. This prevents missing information or an LLM from becoming a fabricated denial or approval.

The current implementation covers corpus loading, contextual chunking, lexical retrieval, citations, and deterministic triage. Embedding retrieval, a versioned editable rule table, and a reviewer UI are the next production phases described in `prior_auth_rag_spec.md`.