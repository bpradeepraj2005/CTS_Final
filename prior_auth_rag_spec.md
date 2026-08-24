# Prior Authorization Triage & Policy Companion — RAG Architecture Specification

**Purpose of this document:** a complete, implementable build spec for a hybrid rules + retrieval system that adjudicates prior authorization requests against clinical guidelines. Hand this to an implementation agent as-is. Section 12 contains the agent prompt.

**Input corpus:** `icmr_stw_vol1_rules.json` — 53 structured clinical rule records extracted from ICMR Standard Treatment Workflows of India (2019, Vol 1). Optionally extend with Vol 2/3 and CMS Medicare Coverage Database.

---

## 0. Architectural principle (read this before anything else)

**Do not build a pure vector RAG system.** The problem demands a *configurable rule set* and *shown reasoning*. An LLM reading retrieved text and emitting "approve" is a black box with citations attached — it will be non-deterministic across identical inputs, which is disqualifying for a payer workflow.

Build **two cooperating layers**:

| Layer | Contains | Executes as | Answers |
|---|---|---|---|
| **Deterministic rule engine** | Numeric thresholds, closed indication lists, contraindication lists, care-level gates | Code (SQL / Python predicates) | "Does this request satisfy the coded criteria?" |
| **Retrieval layer (RAG)** | Full guideline narrative, context, edge cases, rationale | Vector + lexical search → rerank → LLM | "What does the guideline say, and what's missing?" |

The rule engine produces the **decision**. The RAG layer produces the **explanation, the missing-information list, and handles cases with no coded rule**. The LLM never overrides a deterministic DENY; it can only escalate APPROVE → PEND when it finds a contraindication the rules missed.

This is what makes the system auditable, reproducible, and genuinely "configurable."

---

## 1. Phase overview

```
PHASE A  Corpus preparation      → normalized records + provenance
PHASE B  Rule extraction         → executable rule table (deterministic layer)
PHASE C  Chunking                → semantic nodes with contextual headers
PHASE D  Enrichment              → hypothetical questions, entity tags, code mapping
PHASE E  Embedding & indexing    → dense + sparse hybrid index
PHASE F  Query understanding     → request → structured clinical facts
PHASE G  Retrieval               → filtered hybrid search + fusion
PHASE H  Reranking               → cross-encoder + diversity
PHASE I  Decisioning             → rule engine + LLM synthesis
PHASE J  Output & audit          → decision packet with citations
PHASE K  Evaluation              → retrieval + decision metrics
PHASE L  Operations              → caching, versioning, drift
```

---

## PHASE A — Corpus preparation

### A1. Load and validate
Load `icmr_stw_vol1_rules.json`. Validate every record has: `id`, `condition`, `specialty`, `source.page`. Fail loudly on missing provenance — a chunk without a page number cannot be cited and must not enter the index.

### A2. Normalize the vocabulary
The source text says "greater than", "more than", "over", ">" interchangeably. Build a normalization pass:

- Numeric comparators → canonical form: `{"op": "gt", "value": 1, "unit": null}`
- Units → SI where unambiguous (mg/dl, ml/min, mmHg, cm, weeks)
- Drug names → lowercase, strip salt forms into a separate field (`amoxycillin clavulanic acid` → `amoxicillin` + `clavulanic acid`)
- Spelling variants → canonical (`amoxycillin`/`amoxicillin`, `haemorrhage`/`hemorrhage`)

Store both `raw` and `normalized` — never destroy the original wording, because citations must quote the source faithfully.

### A3. Attach code mappings
Each record has ICD-10. Additionally map procedures to a procedure vocabulary. For an Indian payer context use **PM-JAY HBP (Health Benefit Package) package codes**; for a US/CMS framing use CPT/HCPCS. Store as:

```json
"procedure_codes": [
  {"system": "HBP-2.0", "code": "MG010", "display": "Total abdominal hysterectomy"}
]
```

This mapping is the single highest-leverage piece of work in the whole build — an incoming PA request arrives as a *code*, not a condition name. Without it, you're doing fuzzy string matching on free text and precision collapses. Build it manually for the ~40 procedures in the corpus; that is a few hours of work and it is worth more than any retrieval optimization.

### A4. Provenance record
For every record, store `{doc_id, doc_title, version, page, extraction_date, extraction_method, review_status}`. Set `review_status: "unverified"` initially. Numeric thresholds must be flipped to `"human_verified"` before production use — extraction from infographic PDFs is error-prone precisely on numbers.

---

## PHASE B — Rule extraction (the deterministic layer)

This is where the system gets its spine. Walk the corpus and lift every **executable** rule into a rule table.

### B1. Rule taxonomy

| Rule type | Example from corpus | Executable form |
|---|---|---|
| `numeric_threshold` | Fibroid ≥14 weeks uterine size for hysterectomy | `uterine_size_weeks >= 14` |
| `score_threshold` | OAC if CHA2DS2-VASc >1 (men), >2 (women) | `score > 1 if sex=M else > 2` |
| `closed_indication_list` | Dialysis in AKI — 7 listed indications | `any(indication in [...])` |
| `explicit_exclusion` | Hysterectomy NOT for cervicitis, white discharge, etc. | `if indication in DENY_LIST → deny` |
| `contraindication` | Carboprost contraindicated in asthma | `if asthma → flag` |
| `prior_therapy_required` | CRT requires ≥3 months optimal medical therapy | `omt_duration_months >= 3` |
| `care_level_gate` | RFA only at tertiary centre | `facility_level == "tertiary"` |
| `temporal_window` | tPA 0–4.5h; thrombectomy 4.5–24h | `4.5 <= hours_from_onset <= 24` |
| `frequency_threshold` | Tonsillectomy: 7/1yr or 5/yr×2 or 3/yr×3 | disjunction of counts |
| `sequencing_rule` | MET trial 4 weeks before stone surgery | `met_trial_completed == true` |
| `documentation_requirement` | Second opinion if age <40 for hysterectomy | `if age < 40 → require second_opinion_doc` |

### B2. Rule schema

```json
{
  "rule_id": "R-OBG-HYST-002",
  "source_record_id": "stw-v1-obg-hysterectomy",
  "source_page": 38,
  "applies_to": {
    "procedure_codes": ["HBP-MG010"],
    "icd10_any": ["D25"]
  },
  "rule_type": "numeric_threshold",
  "logic": {
    "all": [
      {"field": "uterine_size_weeks", "op": "gte", "value": 14}
    ]
  },
  "outcome_if_true": "criteria_met",
  "outcome_if_false": "criteria_not_met",
  "outcome_if_unknown": "request_more_information",
  "required_fields": ["uterine_size_weeks"],
  "citation_text": "Asymptomatic fibroids greater than or equal to 14 weeks uterine size",
  "severity": "hard",
  "version": "1.0",
  "review_status": "unverified"
}
```

**`outcome_if_unknown` is the most important field.** It is what generates the RFI (request-more-information) branch, which the problem statement explicitly requires. Missing data must never silently become a denial.

`severity: "hard"` = rules the LLM cannot override. `"soft"` = advisory, LLM may weigh.

### B3. Rule storage
Postgres (or SQLite for a prototype). Tables: `rules`, `rule_versions`, `rule_audit_log`. Every rule edit writes an audit row. Expose a config UI or YAML file so rules are editable without code changes — **this is literally what "configurable rule set" means in the problem statement, and reviewers will look for it.**

### B4. Expected yield
From this corpus expect roughly:
- ~120–160 executable rules
- Highest density: Hysterectomy, Uterine Fibroids, HMB, Renal Stones, Pharyngitis (Centor + tonsillectomy), CKD (eGFR gates), Stroke (time windows), Psychosis (clozapine gate), Dengue (platelet threshold), Fever in Children (investigation tiers)

Those ten records alone will carry most of your demo.

---

## PHASE C — Chunking

### C1. Chunk at semantic nodes, never by character count

Fixed-size chunking destroys this corpus. A 512-token window will slice the middle of an indication list, and a retrieved half-list is worse than no retrieval — it produces confidently wrong denials.

**Rule:** one chunk per second-level semantic node. Walk the record tree and emit a chunk for each of:

| Node | chunk_type |
|---|---|
| `clinical_criteria` (symptoms, signs, red flags) | `criteria` |
| each entry in `severity_scores[]` | `score` |
| `investigations` (whole tiered block) | `investigation` |
| each entry in `procedures[]` | `procedure` |
| `care_level` (per level: phc / district / tertiary) | `care_level` |
| `referral_criteria` | `referral` |
| `admission_criteria` | `admission` |
| `discharge_criteria` | `discharge` |
| `drugs[]` (group into 1–3 chunks, not one per drug) | `drug` |
| `contraindications` / `explicit_non_indications` | `exclusion` |
| `prior_auth_notes` | `payer_note` |
| record header (condition + definition + ICD) | `overview` |

Expected: **450–600 chunks** from Vol 1.

### C2. Never split a list

If a node contains a list (indications, contraindications, red flags), it is **atomic**. If it exceeds your token budget, do not split — instead emit the full list and accept the longer chunk. Completeness beats size uniformity here. In practice no node in this corpus exceeds ~700 tokens.

### C3. Contextual chunk headers (high impact, low effort)

Prepend a generated context line to every chunk before embedding. This is the single highest-ROI retrieval improvement available and typically cuts failed retrievals by a third or more.

Format:

```
[{condition} | {specialty} | ICD-10 {codes} | {chunk_type} | ICMR STW 2019 Vol 1 p.{page}]
{rendered prose}
```

Example:

```
[Hysterectomy for Benign Gynaecological Conditions | Obstetrics and Gynaecology |
 procedure | ICMR STW 2019 Vol 1 p.38]
Conditions for which hysterectomy should NOT be performed: white discharge per
vaginum; cervicitis; non-specific abdominal or pelvic pain; minor degree of utero
vaginal prolapse; fibroids which are small (less than 5 cm) or asymptomatic (less
than 12 weeks size uterus); simple ovarian cyst 5 cm or less.
```

Without the header, that chunk contains the word "hysterectomy" zero times in its body and will never be retrieved for a hysterectomy query. This failure mode is pervasive in structured-data chunking and the header fixes it.

### C4. Render JSON to prose

Do not embed raw JSON. Write a deterministic renderer per `chunk_type` — a template function, not an LLM call (LLM rendering introduces drift and cost for no benefit). Keep the original JSON node in the payload as `structured_data` so the rule engine and the citation formatter can both use it.

### C5. Parent-document linkage

Every chunk stores `parent_record_id`. At retrieval time you search over chunks but may expand to the full parent record when the LLM needs surrounding context. This gives small-chunk precision with large-chunk context — retrieve narrow, read wide.

### C6. Chunk schema

```json
{
  "chunk_id": "stw-v1-obg-hysterectomy::exclusion::0",
  "parent_record_id": "stw-v1-obg-hysterectomy",
  "chunk_type": "exclusion",
  "text": "<contextual header + rendered prose>",
  "text_for_display": "<prose without header>",
  "structured_data": { },
  "metadata": {
    "condition": "Hysterectomy for Benign Gynaecological Conditions",
    "specialty": "Obstetrics and Gynaecology",
    "icd10": ["D25", "N84"],
    "procedure_codes": ["HBP-MG010"],
    "page": 38,
    "doc_version": "ICMR-STW-2019-V1",
    "care_levels": ["district", "tertiary"],
    "has_numeric_threshold": true,
    "linked_rule_ids": ["R-OBG-HYST-002", "R-OBG-HYST-007"],
    "review_status": "unverified"
  }
}
```

`linked_rule_ids` closes the loop between the two layers: when a rule fires, you retrieve its supporting chunk for the citation; when a chunk is retrieved, you know which rules govern it.

---

## PHASE D — Enrichment

### D1. Hypothetical question generation

For each chunk, generate 3–5 questions it would answer. Store them as additional embedded vectors pointing to the same chunk. This bridges the vocabulary gap between how guidelines are *written* ("Asymptomatic fibroids greater than or equal to 14 weeks uterine size") and how requests are *phrased* ("Is a hysterectomy covered for a 12-week fibroid?").

Do this once, offline, with an LLM. It is a one-time cost and it measurably lifts recall on paraphrased queries.

### D2. Entity extraction

Tag each chunk with normalized entities: drugs, procedures, lab tests, imaging modalities, anatomical sites. Store as arrays in metadata. Enables precise filtering ("show me every chunk mentioning MRI").

### D3. Threshold index

Build a separate flat lookup of every numeric threshold in the corpus:

```json
{"condition": "Uterine Fibroids", "parameter": "uterine_size", "op": "gte",
 "value": 12, "unit": "weeks", "consequence": "consider surgical options",
 "page": 40, "chunk_id": "..."}
```

Query this directly for numeric questions instead of hoping semantic search surfaces the right number. Semantic embeddings are notoriously poor at numeric discrimination — "greater than 12 weeks" and "greater than 14 weeks" embed almost identically. Do not rely on vectors for numbers.

---

## PHASE E — Embedding & indexing

### E1. Embedding model

Use a **hybrid of dense + sparse**, not dense alone.

- **Dense:** `BAAI/bge-m3` (multilingual, long context, strong on technical text) or `intfloat/e5-large-v2`. If you want biomedical specialization, `MedCPT` or `S-PubMedBert-MS-MARCO`. Benchmark on your own eval set before committing — general models often beat biomedical ones on guideline text, which is closer to policy prose than to research abstracts.
- **Sparse:** BM25 or SPLADE.

Sparse matters here more than usual. This corpus is dense with exact terms — drug names, scores (CHA2DS2-VASc), ICD codes, thresholds. BM25 nails exact-term matching; dense handles paraphrase. Neither alone is sufficient.

### E2. Vector store

Any of Qdrant, Weaviate, or pgvector. **Recommend Qdrant** for this build: native hybrid search, first-class payload filtering, and quantization if you scale to all volumes. If you're already on Postgres for the rule table, pgvector keeps everything in one database — a real operational advantage for a prototype, at the cost of weaker hybrid support.

### E3. Indexing configuration

- Distance: cosine
- Create **payload indexes** on: `chunk_type`, `specialty`, `condition`, `icd10`, `procedure_codes`, `doc_version`
- Store full `structured_data` in the payload — avoids a second round-trip

### E4. Index the enrichments

Three vectors per chunk region: the chunk text, each hypothetical question, and (optionally) a summary. All point back to the same `chunk_id`. Deduplicate by `chunk_id` after retrieval.

---

## PHASE F — Query understanding

An incoming PA request is not a question. It's a structured artifact: patient demographics, diagnosis codes, requested procedure, clinical notes, prior treatments. Convert it into a query plan before retrieving anything.

### F1. Fact extraction

LLM call with strict JSON schema output:

```json
{
  "patient": {"age": 34, "sex": "F", "pregnancy_status": null},
  "diagnosis": {"icd10": ["D25.9"], "text": "symptomatic uterine fibroid"},
  "requested_service": {"code": "HBP-MG010", "text": "total abdominal hysterectomy",
                        "setting": "inpatient", "facility_level": "district"},
  "clinical_findings": [
    {"parameter": "uterine_size_weeks", "value": 10, "unit": "weeks"},
    {"parameter": "hemoglobin", "value": 9.2, "unit": "g/dl"}
  ],
  "prior_therapies": [
    {"therapy": "tranexamic acid", "duration_months": 2, "outcome": "partial response"}
  ],
  "documentation_present": ["ultrasound_report", "hemogram"],
  "documentation_absent": ["second_opinion", "endometrial_sampling"],
  "extraction_confidence": {"uterine_size_weeks": "high", "prior_therapies": "medium"}
}
```

**Emit `null` for anything not stated. Never infer.** A hallucinated clinical value produces a fabricated denial — the worst possible failure mode in this system. Track per-field confidence and treat "low" as absent.

### F2. Query decomposition

Generate targeted sub-queries per `chunk_type`:

- `criteria` → "diagnostic criteria for uterine fibroids"
- `exclusion` → "when hysterectomy should not be done"
- `procedure` → "indications for hysterectomy in leiomyoma"
- `investigation` → "required investigations before hysterectomy"
- `care_level` → "which facility level performs hysterectomy"

Run these in parallel. Aggregate at the fusion step. This beats a single vague query by a wide margin because each sub-query is short, specific, and matches the shape of one chunk type.

### F3. Filter derivation

From extracted facts, build hard filters:

```python
must = [
  {"key": "procedure_codes", "match": {"any": ["HBP-MG010"]}},
]
should = [
  {"key": "icd10", "match": {"any": ["D25", "N84"]}},
]
```

Always OR the procedure filter with the ICD filter — never AND them. Codes are imperfect and an over-restrictive filter returns nothing, which the LLM will interpret as "no guideline exists."

---

## PHASE G — Retrieval

### G1. Hybrid search

For each sub-query, run dense and sparse in parallel. Retrieve top-k = 20 each.

### G2. Fusion

Combine with **Reciprocal Rank Fusion**:

```
RRF_score(d) = Σ over rankers r of  1 / (k + rank_r(d)),   k = 60
```

RRF is score-scale-agnostic, needs no tuning, and reliably beats weighted score blending. Use it unless you have a labeled set large enough to fit weights properly.

### G3. Mandatory rule-linked retrieval

**Independently of semantic search**, retrieve every chunk whose `linked_rule_ids` intersects the set of rules the engine evaluated. These chunks are not optional — they are the evidence for the decision. Merge them into the candidate pool with a guaranteed slot.

This is the mechanism that guarantees the explanation matches the decision. Without it, the rule engine may deny on rule X while the LLM cites chunk Y, and the two drift apart.

### G4. Exclusion sweep

Always retrieve all `chunk_type: exclusion` chunks for the matched condition, regardless of semantic score. Exclusion lists are short, phrased negatively, and semantically distant from the request — they systematically under-retrieve, and missing one causes a false approval. Force them in.

### G5. Candidate pool

Target 30–50 unique chunks entering the reranker.

---

## PHASE H — Reranking

### H1. Cross-encoder rerank

Model: `BAAI/bge-reranker-v2-m3` or Cohere Rerank v3.

Score each `(sub_query, chunk_text)` pair. Cross-encoders read query and document jointly and are dramatically more accurate than bi-encoder similarity — expect a large precision gain at the top of the list. Keep top 12–15.

### H2. Protected slots

Exempt from rerank pruning:
- All rule-linked chunks (G3)
- All exclusion chunks (G4)
- The `overview` chunk of the primary matched condition

These bypass reranking entirely and occupy guaranteed slots. Reranking optimizes relevance; these chunks are included for *correctness*, which is a different objective.

### H3. Diversity enforcement

Cap at 3 chunks per `chunk_type` and 8 per `parent_record_id`. Prevents the context window filling with five near-identical drug chunks while the exclusion list falls off the end.

### H4. Final context assembly

Order chunks: `overview` → `criteria` → `procedure` → `exclusion` → `investigation` → `care_level` → everything else. LLMs attend more strongly to the beginning and end of context; put the decision-critical material at the front and the exclusion list near the end where recall is also strong.

---

## PHASE I — Decisioning

### I1. Execute the rule engine FIRST

Before any LLM call. For each rule matching the requested procedure/diagnosis:

- All required fields present and logic true → `criteria_met`
- All required fields present and logic false → `criteria_not_met`
- Any required field missing → `insufficient_information`, record which field

Aggregate:

| Rule engine state | Provisional decision |
|---|---|
| Any hard `explicit_exclusion` fires | **DENY** |
| All applicable hard rules `criteria_met` | **APPROVE** |
| Any hard rule `criteria_not_met` | **PEND for clinical review** |
| Any rule `insufficient_information` | **REQUEST MORE INFORMATION** |
| No rule matches | **PEND** (fall through to LLM-assisted review) |

Precedence: DENY > RFI > PEND > APPROVE. Missing information never becomes a denial.

### I2. LLM synthesis

Give the LLM: the extracted facts, the rule engine output with per-rule detail, and the reranked chunks. Its job is **explanation and gap-finding, not decision-making**.

Permitted actions:
- Write the human-readable rationale, citing chunks
- Identify contraindications or missing prerequisites the rules did not encode
- **Escalate** APPROVE → PEND (never the reverse)
- List specific missing documents for the RFI branch

Forbidden:
- Overturn a deterministic DENY
- Upgrade PEND/RFI → APPROVE
- Assert any clinical fact not present in the retrieved chunks

### I3. Structured output

```json
{
  "decision": "request_more_information",
  "confidence": "high",
  "rule_evaluations": [
    {"rule_id": "R-OBG-HYST-002", "result": "criteria_not_met",
     "detail": "Uterine size 10 weeks is below the 14-week threshold for asymptomatic fibroids",
     "citation": {"page": 38, "chunk_id": "..."}}
  ],
  "missing_information": [
    {"field": "second_opinion_documentation",
     "why": "Patient is 34 years old; a second opinion from a qualified gynaecologist is mandatory for women under 40",
     "citation": {"page": 38}}
  ],
  "clinical_rationale": "...",
  "citations": [{"page": 38, "quote": "...", "chunk_id": "..."}],
  "guideline_version": "ICMR-STW-2019-V1",
  "evaluated_at": "..."
}
```

### I4. Citation enforcement

Post-process: every factual claim in `clinical_rationale` must map to a `chunk_id` in the retrieved set. Run a verification pass — a second LLM call or string-overlap check — that flags unsupported sentences. Unsupported claims get stripped or the response is regenerated. Do not ship a system that can cite a page it did not read.

---

## PHASE J — Output & audit

Every decision writes an immutable audit record:

```
request_id, timestamp, extracted_facts, rules_evaluated[], rule_versions[],
retrieved_chunk_ids[], reranker_scores[], model_ids, prompt_version,
decision, rationale, reviewer_override (nullable)
```

Two properties this buys you:
1. **Reproducibility** — pin rule and prompt versions and you can replay any historical decision exactly.
2. **Appeal support** — when a provider disputes, you show the exact guideline page and the exact rule.

Build a reviewer UI showing: decision, the firing rule with its threshold, the cited guideline text, and the source page image. The page image matters — a nurse reviewer trusts the original infographic far more than a paraphrase.

---

## PHASE K — Evaluation

Do not skip this. Untested RAG systems are confidently wrong in ways that are invisible without measurement.

### K1. Golden set

Hand-build 60–100 test requests spanning:
- Clear approvals (all criteria met)
- Clear denials (explicit exclusion list hit)
- RFI cases (a required field absent)
- Threshold edge cases (13-week vs 14-week fibroid, CHA2DS2-VASc exactly 1 vs 2)
- Cross-condition confusion (stable angina vs UA/NSTEMI vs STEMI)
- Out-of-corpus requests (must not hallucinate a rule)

Label each with expected decision, expected citation pages, and expected missing-info fields.

### K2. Retrieval metrics
- **Recall@20** on gold chunks — target >0.95. Recall is the binding constraint; a chunk not retrieved cannot be reranked.
- **nDCG@10** post-rerank
- **Exclusion-chunk retrieval rate** — must be 1.0. Any miss is a potential false approval.

### K3. Decision metrics
- Exact-match decision accuracy
- **False-approval rate** — weight this heaviest; approving something the guideline excludes is the costliest error
- **False-denial rate** — second heaviest; harms patients and generates appeals
- RFI precision — are the requested fields actually the missing ones?

### K4. Citation metrics
- Citation accuracy: does the cited page actually contain the claim?
- Hallucinated-citation rate — must be 0

### K5. Ablations
Measure the contribution of each component so you can defend the design:
dense only → +sparse → +contextual headers → +rerank → +rule engine. Report the deltas. Contextual headers and reranking will show the largest retrieval gains; the rule engine will show the largest decision-accuracy gain.

---

## PHASE L — Operations

- **Cache** embeddings and reranker scores keyed by content hash
- **Version** rules, prompts, and the corpus independently; log all three per decision
- **Human-in-the-loop:** route low-confidence and all DENY decisions to a reviewer; capture overrides as training signal
- **Drift monitoring:** track distribution of decisions over time; a sudden shift means corpus, prompt, or model changed
- **Guideline updates:** when ICMR publishes a new edition, re-extract, diff the rule table, and require human approval on every changed threshold

---

## 12. AGENT PROMPT

Copy everything below to your implementation agent.

---

> You are building a **Prior Authorization Triage and Policy Companion** — a hybrid rules + retrieval system that evaluates prior authorization requests against clinical guidelines and returns one of three recommendations (**approve**, **pend for nurse review**, **request more information**) with fully cited reasoning.
>
> **Input corpus:** `icmr_stw_vol1_rules.json` — 53 structured clinical rule records extracted from the ICMR Standard Treatment Workflows of India (2019, Volume 1). Each record covers one clinical condition and contains diagnostic criteria, severity scores with component points and thresholds, tiered investigations (essential/desirable/optional), care-level requirements, referral and admission criteria, procedures with indication lists, drugs with doses and contraindications, and a `prior_auth_notes` field. Every record carries a `source.page` for citation.
>
> ### Non-negotiable architectural constraint
>
> Build **two cooperating layers**, not a pure vector RAG pipeline:
>
> 1. **A deterministic rule engine** holding all numeric thresholds, closed indication lists, explicit exclusion lists, contraindications, prior-therapy requirements, care-level gates, and temporal windows. This layer produces the decision. It executes as code and is reproducible across identical inputs.
> 2. **A retrieval layer** over the guideline narrative, which produces the explanation, surfaces missing information, and handles requests no rule covers.
>
> The LLM may never overturn a deterministic denial and may never upgrade a pend or request-more-information to an approval. It may escalate an approval to a pend if it finds a contraindication the rules missed. Decision precedence is DENY > REQUEST_MORE_INFORMATION > PEND > APPROVE. **Missing data must never become a denial.**
>
> ### Build these phases in order
>
> **1. Corpus preparation.** Load and validate the JSON; reject any record lacking `source.page`. Normalize comparators (`greater than` / `more than` / `over` / `>` → a canonical `{op, value, unit}` form), units, and drug-name variants, preserving the raw wording alongside the normalized form. Map every procedure in the corpus to a procedure code system (PM-JAY HBP package codes for an Indian payer context, or CPT/HCPCS for a US context). Build this mapping manually for the roughly 40 procedures present — incoming requests arrive as codes, not condition names, and without this mapping the system degrades to fuzzy string matching. Attach a provenance block to every record with `review_status: "unverified"`.
>
> **2. Rule extraction.** Walk the corpus and lift every executable rule into a versioned rule table (Postgres or SQLite) using this schema: `rule_id`, `source_record_id`, `source_page`, `applies_to` (procedure codes and ICD-10), `rule_type`, `logic` (nested all/any predicates over named fields), `outcome_if_true`, `outcome_if_false`, `outcome_if_unknown`, `required_fields`, `citation_text`, `severity` (hard or soft), `version`, `review_status`. Rule types to cover: numeric threshold, score threshold, closed indication list, explicit exclusion, contraindication, prior-therapy-required, care-level gate, temporal window, frequency threshold, sequencing rule, documentation requirement. The `outcome_if_unknown` field drives the request-more-information branch and is mandatory on every rule. Expect 120–160 rules. Make the rule table editable via YAML or a config UI without code changes — configurability is an explicit requirement. Log every rule edit to an audit table.
>
> **3. Chunking.** Chunk at semantic nodes, never by character count — a sliced indication list produces confidently wrong denials. Emit one chunk per second-level node with `chunk_type` in: overview, criteria, score, investigation, procedure, care_level, referral, admission, discharge, drug, exclusion, payer_note. Lists are atomic and must never be split, even if long. Prepend a contextual header to every chunk before embedding, in the form `[{condition} | {specialty} | ICD-10 {codes} | {chunk_type} | ICMR STW 2019 Vol 1 p.{page}]` — many chunks do not name their own condition in the body and will otherwise be unretrievable. Render JSON to prose with deterministic template functions, not LLM calls. Retain the original JSON node as `structured_data` in the payload. Store `parent_record_id` on every chunk to enable expansion to full-record context at read time. Expect 450–600 chunks.
>
> **4. Enrichment.** For each chunk, generate 3–5 hypothetical questions it would answer and index them as additional vectors pointing to the same chunk id — this bridges the gap between guideline phrasing and request phrasing. Extract and tag normalized entities (drugs, procedures, lab tests, imaging modalities, anatomical sites). Build a separate flat threshold index of every numeric threshold in the corpus and query it directly for numeric questions; embeddings cannot reliably distinguish "greater than 12 weeks" from "greater than 14 weeks".
>
> **5. Embedding and indexing.** Use hybrid dense plus sparse retrieval. Dense: `BAAI/bge-m3` or `intfloat/e5-large-v2`; benchmark a biomedical model such as MedCPT against these on your own eval set rather than assuming domain models win. Sparse: BM25 or SPLADE — sparse matters unusually much here because the corpus is dense with exact terms (drug names, CHA2DS2-VASc, ICD codes, numeric thresholds). Use Qdrant with cosine distance, or pgvector if you want rules and vectors in one database. Create payload indexes on `chunk_type`, `specialty`, `condition`, `icd10`, `procedure_codes`, `doc_version`.
>
> **6. Query understanding.** Convert the incoming request into structured facts via a strict-JSON LLM call: patient demographics, diagnosis codes, requested service with code and facility level, clinical findings as parameter/value/unit triples, prior therapies with durations and outcomes, documentation present and absent, and a per-field extraction confidence. Emit null for anything not stated and never infer a clinical value — a hallucinated value produces a fabricated denial. Treat low-confidence extractions as absent. Then decompose into parallel sub-queries, one per relevant chunk_type. Derive retrieval filters from the extracted facts, always OR-ing the procedure-code filter with the ICD filter rather than AND-ing them, so an imperfect code does not return an empty result set.
>
> **7. Retrieval.** Run dense and sparse in parallel per sub-query at top-k 20 each. Fuse with Reciprocal Rank Fusion at k=60. Then, independently of semantic scores, force two additional sets into the candidate pool: every chunk whose `linked_rule_ids` intersects the rules the engine evaluated (this guarantees the explanation matches the decision), and every `exclusion`-type chunk for the matched condition (exclusion lists are phrased negatively, sit far from the query in embedding space, and systematically under-retrieve — missing one causes a false approval). Target 30–50 unique candidates.
>
> **8. Reranking.** Cross-encoder rerank with `BAAI/bge-reranker-v2-m3` or Cohere Rerank v3, scoring each sub-query against each chunk. Keep top 12–15. Exempt rule-linked chunks, exclusion chunks, and the primary condition's overview chunk from pruning — these occupy guaranteed slots because they are included for correctness rather than relevance. Cap at 3 chunks per chunk_type and 8 per parent record to preserve diversity. Order the final context as overview, criteria, procedure, exclusion, investigation, care_level, then the rest.
>
> **9. Decisioning.** Execute the rule engine before any LLM call. For each applicable rule, resolve to criteria_met, criteria_not_met, or insufficient_information with the specific missing field named. Aggregate by the precedence order above. Then call the LLM with the extracted facts, the per-rule engine output, and the reranked chunks, constrained to writing the rationale, identifying gaps the rules did not encode, and escalating approve to pend where warranted. Return structured JSON containing decision, confidence, per-rule evaluations with citations, missing-information items each with a reason and a citation, clinical rationale, citation list with page numbers and quotes, guideline version, and timestamp. Post-process to verify every factual claim in the rationale maps to a retrieved chunk; strip or regenerate unsupported sentences.
>
> **10. Audit and review UI.** Write an immutable audit record per decision capturing request id, timestamp, extracted facts, rules evaluated with their versions, retrieved chunk ids, reranker scores, model ids, prompt version, decision, rationale, and any reviewer override. Build a reviewer interface showing the decision, the firing rule with its threshold, the cited guideline text, and the source page image — reviewers trust the original infographic far more than a paraphrase.
>
> **11. Evaluation.** Hand-build a golden set of 60–100 requests covering clear approvals, explicit-exclusion denials, request-more-information cases, threshold edge cases (13-week versus 14-week fibroid; CHA2DS2-VASc exactly 1 versus 2), cross-condition confusion (stable angina versus UA/NSTEMI versus STEMI), and out-of-corpus requests that must not produce a fabricated rule. Measure Recall@20 on gold chunks with a target above 0.95, nDCG@10 after reranking, exclusion-chunk retrieval rate which must be 1.0, exact-match decision accuracy, false-approval rate (weight heaviest), false-denial rate, request-more-information precision, citation accuracy, and hallucinated-citation rate which must be zero. Run ablations across dense-only, plus sparse, plus contextual headers, plus reranking, plus rule engine, and report the deltas.
>
> ### Deliverables
>
> `corpus_loader.py`, `rule_extractor.py`, `rules.yaml`, `chunker.py`, `enricher.py`, `indexer.py`, `query_parser.py`, `retriever.py`, `reranker.py`, `rule_engine.py`, `decision_engine.py`, `evaluate.py`, `golden_set.jsonl`, a reviewer UI, and a README documenting how to add or edit a rule without touching code.
>
> ### Standing constraints
>
> Never let the LLM invent a clinical threshold. Never let missing data become a denial. Never ship a citation to a page the system did not retrieve. Treat every numeric threshold in the corpus as unverified until a human has checked it against the source page image — this corpus was extracted from infographic PDFs, where numbers are the most error-prone field and the most consequential.

---

## Appendix — Highest-value records for a demo

These carry the densest, cleanest rules and will show best under scrutiny:

| Record | Why |
|---|---|
| Hysterectomy for Benign Gynaecological Conditions (p.38) | Explicit six-item DENY list, size thresholds, mandatory second opinion under 40 |
| Pharyngitis and Sore Throat (p.30) | Centor score → antibiotic decision; exact tonsillectomy episode counts |
| Renal and Ureteric Stones (p.59) | Size threshold plus mandatory 4-week medical expulsive therapy trial |
| Chronic Kidney Disease (p.31) | eGFR gates for fistula, transplant listing, dialysis initiation |
| Stroke (p.35) | Hard temporal windows for tPA and thrombectomy |
| Psychosis (p.56) | Clozapine gated on two failed adequately-dosed antipsychotic trials |
| Dengue Fever (p.45) | Platelet transfusion threshold under 10,000 — heavily over-utilised in practice |
| Fever in Children (p.46) | Duration-and-localisation matrix governing investigation authorisation |
| Heart Failure (p.14) | CRT requires four simultaneous criteria including 3 months optimal therapy |
| Acute Urinary Retention (p.56) | Explicit "not to be done routinely" list for cystoscopy, CT, urodynamics |
