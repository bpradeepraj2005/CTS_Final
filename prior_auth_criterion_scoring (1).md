# Prior Auth — Criterion Evaluation & Scoring Model

**What this covers:** patient request + documents come in → system retrieves the criteria for that exact condition → evaluates each criterion → each one passes, fails, or needs more info → those verdicts produce a score and a decision.

Nothing else. This is the evaluation core.

---

## 1. The unit of work is a criterion

Every guideline condition decomposes into a list of criteria. A criterion is one checkable statement with an ID, a type, a weight, and a citation.

```json
{
  "criterion_id": "C-HYST-003",
  "condition": "Hysterectomy for benign gynaecological conditions",
  "text": "Asymptomatic fibroids greater than or equal to 14 weeks uterine size",
  "type": "mandatory",
  "evaluator": "deterministic",
  "check": {"field": "uterine_size_weeks", "op": "gte", "value": 14},
  "weight": 5,
  "citation": {"page": 38}
}
```

The system's job is to assign a **verdict** to each criterion, then turn the set of verdicts into a score.

---

## 2. Five criterion types

Type determines how a verdict affects the score. This is the backbone of the model.

| Type | Meaning | Example |
|---|---|---|
| **EXCLUSION** | Guideline explicitly says do *not* do this | "Hysterectomy should not be done for cervicitis" |
| **GATEWAY** | Must be true or nothing else matters | Diagnosis matches; procedure is the one the guideline covers |
| **MANDATORY** | Required for the service to be indicated | Uterus ≥14 weeks; EF <35% for CRT; onset <4.5h for tPA |
| **SUPPORTING** | Strengthens the case, not required alone | Anaemia present; documented dysmenorrhoea |
| **CONTEXTUAL** | Judgment call the guideline implies but doesn't quantify | "Failed medical management"; "rapidly enlarging"; "symptomatic" |

---

## 3. Four verdicts

| Verdict | When | Symbol |
|---|---|---|
| **PASS** | Evidence in the request satisfies the criterion | ✓ |
| **FAIL** | Evidence contradicts the criterion | ✗ |
| **INSUFFICIENT** | The data needed isn't in the request or documents | ? |
| **NOT_APPLICABLE** | Criterion doesn't apply to this patient | – |

**INSUFFICIENT is not FAIL.** "We can't tell if the uterus is 14 weeks" is a completely different statement from "the uterus is 10 weeks." Treating them the same denies people for paperwork.

**NOT_APPLICABLE removes the criterion from scoring entirely** — numerator and denominator both. A pregnancy criterion on a male patient shouldn't dilute anything.

---

## 4. The scoring rules — when each verdict scores high or low

Read this as: *given this criterion type and this verdict, what happens to the score.*

| Type | PASS ✓ | FAIL ✗ | INSUFFICIENT ? | N/A – |
|---|---|---|---|---|
| **EXCLUSION** | **Score = 0. Stop.** Patient matches a "never do this" | No effect (good — exclusion doesn't apply) | Flag for review; cap score at 60 | Drop |
| **GATEWAY** | Proceed to scoring | **Score = 0. Stop.** Wrong guideline or wrong procedure | **Score = null.** Cannot evaluate anything | Drop |
| **MANDATORY** | Full weight earned | **Cap total score at 35** and earn 0 | Earn 0, drop from denominator, **add to blocking gaps** | Drop |
| **SUPPORTING** | Full weight earned | Earn 0, stays in denominator | Earn 0, drop from denominator, add to material gaps | Drop |
| **CONTEXTUAL** | Weight × LLM confidence | Earn 0, stays in denominator | Earn 0, drop from denominator | Drop |

Three behaviours worth naming, because they're the ones that go wrong in naive implementations:

**A failed MANDATORY caps the score.** It does not merely subtract points. You cannot pile up supporting evidence to outweigh a criterion the guideline says is required. If two mandatory criteria fail, cap at 20.

**INSUFFICIENT leaves the denominator.** The score reflects only what could actually be evaluated. Six criteria where two are unknown means you score out of four — and you separately report that you only saw 4/6.

**A matched EXCLUSION is terminal.** No score, no weighing. The guideline named this as a non-indication.

---

## 5. The formula

```
1. Evaluate GATEWAY criteria.
   Any FAIL         → score = 0,    decision = not indicated / wrong guideline
   Any INSUFFICIENT → score = null, decision = need more info

2. Evaluate EXCLUSION criteria.
   Any PASS → score = 0, decision = excluded by guideline. Stop.

3. Score the rest.
   evaluable = [c for c in criteria if verdict in (PASS, FAIL)]

   earned    = Σ (weight × credit)      credit: PASS = 1, FAIL = 0,
                                                CONTEXTUAL PASS = llm_confidence
   available = Σ (weight) over evaluable

   score = 100 × earned / available

4. Apply caps.
   1 mandatory FAIL           → score = min(score, 35)
   2+ mandatory FAIL          → score = min(score, 20)
   any EXCLUSION INSUFFICIENT → score = min(score, 60)

5. Completeness (reported alongside, never blended in).
   completeness = 100 × count(PASS or FAIL) / count(all applicable criteria)
```

**Two numbers out: `score` and `completeness`.** Score says how well it meets criteria. Completeness says how much of the picture you actually had. Keep them separate — a score of 90 computed from 3 of 9 criteria is not the same as 90 from 9 of 9, and blending them hides exactly that.

### Weights

| Type | Default weight |
|---|---|
| MANDATORY | 5 |
| CONTEXTUAL (blocking judgment) | 4 |
| SUPPORTING (strong) | 3 |
| SUPPORTING (weak) | 2 |
| Documentation-only | 1 |

---

## 6. Where the LLM does the work

Split evaluation by criterion, not by request. Some criteria are arithmetic; some are judgment.

### Deterministic evaluator — code

Use when the criterion is a comparison against a field:

- `uterine_size_weeks >= 14`
- `hours_from_onset <= 4.5`
- `episodes_last_year >= 7`
- `egfr < 30`
- `centor_score == 4`

Code does this. Always the same answer, every time.

### LLM evaluator — judgment

Use when the guideline says something no comparison operator can capture. The corpus is full of these:

| Criterion text | Why code can't do it |
|---|---|
| "Symptomatic fibroids especially if not responding to medical management" | "Not responding" is a clinical read of the notes |
| "Rapidly enlarging fibroids" | No rate defined; requires comparing serial scans |
| "Failed medical treatment" | Depends on drug, dose, duration, adherence, and outcome |
| "Recurrence after failed conservative surgical management" | Needs a surgical history narrative |
| "Toxic appearance" | Pure clinical gestalt |
| "Adnexal masses: need to be individualised and justified" | The guideline explicitly defers to judgment |

**The LLM gets one criterion at a time**, plus the patient record and the retrieved guideline chunk. Not the whole case. One criterion, one verdict.

Required output per criterion:

```json
{
  "criterion_id": "C-HYST-005",
  "verdict": "PASS",
  "confidence": 0.72,
  "evidence_quote": "Patient on tranexamic acid and norethisterone for 5 months with continued heavy bleeding and Hb drop from 11.4 to 9.2",
  "evidence_source": "gynae_consult_note_2026-06-14",
  "reasoning": "Two agents from the recommended list, both beyond the guideline's 4-6 cycle trial period, with objective worsening. Meets 'failed medical treatment'."
}
```

Four hard rules on the LLM evaluator:

1. **Must quote actual evidence.** No quote → verdict is forced to INSUFFICIENT. This alone eliminates most hallucinated passes.
2. **Confidence below 0.5 → verdict becomes INSUFFICIENT.** A hedged pass is not a pass.
3. **Cannot invent clinical values.** If the note doesn't state duration, it doesn't state duration.
4. **Confidence scales the credit.** A CONTEXTUAL pass at 0.72 earns 72% of its weight, not 100%. Judgment calls shouldn't score like measurements.

### LLM's second job — semantic field mapping

Before any criterion is evaluated, the LLM maps messy input to canonical fields: "GFR 28" → `egfr: 28`, "uterus ~10 wk size" → `uterine_size_weeks: 10`, "on TXA since Feb" → `prior_therapy: {drug: tranexamic acid, start: 2026-02}`. This is the genuinely hard part and the part language models are actually good at. Emit `null` for anything absent — never infer.

---

## 7. Decision from score

| Score | Completeness | Blocking gaps | Decision |
|---|---|---|---|
| 0 (exclusion matched) | any | any | **DENY** — guideline excludes this |
| 0 (gateway failed) | any | any | **DENY** — not indicated / wrong guideline |
| null | any | any | **REQUEST MORE INFO** |
| ≥ 80 | ≥ 80% | none | **APPROVE** |
| ≥ 80 | ≥ 80% | any | **REQUEST MORE INFO** |
| ≥ 80 | < 80% | any | **REQUEST MORE INFO** |
| 55–79 | ≥ 70% | none | **PEND** — nurse review |
| 55–79 | any | any | **REQUEST MORE INFO** |
| 20–54 | ≥ 70% | none | **PEND** |
| 20–54 | any | any | **REQUEST MORE INFO** |
| < 20 | ≥ 70% | none | **PEND** — likely not indicated |

Three invariants, enforced with tests:

- **Never approve with a blocking gap**, no matter how high the score.
- **Never approve below 80% completeness.** A high score on a third of the criteria means nothing.
- **Never deny on a low score.** Only a matched exclusion or a failed gateway denies. A low score means the submission didn't establish necessity — that's a pend, and a human decides.

---

## 8. What to ask for

Every INSUFFICIENT verdict names the field it needed. Convert to a request list:

| Criticality | Rule | Ask? |
|---|---|---|
| **Blocking** | INSUFFICIENT on a MANDATORY or GATEWAY criterion | Yes, first |
| **Material** | INSUFFICIENT on a criterion whose weight ≥ 3 | Yes |
| **Minor** | Everything else | No |

**Cap at 3 items.** A long list is a denial wearing a costume, and providers treat it as one. Ask for the two or three things that decide the case.

Each request states the field, why it matters, and the page:

> *Second opinion from a qualified gynaecologist — mandatory for hysterectomy in women under 40 (ICMR STW 2019 Vol 1, p.38)*

---

## 9. Output

```json
{
  "condition_matched": "Hysterectomy for benign gynaecological conditions",
  "guideline_page": 38,
  "score": 35,
  "completeness": 67,
  "decision": "request_more_information",

  "criteria": [
    {"id": "C-HYST-001", "type": "gateway",   "verdict": "PASS", "weight": 5,
     "text": "Diagnosis is a benign gynaecological condition",
     "evidence": "D25.9 symptomatic uterine fibroid"},

    {"id": "C-HYST-002", "type": "exclusion", "verdict": "FAIL", "weight": 0,
     "text": "Not for asymptomatic fibroids under 12 weeks",
     "evidence": "Symptomatic: menorrhagia with Hb 9.2"},

    {"id": "C-HYST-003", "type": "mandatory", "verdict": "FAIL", "weight": 5,
     "text": "Fibroid 14 weeks uterine size or more",
     "evidence": "USG 2026-07-02: 10 week size uterus",
     "impact": "caps score at 35"},

    {"id": "C-HYST-005", "type": "contextual","verdict": "PASS", "weight": 4,
     "confidence": 0.72,
     "text": "Symptomatic and not responding to medical management",
     "evidence": "TXA + norethisterone 5 months, Hb fell 11.4 to 9.2",
     "evaluator": "llm"},

    {"id": "C-HYST-009", "type": "mandatory", "verdict": "INSUFFICIENT", "weight": 5,
     "text": "Second opinion required if age under 40",
     "missing_field": "second_opinion_documentation",
     "criticality": "blocking"},

    {"id": "C-HYST-011", "type": "supporting","verdict": "PASS", "weight": 2,
     "text": "Anaemia attributable to bleeding",
     "evidence": "Hb 9.2 g/dl"}
  ],

  "tally": {"pass": 3, "fail": 2, "insufficient": 1, "not_applicable": 0},
  "caps_applied": ["mandatory_fail -> 35"],

  "request_more_information": [
    {"item": "Second opinion from a qualified gynaecologist",
     "why": "Mandatory for hysterectomy in women under 40",
     "page": 38, "criticality": "blocking"}
  ],

  "summary": "Symptomatic fibroid with anaemia and documented failure of medical management, but uterine size is 10 weeks against a 14-week threshold, and the mandatory second opinion for patients under 40 is not on file. Cannot approve as submitted."
}
```

Show the criteria table in the UI, not the number. A reviewer needs to see *which three passed and which two failed* — the score alone tells them nothing actionable.

---

## 10. LLM evaluator prompt

```
You are evaluating ONE clinical criterion against ONE patient's submitted record.

CRITERION
  id:   {criterion_id}
  text: "{criterion_text}"
  source: {guideline_name}, page {page}

GUIDELINE CONTEXT (retrieved)
  {chunk_text}

PATIENT RECORD
  {structured_facts}

SUBMITTED DOCUMENTS
  {document_excerpts}

Return exactly one JSON object:

{
  "verdict": "PASS" | "FAIL" | "INSUFFICIENT" | "NOT_APPLICABLE",
  "confidence": 0.0-1.0,
  "evidence_quote": "verbatim text from the record supporting your verdict, or null",
  "evidence_source": "which document, or null",
  "reasoning": "one or two sentences",
  "missing_field": "what you would need to decide, only when INSUFFICIENT"
}

RULES
- PASS or FAIL requires a verbatim evidence_quote. No quote -> verdict is INSUFFICIENT.
- Never infer a clinical value that is not written down. Absence of a statement
  is not evidence of absence of a finding.
- Confidence below 0.5 -> return INSUFFICIENT, not a hedged PASS.
- NOT_APPLICABLE only when the criterion cannot apply to this patient at all
  (wrong sex, wrong age band, different diagnosis).
- Judge only this criterion. Do not comment on the overall request.
- Do not consider cost, coverage, or approval likelihood. Clinical criterion only.
```

---

## 11. Worked example

**Request:** hysterectomy · F/34 · symptomatic fibroid · USG 10-week uterus · Hb 9.2 · TXA + norethisterone × 5 months · no second opinion

| Criterion | Type | W | Verdict | Credit |
|---|---|---|---|---|
| Benign gynae diagnosis | gateway | 5 | PASS | proceed |
| Not an excluded indication | exclusion | – | FAIL (doesn't match) | no effect |
| Uterus 14 weeks or more | mandatory | 5 | **FAIL** | 0 → **cap 35** |
| Failed medical management | contextual | 4 | PASS @ 0.72 | 2.88 |
| Second opinion if under 40 | mandatory | 5 | **INSUFFICIENT** | dropped → **blocking** |
| Anaemia from bleeding | supporting | 2 | PASS | 2 |
| Child-bearing complete | contextual | 3 | INSUFFICIENT | dropped |

```
evaluable  = uterus(5), failed-management(4), anaemia(2)
earned     = 0 + 2.88 + 2 = 4.88
available  = 11
raw score  = 44
cap (1 mandatory fail) -> 35

completeness  = 4 evaluated / 6 applicable = 67%
blocking gaps = 1
```

**→ REQUEST MORE INFORMATION.** Ask for the second opinion, and for confirmation that child-bearing is complete.

Not a denial. The guideline doesn't prohibit hysterectomy for symptomatic fibroids — this submission hasn't cleared the size threshold and hasn't supplied a mandatory document. Those are different things, and the output says so.
