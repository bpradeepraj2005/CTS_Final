# Prior Authorization Platform

A prior-authorization system for medical requests. A provider submits a request,
the platform checks it against a cited clinical guideline, and it is either
auto-approved, auto-denied, or routed to a human reviewer — with the reasoning
shown criterion by criterion.

The decision itself is made by a deterministic rules engine, not a classifier. A
decision that affects someone's treatment has to be reconstructable line by line
for an audit. Two models inform that engine; neither replaces it.

---

## Architecture

```
                          Provider submits request
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │  MODEL 1 — guideline reasoning      │
              │  (hosted service, POST /analyze)    │
              └─────────────────────────────────────┘
                                    │
                        Report: per-criterion verdicts
                        + page citations + flags
                                    │
                                    ▼
              ┌─────────────────────────────────────┐
              │  Decision Router (necessity_engine) │
              └─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   AUTO_APPROVED             PENDING_REVIEW                  denied
   (fit ≥ 0.62)              (0.38 < fit < 0.62)           (fit ≤ 0.38)
        │                           │                           │
        │                           │                           ▼
        │                           │        ┌──────────────────────────────────┐
        │                           │        │  MODEL 2 — supporting material   │
        │                           │        │  appeal regressor + gap split    │
        │                           │        └──────────────────────────────────┘
        │                           │                           │
        │                           │              ┌────────────┴────────────┐
        │                           │              ▼                         ▼
        │                           │        fixable gaps →           only hard gaps →
        │                           │        HUMAN_REVIEW              AUTO_DENIED
        │                           │              │                         │
        └───────────────────────────┴──────────────┴─────────────────────────┘
                                    ▼
                      Decision ledger + audit trail
```

**Model 2 only ever runs on denials.** An approved request carries no appeal
prediction, because the question was never asked.

---

## The two models

### Model 1 — guideline reasoning

A hosted service that retrieves the guideline record covering a case and reasons
over it, returning a verdict per criterion with a page citation.

| | |
|---|---|
| Endpoint | `POST /analyze` |
| Guideline | ICMR STW 2019 (`ICMR-STW-2019-V1`) |
| Corpus | 53 conditions · 68 criteria · 60 procedure codes · 455 chunks |
| Latency | ~54s warm, 50–90s cold |

From one Report the backend derives **approval likelihood** (weighted share of
satisfied criteria, capped when mandatory ones fail), a **complexity score**, and
the **attribution rail**. One HTTP call per request.

### Model 2 — supporting-material assessment

Two independent halves, and the split is deliberate:

**A rule-based gap classifier** (`app/services/curability.py`) sorts unmet
criteria by what it would actually take to close them:

| Class | Time to close | Fixable? |
|---|---|---|
| `CLERICAL` | a form, a signature — hours | yes |
| `PROCURABLE_FAST` | a test, a scan, a lab panel — days | yes |
| `PROCURABLE_SLOW` | a treatment trial — weeks | no |
| `BEHAVIOURAL` | the patient declined | no |
| `CLINICAL_FACT` | a measurement against a threshold | no |
| `CATEGORICAL` | an explicit guideline exclusion | no |

**A HistGradientBoostingRegressor bundle** (`ml/models/appeal_propensity.joblib`)
— five regressors: one for appeal propensity, four for the outcome distribution.

Routing turns on the **gap split**, not the model. That separation matters — see
Known limitations.

---

## Setup

**Requires:** Python 3.13, Node 18+

```bash
# backend
cd backend
pip install -r requirements.txt
cp .env.example .env          # then set JWT_SECRET
uvicorn app.main:app --reload --port 8000

# frontend (separate terminal)
cd frontend
npm install
npm run dev
```

On start the backend prints:

```
Model 1  guideline service: REACHABLE
Model 2  supporting-material: READY
```

`UNREACHABLE` usually means the guideline service is waking from idle — wait a
minute and retry.

### Configuration

Set in `backend/.env`:

| Variable | Default | Notes |
|---|---|---|
| `PRIOR_AUTH_URL` | Render URL | Model 1 endpoint |
| `PRIOR_AUTH_READ_TIMEOUT` | `120` | Sized for cold start — don't lower |
| `MODEL2_ENABLED` | `1` | |
| `MODEL2_REAPPEAL_PERCENTILE` | `80` | Set to `100` to disable the escalation override |
| `AUTO_APPROVE_MIN_POLICY_FIT` | `0.62` | Router threshold |
| `AUTO_DENY_MAX_POLICY_FIT` | `0.38` | Router threshold |
| `JWT_SECRET` | — | **Change before deploying** |

---

## Training Model 2

```bash
cd backend
python ml/train_appeal.py --csv data/appeals_prediction_transformed.csv
```

Writes `ml/models/appeal_propensity.joblib` and `appeal_metrics.json`. The
artifact is committed so a fresh clone runs without a 90k-row training step.

> `ml/train.py` is **retired** and refuses to run. It wrote to the same filename
> and would overwrite the regressor with the old classifier.

---

## Project structure

```
backend/
  app/
    main.py                  FastAPI app, startup checks
    config.py                all settings and thresholds
    models.py                SQLAlchemy schema (7 tables)
    routers/                 auth, requests, review, dashboard, admin, validation, chat
    services/
      prior_auth_client.py   Model 1 client — case building, retry, caching
      model2.py              Model 2 — scoring + routing rules
      curability.py          rule-based gap classification
      ml.py                  approval likelihood, complexity, provenance
      explain.py             per-criterion attribution
      pipeline.py            end-to-end orchestration
      necessity_engine.py    the Decision Router
      hospital_predictor.py  packet-completeness scorer for surgical PA
      routing.py             reviewer assignment
  ml/
    feature_schema.py        shared feature contract
    train_appeal.py          Model 2 trainer
    models/                  appeal_propensity.joblib, appeal_metrics.json
  data/                      training CSV
frontend/
  src/
    pages/                   Requests, Review, ModelCard, dashboards, auth
    components/Explain.jsx   DecisionLedger, AttributionRail, AppealForecast,
                             SupportingMaterial
```

**Database:** SQLite (`backend/priorauth.db`) via SQLAlchemy — `organizations`,
`users`, `patients`, `auth_requests`, `documents`, `appeals`, `audit_events`.
Uploaded PDFs live on disk in `backend/uploads/`; only extracted text is stored.
Seting `DATABASE_URL` to a Postgres URL to switch — no code changes.

---

## Known limitations

Stated plainly, because a system that makes medical decisions should be honest
about what it does not know.

**The appeal regressor has almost no predictive signal.** Held-out ROC-AUC is
**0.537** against 0.500 for random guessing. Its four-outcome distribution beats
the majority-class baseline by 0.44 percentage points. In practice it reproduces
the class priors — roughly the same number for every patient.

This was verified three ways:

- adding back the two excluded feature columns changed AUC by **+0.0001**
- a **shuffled-label control** scored 0.5029 against 0.5372 for real labels,
  confirming the pipeline is sound and the data is not
- the original classifier on the same corpus scored 0.517

The dataset records case features and appeal outcomes but almost no relationship
between them. `subsequent_attempt_count` correlates 0.77 with appeal, but it is
recorded *after* the appeal — using it would be target leakage.

**This is why routing does not depend on it.** The fixable-vs-unfixable split is
rule-based and auditable; the model only supplies percentages and one percentile
override. The appeal card shows a "treat as uninformative" caveat automatically.
Fixing this needs different data, not different code.

**Other constraints:**

- Model 1 runs on a free instance that spins down when idle — first request after
  a quiet period takes 50–90 seconds
- `policy_fit_score` and `clinical_evidence_score` are excluded from Model 2's
  features: the live pipeline no longer produces them in the units the CSV
  recorded, and training on a feature whose meaning shifts at serve time is how a
  model quietly degrades
- The training corpus is denial-only, which matches where Model 2 runs — it knows
  nothing about approved requests and must never be asked about one
- Auto-denial is deliberately rare: it requires *every* unmet criterion to be
  unfixable by documentation

---

## Design notes

**Why a rules engine and not an end-to-end classifier.** The training data
contains denied cases only, so an approve/deny boundary cannot be learned from
it. More fundamentally, a denial has to be explainable to the provider and
defensible to a regulator — the decision ledger reconstructs every decision
criterion by criterion.

**Why the gap split is not learned.** It decides whether a case reaches a human.
That question should not rest on a fitted model, and given the regressor's
measured performance, visibly should not.

**Why hospital packets go through both.** `hospital_predictor.py` scores whether
a packet is *complete*, not whether the procedure is *indicated*. A thoroughly
documented request for an unsupported procedure used to auto-approve. Both legs
must now hold; the guideline can downgrade an approval but never create one.
