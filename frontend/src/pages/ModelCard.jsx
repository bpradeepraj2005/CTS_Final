import { useEffect, useState } from 'react'
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import { api } from '../lib/api'
import { Card, Spinner, pct } from '../components/ui'

/* The figures on this page are read from what is actually running: the live
   /health of the guideline service and the loaded artifact of the supporting-
   material model. Nothing here is a stored training metric, because neither
   model is trained by this repository any more.

   The previous version of this page reported held-out R2 and MAE for a
   scikit-learn regressor. That model has been replaced by a retrieval-and-
   reasoning service, which has no such numbers. Rather than fill the fields
   with something that looks like accuracy, the page now reports provenance:
   which guideline, which version, and what the model behind the appeal
   percentages was actually fitted on. */
export default function ModelCard() {
  const [data, setData] = useState(null)

  useEffect(() => {
    api
      .get('/api/dashboard/model-card')
      .then(setData)
      .catch(() => setData({ metrics: {}, ready: {} }))
  }, [])

  if (!data) return <Spinner label="Loading model provenance" />

  const m = data.metrics || {}
  const m1 = m.model_1
  const m2 = m.model_2

  if (!m1 && !m2) {
    return (
      <Card title="No model information available">
        <p className="text-[13px] text-ink-2">
          The backend did not return provenance for either model. Check that the
          API is running and that <span className="num">PRIOR_AUTH_URL</span> is set.
        </p>
      </Card>
    )
  }

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <div>
        <div className="eyebrow">Transparency</div>
        <h1 className="mt-1 text-2xl font-semibold">Model card</h1>
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">
          What this platform runs, and what each part is and is not evidence of. The
          approve, deny and route-to-human decision is made by a deterministic rules
          engine; the two models below inform it rather than replace it.
        </p>
      </div>

      {m1 && <ModelOne m1={m1} />}
      {m2 && <ModelTwo m2={m2} />}

      <Card eyebrow="Not a model" title="Medical-necessity rules engine">
        <p className="text-[13px] leading-relaxed text-ink-2">
          The decision itself is a deterministic, weighted rules engine rather than a
          classifier. A decision that affects someone&apos;s treatment has to be
          reconstructable criterion by criterion for an audit, which the decision
          ledger on every case provides. Model 1 grounds that engine in a cited
          guideline; Model 2 only decides whether a denial is worth a human&apos;s time.
        </p>
        {m.notes?.length > 0 && (
          <ul className="mt-3 space-y-1.5">
            {m.notes.map((n) => (
              <li key={n} className="flex gap-2.5 text-[13px] leading-relaxed text-ink-2">
                <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
                {n}
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  )
}

/* ------------------------------------------------------------------ */

function ModelOne({ m1 }) {
  const up = m1.reachable
  return (
    <Card
      eyebrow="Model 1 of 2"
      title={m1.name || 'Guideline reasoning service'}
      action={
        <span
          className={`chip ${
            up
              ? 'border-approve-line bg-approve-soft text-approve'
              : 'border-deny-line bg-deny-soft text-deny'
          }`}
        >
          {up ? <CheckCircle2 size={11} /> : <XCircle size={11} />}
          {up ? 'reachable' : 'unreachable'}
        </span>
      }
    >
      <p className="text-[13px] leading-relaxed text-ink-2">
        Retrieves the guideline record that covers the case, then reasons over it to
        return a verdict per criterion with a page citation. The approval likelihood
        is a weighted share of the criteria the case satisfies — not a learned score —
        and unmet mandatory criteria cap it.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-4">
        <Stat label="Conditions indexed" value={fmtInt(m1.conditions_indexed)} />
        <Stat label="Criteria indexed" value={fmtInt(m1.criteria_indexed)} />
        <Stat label="Procedure codes" value={fmtInt(m1.procedure_codes)} />
        <Stat label="Text chunks" value={fmtInt(m1.chunks)} />
      </div>

      <dl className="mt-4 divide-y divide-rule/70 border-t border-rule">
        <Row label="Guideline version" value={m1.guideline_version} />
        <Row label="Rule table version" value={m1.rule_table_version} />
        <Row label="Prompt version" value={m1.prompt_version} />
        <Row label="Reasoning model" value={m1.reasoning_model} />
        <Row label="Endpoint" value={m1.endpoint} />
      </dl>

      {!up && (
        <div className="mt-4 rounded-md border border-deny-line bg-deny-soft p-4">
          <h4 className="text-[13px] font-semibold text-deny">
            The service is not answering
          </h4>
          <p className="mt-2 text-[13px] leading-relaxed text-deny">
            Adjudication returns 503 until it responds. On the free tier the instance
            spins down when idle and a cold start takes 50–90 seconds, so this often
            clears on its own within a minute.
          </p>
        </div>
      )}

      {m1.notes?.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {m1.notes.map((n) => (
            <li key={n} className="flex gap-2.5 text-[13px] leading-relaxed text-ink-2">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
              {n}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

function ModelTwo({ m2 }) {
  const loaded = m2.available
  /* Fitted on synthetic labels, so it has no demonstrated skill on real appeal
     behaviour. Say so plainly rather than presenting the percentages as findings. */
  const synthetic = String(m2.trained_on || '').toLowerCase().includes('synthetic')

  return (
    <Card
      eyebrow="Model 2 of 2"
      title={m2.name || 'Supporting-material assessment'}
      action={
        <span
          className={`chip ${
            !loaded
              ? 'border-deny-line bg-deny-soft text-deny'
              : synthetic
                ? 'border-review-line bg-review-soft text-review'
                : 'border-approve-line bg-approve-soft text-approve'
          }`}
        >
          {!loaded ? <XCircle size={11} /> : <AlertTriangle size={11} />}
          {!loaded ? 'not loaded' : synthetic ? 'synthetic training' : 'loaded'}
        </span>
      }
    >
      <p className="text-[13px] leading-relaxed text-ink-2">
        Runs on denied requests only. It splits every unmet criterion into gaps a
        provider can close with documentation and gaps no document will fix. Anything
        fixable pulls the case back to a human rather than auto-denying it.
      </p>

      {!loaded ? (
        <div className="mt-4 rounded-md border border-deny-line bg-deny-soft p-4">
          <h4 className="text-[13px] font-semibold text-deny">Model did not load</h4>
          <p className="mt-2 text-[13px] leading-relaxed text-deny">
            {m2.reason || 'Unknown error.'} Denials will stand as auto-denied without a
            supporting-material check until this is resolved.
          </p>
        </div>
      ) : (
        <>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <Stat label="Training rows" value={fmtInt(m2.training_rows)} />
            <Stat label="Positive base rate" value={pct(m2.base_rate, 1)} />
            <Stat
              label="Escalation percentile"
              value={m2.reappeal_percentile_threshold ?? '—'}
            />
          </div>

          <dl className="mt-4 divide-y divide-rule/70 border-t border-rule">
            <Row label="Artifact version" value={m2.version} />
            <Row label="Trained on" value={m2.trained_on} />
            <Row label="Runs on" value={m2.runs_on} />
          </dl>

          {synthetic && (
            <div className="mt-4 rounded-md border border-review-line bg-review-soft p-4">
              <h4 className="text-[13px] font-semibold text-review">
                The percentages are not yet evidence about real appeals
              </h4>
              <p className="mt-2 text-[13px] leading-relaxed text-review">
                {m2.caveat ||
                  'This model was fitted on synthetic labels. The pipeline is production-shaped, but the numbers are not observations of real appeal behaviour.'}
              </p>
              <p className="mt-2 text-[13px] leading-relaxed text-review">
                The fixable / not-fixable split a reviewer acts on is rule-based and
                auditable, and does not depend on this model. Only the ranking between
                cases does. Retrain on observed outcomes before treating the
                percentages as rates.
              </p>
            </div>
          )}
        </>
      )}

      {m2.notes?.length > 0 && (
        <ul className="mt-4 space-y-1.5">
          {m2.notes.map((n) => (
            <li key={n} className="flex gap-2.5 text-[13px] leading-relaxed text-ink-2">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
              {n}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

/* ------------------------------------------------------------------ */

const fmtInt = (v) =>
  v === null || v === undefined ? '—' : Number(v).toLocaleString()

function Stat({ label, value }) {
  return (
    <div className="rounded-md border border-rule bg-canvas px-3 py-2.5">
      <div className="eyebrow">{label}</div>
      <div className="num mt-1 text-lg font-semibold leading-none">{value ?? '—'}</div>
    </div>
  )
}

function Row({ label, value }) {
  if (!value) return null
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <dt className="shrink-0 text-2xs uppercase tracking-wide text-ink-3">{label}</dt>
      <dd className="num truncate text-right text-[13px] text-ink-2" title={String(value)}>
        {String(value)}
      </dd>
    </div>
  )
}
