import { Check, Lock, Minus, X } from 'lucide-react'
import { Card, Meter, pct } from './ui'

/* --------------------------------------------------------------------------
   Decision ledger

   The audit artifact. One row per criterion: what was tested, what the case
   showed, what the policy required, and how much the criterion counted toward
   the outcome. A reviewer should be able to reconstruct the decision from this
   table alone, without reading any code.
   -------------------------------------------------------------------------- */
export function DecisionLedger({ criteria, rationale, necessityScore }) {
  if (!criteria?.length) return null
  const scored = criteria.filter((c) => !c.blocking)
  const blocking = criteria.filter((c) => c.blocking)

  return (
    <Card
      eyebrow="Audit record"
      title="Decision ledger"
      bodyClass="p-0"
      action={
        necessityScore != null && (
          <div className="text-right">
            <div className="eyebrow">Necessity</div>
            <div className="num text-lg font-semibold leading-none">{pct(necessityScore)}</div>
          </div>
        )
      }
    >
      {rationale && (
        <p className="border-b border-rule bg-canvas/60 px-4 py-3 text-[13px] text-ink-2">
          {rationale}
        </p>
      )}

      <table className="w-full">
        <thead>
          <tr>
            <th className="th w-8" />
            <th className="th">Criterion</th>
            <th className="th">Observed</th>
            <th className="th">Policy requires</th>
            <th className="th w-28 text-right">Weight</th>
          </tr>
        </thead>
        <tbody>
          {[...blocking, ...scored].map((c) => (
            <tr key={c.code} className={c.passed ? '' : 'bg-deny-soft/40'}>
              <td className="td text-center">
                {c.passed ? (
                  <Check size={14} className="mx-auto text-approve" strokeWidth={2.5} />
                ) : (
                  <X size={14} className="mx-auto text-deny" strokeWidth={2.5} />
                )}
              </td>
              <td className="td">
                <div className="flex items-center gap-2">
                  <span className="num text-2xs text-ink-3">{c.code}</span>
                  <span className="font-medium">{c.label}</span>
                  {c.blocking && (
                    <span title="Coverage gate — overrides all scoring">
                      <Lock size={11} className="text-ink-3" />
                    </span>
                  )}
                </div>
              </td>
              <td className="td text-ink-2">{c.observed}</td>
              <td className="td text-ink-3">{c.expected}</td>
              <td className="td">
                {c.blocking ? (
                  <div className="text-right text-2xs uppercase tracking-wide text-ink-3">
                    Gate
                  </div>
                ) : (
                  <div className="flex items-center justify-end gap-2">
                    <div className="w-14">
                      <Meter value={c.weight} max={0.25} tone={c.passed ? 'approve' : 'deny'} />
                    </div>
                    <span className="num w-8 text-right text-2xs text-ink-3">
                      {c.weight.toFixed(2)}
                    </span>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

/* --------------------------------------------------------------------------
   Attribution rail

   Per-case feature contributions to the policy-fit score, diverging from a
   center axis: right of the line pushed the score up, left pushed it down.
   -------------------------------------------------------------------------- */
export function AttributionRail({ explanation }) {
  if (!explanation?.contributions?.length) return null
  const rows = explanation.contributions
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution))) || 1

  return (
    <Card
      eyebrow="Explainable AI"
      title="What moved the policy-fit score"
      bodyClass="p-0"
      action={
        <div className="text-right">
          <div className="eyebrow">Score</div>
          <div className="num text-lg font-semibold leading-none">
            {explanation.base_score?.toFixed(3)}
          </div>
        </div>
      }
    >
      <div className="divide-y divide-rule/70">
        {rows.map((r) => {
          const up = r.contribution > 0
          const width = (Math.abs(r.contribution) / max) * 50
          return (
            <div key={r.feature} className="grid grid-cols-[minmax(0,1fr)_180px] items-center gap-3 px-4 py-2">
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="truncate text-[13px] font-medium">{r.label}</span>
                  {r.actionable && (
                    <span className="chip border-provider-line bg-provider-soft text-provider-deep">
                      fixable
                    </span>
                  )}
                </div>
                <div className="truncate text-2xs text-ink-3">
                  <span className="num">{String(r.value)}</span>
                  <span className="mx-1.5">vs typical</span>
                  <span className="num">{String(r.reference)}</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="relative h-4 flex-1">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-ruleStrong" />
                  <div
                    className={`absolute top-1/2 h-2.5 -translate-y-1/2 rounded-sm ${
                      up ? 'bg-approve' : 'bg-deny'
                    }`}
                    style={
                      up
                        ? { left: '50%', width: `${width}%` }
                        : { right: '50%', width: `${width}%` }
                    }
                  />
                </div>
                <span
                  className={`num w-14 text-right text-2xs ${up ? 'text-approve' : 'text-deny'}`}
                >
                  {r.contribution > 0 ? '+' : ''}
                  {r.contribution.toFixed(4)}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {explanation.levers?.length > 0 && (
        <footer className="border-t border-rule bg-provider-soft px-4 py-3">
          <div className="eyebrow text-provider-deep">Raise the score by fixing</div>
          <ul className="mt-1.5 space-y-1">
            {explanation.levers.map((l) => (
              <li key={l.feature} className="flex items-start gap-2 text-[13px] text-provider-deep">
                <Minus size={12} className="mt-1 shrink-0" />
                <span>
                  {l.label} is <span className="num">{String(l.value)}</span> against a typical{' '}
                  <span className="num">{String(l.reference)}</span>, costing{' '}
                  <span className="num">{Math.abs(l.contribution).toFixed(4)}</span>.
                </span>
              </li>
            ))}
          </ul>
        </footer>
      )}

      <p className="border-t border-rule px-4 py-2.5 text-2xs text-ink-3">
        {explanation.method}. {explanation.caveat}
      </p>
    </Card>
  )
}

/* Appeal-propensity model output, shown with its own accuracy so nobody reads
   more into the number than the model supports. */
export function AppealForecast({ prediction }) {
  if (!prediction) return null
  const weak =
    prediction.model_macro_auc != null && prediction.model_macro_auc < 0.6

  return (
    <Card eyebrow="Appeal prediction model" title="Likelihood of an appeal" bodyClass="p-4">
      <div className="flex items-baseline gap-2">
        <span className="num text-[30px] font-semibold leading-none">
          {pct(prediction.any_appeal_probability)}
        </span>
        <span className="text-[13px] text-ink-2">chance this denial is challenged</span>
      </div>
      <p className="mt-1 text-[13px] text-ink-2">
        Most likely path: <span className="font-medium">{prediction.top_label}</span> at{' '}
        <span className="num">{pct(prediction.top_probability)}</span>.
      </p>

      <div className="mt-4 space-y-2">
        {prediction.distribution?.map((d) => (
          <div key={d.outcome} className="grid grid-cols-[150px_1fr_44px] items-center gap-3">
            <span className="truncate text-2xs text-ink-2">{d.label}</span>
            <Meter value={d.probability} tone="payer" />
            <span className="num text-right text-2xs text-ink-3">{pct(d.probability, 1)}</span>
          </div>
        ))}
      </div>

      {weak && (
        <p className="mt-4 rounded-md border border-review-line bg-review-soft px-3 py-2 text-2xs leading-relaxed text-review">
          Held-out macro AUC is{' '}
          <span className="num">{prediction.model_macro_auc?.toFixed(3)}</span> against 0.500 for
          random guessing, and accuracy is{' '}
          <span className="num">{pct(prediction.model_accuracy, 1)}</span> against a{' '}
          <span className="num">{pct(prediction.baseline_accuracy, 1)}</span> majority-class
          baseline. Treat these probabilities as close to uninformative until the model is
          retrained on data that separates the classes.
        </p>
      )}
    </Card>
  )
}