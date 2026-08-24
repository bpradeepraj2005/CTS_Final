import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  ArrowLeft, ArrowRightLeft, CheckCircle2, Inbox, PauseCircle, Users, XCircle,
} from 'lucide-react'
import { api } from '../lib/api'
import {
  Alert, Card, Empty, Field, Meter, Spinner, Status, fmtDate, pct,
} from '../components/ui'
import { AppealForecast, AttributionRail, DecisionLedger } from '../components/Explain'
import { AuditTrail, SubmittedValues } from './Requests'

const SCOPES = [
  ['mine', 'Assigned to me'],
  ['unassigned', 'Unassigned'],
  ['all', 'Everything pending'],
]

function urgencyTone(u) {
  if (u >= 0.75) return ['Critical', 'border-deny-line bg-deny-soft text-deny', 'deny']
  if (u >= 0.55) return ['High', 'border-review-line bg-review-soft text-review', 'review']
  if (u >= 0.35) return ['Standard', 'border-info-line bg-info-soft text-info', 'ink']
  return ['Routine', 'border-rule bg-canvas text-ink-3', 'ink']
}

/* The dark blue insurance-portal background is painted once by
   Layout for every payer route, so these pages only draw the
   panels that sit on top of it. */

export function ReviewQueue() {
  const [scope, setScope] = useState('mine')
  const [rows, setRows] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    setRows(null)
    api.get(`/api/review/queue?scope=${scope}`).then(setRows).catch(() => setRows([]))
  }, [scope])

  return (
    <>
      <div className="space-y-5">
        <div>
          <div className="eyebrow">Insurance organization</div>
          <h1 className="mt-1 text-2xl font-semibold">Review queue</h1>
          <p className="mt-1.5 text-[13px] text-ink-2">
            Ordered by clinical urgency, not arrival time. The most contested and most
            severe cases surface first.
          </p>
        </div>

        <div className="flex gap-1">
          {SCOPES.map(([value, label]) => (
            <button
              key={value} onClick={() => setScope(value)}
              className={
                scope === value
                  ? 'btn border-transparent bg-payer text-white hover:bg-payer-deep'
                  : 'btn-ghost'
              }
            >
              {label}
            </button>
          ))}
        </div>

        {!rows ? (
          <Spinner label="Loading queue" />
        ) : rows.length === 0 ? (
          <Card bodyClass="p-0">
            <Empty icon={Inbox} title="Queue is clear">
              {scope === 'mine'
                ? 'Nothing is assigned to you right now. Check the unassigned queue for cases waiting on any reviewer.'
                : 'No cases are waiting on a human reviewer.'}
            </Empty>
          </Card>
        ) : (
          <Card bodyClass="p-0" title={`${rows.length} case${rows.length === 1 ? '' : 's'} pending`}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="th w-28">Urgency</th>
                    <th className="th">Case</th>
                    <th className="th">Diagnosis and therapy</th>
                    <th className="th">Assigned</th>
                    <th className="th text-right">Policy fit</th>
                    <th className="th text-right">Necessity</th>
                    <th className="th text-right">Appeal risk</th>
                    <th className="th text-right">Filed</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const [label, chip, tone] = urgencyTone(r.urgency_score ?? 0)
                    return (
                      <tr key={r.id} className="row-link" onClick={() => navigate(`/payer/cases/${r.id}`)}>
                        <td className="td">
                          <span className={`chip ${chip}`}>{label}</span>
                          <div className="mt-1.5 w-16"><Meter value={r.urgency_score ?? 0} tone={tone} /></div>
                        </td>
                        <td className="td num text-2xs">{r.case_number}</td>
                        <td className="td">
                          <div className="font-medium">{r.diagnosis}</div>
                          <div className="text-2xs text-ink-3">
                            {r.requested_treatment} · {r.disease_severity} · {r.provider_specialty}
                          </div>
                        </td>
                        <td className="td">
                          {r.assigned_reviewer ? (
                            <>
                              <div className="text-[13px]">{r.assigned_reviewer.name}</div>
                              {r.assignment_was_reassigned && (
                                <div className="flex items-center gap-1 text-2xs text-review">
                                  <ArrowRightLeft size={10} /> reassigned
                                </div>
                              )}
                            </>
                          ) : (
                            <span className="text-2xs text-ink-3">Unassigned</span>
                          )}
                        </td>
                        <td className="td num text-right">{r.policy_fit_score?.toFixed(3) ?? '—'}</td>
                        <td className="td num text-right">{pct(r.necessity_score)}</td>
                        <td className="td num text-right text-ink-2">{pct(r.appeal_probability)}</td>
                        <td className="td text-right text-2xs text-ink-3">{fmtDate(r.created_at)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </>
  )
}

export function ReviewCase() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [audit, setAudit] = useState([])
  const [notes, setNotes] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = () => {
    api.get(`/api/review/cases/${id}`).then(setData).catch((e) => setError(e.message))
    api.get(`/api/requests/${id}/audit`).then(setAudit).catch(() => {})
  }
  useEffect(load, [id])

  if (error && !data) return <Alert>{error}</Alert>
  if (!data) return <Spinner label="Loading case" />

  const decide = async (decision) => {
    setBusy(true)
    setError(null)
    try {
      await api.post(`/api/review/cases/${id}/decide`, { decision, notes })
      navigate('/payer/queue')
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  const reassign = async (reviewerId) => {
    setBusy(true)
    try {
      await api.post(`/api/review/cases/${id}/reassign?reviewer_id=${reviewerId}`)
      load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const open = data.status === 'PENDING_REVIEW'
  const leaning = (data.necessity_score ?? 0) >= 0.5 ? 'approval' : 'denial'
  const [urgLabel, urgChip] = urgencyTone(data.urgency_score ?? 0)

  return (
    <div className="space-y-5">
      <Link to="/payer/queue" className="inline-flex items-center gap-1.5 text-[13px] text-ink-3 hover:text-ink">
        <ArrowLeft size={13} /> Review queue
      </Link>

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="num text-2xs text-ink-3">{data.case_number}</span>
            <span className={`chip ${urgChip}`}>{urgLabel} urgency</span>
          </div>
          <h1 className="mt-1.5 text-2xl font-semibold">{data.diagnosis}</h1>
          <p className="mt-1 text-[13px] text-ink-2">
            {data.requested_treatment} · {data.features.dose_category} ·{' '}
            {data.features.frequency} · {data.features.route} · requested for{' '}
            {data.features.requested_duration_months} months
          </p>
        </div>
        <div className="flex items-center gap-6">
          <Metric label="Policy fit" value={data.policy_fit_score?.toFixed(3)} />
          <Metric label="Necessity" value={pct(data.necessity_score)} />
          <div className="text-right">
            <div className="eyebrow mb-1.5">Status</div>
            <Status value={data.status} />
          </div>
        </div>
      </div>

      {data.assignment_reason && (
        <Alert tone={data.assignment_was_reassigned ? 'review' : 'info'}>
          {data.assignment_reason}
        </Alert>
      )}

      <DecisionLedger
        criteria={data.criteria?.criteria}
        rationale={data.criteria?.rationale}
        necessityScore={data.necessity_score}
      />

      <div className="grid gap-5 lg:grid-cols-2">
        <AttributionRail explanation={data.explanation} />

        <div className="space-y-5">
          {open && (
            <Card eyebrow="Adjudication" title="Record your decision">
              {error && <div className="mb-3"><Alert onDismiss={() => setError(null)}>{error}</Alert></div>}
              <p className="mb-3 text-[13px] text-ink-2">
                The engine leans toward <span className="font-medium">{leaning}</span> at{' '}
                <span className="num">{pct(data.necessity_score)}</span> of weighted criteria.
                Deciding against that leaning is recorded as an override.
              </p>
              <Field label="Clinical rationale" hint="Written to the audit trail and shown to the hospital.">
                <textarea
                  className="input" rows={4} value={notes}
                  placeholder="Explain what you relied on."
                  onChange={(e) => setNotes(e.target.value)}
                />
              </Field>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <button onClick={() => decide('APPROVED')} disabled={busy || notes.length < 5} className="btn-approve">
                  <CheckCircle2 size={14} /> Approve
                </button>
                <button onClick={() => decide('DENIED')} disabled={busy || notes.length < 5} className="btn-deny">
                  <XCircle size={14} /> Deny
                </button>
              </div>
            </Card>
          )}

          {data.reviewer_notes && (
            <Card eyebrow="Recorded decision" title="Reviewer rationale">
              <p className="text-[13px] text-ink-2">{data.reviewer_notes}</p>
            </Card>
          )}

          <AppealForecast prediction={data.appeal_prediction} />

          {open && data.routing_candidates?.length > 0 && (
            <RoutingPanel
              candidates={data.routing_candidates}
              assignedId={data.assigned_reviewer?.id}
              onReassign={reassign}
              busy={busy}
            />
          )}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <SubmittedValues features={data.features} documents={data.documents} />
        <AuditTrail events={audit} />
      </div>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="text-right">
      <div className="eyebrow">{label}</div>
      <div className="num mt-1 text-lg font-semibold leading-none">{value ?? '—'}</div>
    </div>
  )
}

/* Shows why this reviewer was chosen and who the alternates are, with the
   score decomposed into its four factors. */
function RoutingPanel({ candidates, assignedId, onReassign, busy }) {
  return (
    <Card eyebrow="Explainable routing" title="Why this reviewer" bodyClass="p-0">
      <div className="divide-y divide-rule">
        {candidates.map((c) => {
          const blocked = !c.is_available || c.at_capacity
          return (
            <div key={c.reviewer_id} className="px-4 py-3">
              <div className="flex items-baseline justify-between gap-3">
                <div className="min-w-0">
                  <span className="text-[13px] font-medium">{c.name}</span>
                  {c.reviewer_id === assignedId && (
                    <span className="chip ml-2 border-payer-line bg-payer-soft text-payer-deep">assigned</span>
                  )}
                  {blocked && (
                    <span className="chip ml-2 border-review-line bg-review-soft text-review">
                      <PauseCircle size={10} />
                      {c.at_capacity ? 'at capacity' : 'paused'}
                    </span>
                  )}
                  <div className="text-2xs text-ink-3">
                    {c.specialty}
                    {c.specialty_match && ' · specialty match'} · {c.open_cases}/
                    {c.daily_capacity} open
                  </div>
                </div>
                <span className="num shrink-0 text-sm font-semibold">{c.score.toFixed(3)}</span>
              </div>

              <div className="mt-2 flex gap-3">
                {Object.entries(c.factors).map(([name, v]) => (
                  <div key={name} className="flex-1">
                    <div className="mb-1 flex justify-between text-2xs text-ink-3">
                      <span className="truncate">{name.replace(/_/g, ' ')}</span>
                      <span className="num">{v.toFixed(2)}</span>
                    </div>
                    <Meter value={v} max={0.45} tone="payer" />
                  </div>
                ))}
              </div>

              {c.reviewer_id !== assignedId && !blocked && (
                <button
                  onClick={() => onReassign(c.reviewer_id)} disabled={busy}
                  className="btn-ghost mt-2.5 h-7 text-2xs"
                >
                  <ArrowRightLeft size={11} /> Move case here
                </button>
              )}
            </div>
          )
        })}
      </div>
    </Card>
  )
}

export function Appeals() {
  const [rows, setRows] = useState(null)
  const [active, setActive] = useState(null)
  const [form, setForm] = useState({ status: 'UPHELD', outcome_notes: '' })
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = () => api.get('/api/review/appeals').then(setRows).catch(() => setRows([]))
  useEffect(() => { load() }, [])

  const resolve = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.post(`/api/review/appeals/${active.id}/resolve`, form)
      setActive(null)
      setForm({ status: 'UPHELD', outcome_notes: '' })
      load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!rows) return <Spinner label="Loading appeals" />

  return (
    <>
      <div className="space-y-5">
      <div>
        <div className="eyebrow">Insurance organization</div>
        <h1 className="mt-1 text-2xl font-semibold">Appeals</h1>
      </div>

      {rows.length === 0 ? (
        <Card bodyClass="p-0">
          <Empty icon={Inbox} title="No appeals filed">
            When a hospital challenges a denial it appears here alongside what the
            appeal-propensity model predicted at the time of the decision.
          </Empty>
        </Card>
      ) : (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card bodyClass="p-0" title={`${rows.length} appeal${rows.length === 1 ? '' : 's'}`}>
            <div className="divide-y divide-rule">
              {rows.map((a) => (
                <button
                  key={a.id} onClick={() => setActive(a)}
                  className={`block w-full px-4 py-3 text-left transition-colors ${
                    active?.id === a.id
                      ? 'bg-payer-soft'
                      : 'hover:bg-canvas'
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="num text-2xs text-ink-3">{a.case_number}</span>
                    <Status value={a.status} />
                  </div>
                  <div className="mt-1 text-[13px] font-medium">{a.diagnosis}</div>
                  <div className="text-2xs text-ink-3">
                    {a.requested_treatment}
                    {a.new_documentation && ' · new documentation submitted'}
                  </div>
                  {a.predicted_at_filing && (
                    <div className="mt-1.5 text-2xs text-ink-3">
                      Model predicted{' '}
                      <span className="num">{pct(a.predicted_at_filing.any_appeal_probability)}</span>{' '}
                      chance of appeal at decision time
                    </div>
                  )}
                </button>
              ))}
            </div>
          </Card>

          {active && (
            <Card eyebrow={active.case_number} title="Resolve this appeal">
              {error && <div className="mb-3"><Alert onDismiss={() => setError(null)}>{error}</Alert></div>}
              <div className="mb-4">
                <div className="eyebrow mb-1">Grounds submitted</div>
                <p className="text-[13px] text-ink-2">{active.rationale}</p>
              </div>

              {active.status !== 'OPEN' ? (
                <Alert tone={active.status === 'OVERTURNED' ? 'approve' : 'info'}>
                  Already resolved as {active.status.toLowerCase()}.
                  {active.outcome_notes && ` ${active.outcome_notes}`}
                </Alert>
              ) : (
                <>
                  <Field label="Outcome">
                    <select className="input" value={form.status}
                      onChange={(e) => setForm({ ...form, status: e.target.value })}>
                      <option value="UPHELD">Uphold the denial</option>
                      <option value="OVERTURNED">Overturn and approve</option>
                    </select>
                  </Field>
                  <div className="mt-4">
                    <Field label="Reasoning" hint="Written to the audit trail.">
                      <textarea className="input" rows={4} value={form.outcome_notes}
                        onChange={(e) => setForm({ ...form, outcome_notes: e.target.value })} />
                    </Field>
                  </div>
                  <button
                    onClick={resolve} disabled={busy || form.outcome_notes.length < 5}
                    className="btn mt-4 w-full bg-payer text-white border-payer hover:bg-payer-deep"
                  >
                    {busy ? 'Recording…' : 'Record outcome'}
                  </button>
                </>
              )}
            </Card>
          )}
        </div>
      )}
      </div>
    </>
  )
}

export function Reviewers() {
  const [rows, setRows] = useState(null)
  useEffect(() => {
    api.get('/api/review/reviewers').then(setRows).catch(() => setRows([]))
  }, [])

  if (!rows) return <Spinner label="Loading reviewers" />

  return (
    <>
      <div className="space-y-5">
      <div>
        <div className="eyebrow">Insurance organization</div>
        <h1 className="mt-1 text-2xl font-semibold">Reviewer roster</h1>
        <p className="mt-1.5 text-[13px] text-ink-2">
          Everyone registered on the payer portal. Specialty and capacity determine
          where cases route; pausing yourself moves new cases to an alternate.
        </p>
      </div>

      {rows.length === 0 ? (
        <Card bodyClass="p-0">
          <Empty icon={Users} title="No reviewers registered" />
        </Card>
      ) : (
        <Card bodyClass="p-0">
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">Reviewer</th>
                <th className="th">Specialty</th>
                <th className="th">Availability</th>
                <th className="th w-52">Load</th>
                <th className="th text-right">Open / capacity</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td className="td">
                    <div className="font-medium">
                      {r.name}
                      {r.is_self && <span className="chip ml-2 border-rule bg-canvas text-ink-3">you</span>}
                    </div>
                    <div className="text-2xs text-ink-3">{r.email}</div>
                  </td>
                  <td className="td">{r.specialty}</td>
                  <td className="td">
                    {r.is_available ? (
                      <span className="chip border-approve-line bg-approve-soft text-approve">taking cases</span>
                    ) : (
                      <span className="chip border-review-line bg-review-soft text-review">
                        {r.unavailable_reason || 'paused'}
                      </span>
                    )}
                  </td>
                  <td className="td">
                    <Meter
                      value={r.open_cases} max={r.daily_capacity}
                      tone={r.open_cases >= r.daily_capacity ? 'deny' : 'payer'}
                    />
                  </td>
                  <td className="td num text-right">
                    {r.open_cases} / {r.daily_capacity}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      </div>
    </>
  )
}