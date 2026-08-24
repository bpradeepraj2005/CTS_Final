import { AlertCircle, Check, Info, Loader2, X } from 'lucide-react'

/* ---- status vocabulary ---------------------------------------------------
   One word per state, used identically everywhere it appears. */
const STATUS = {
  SUBMITTED: ['Submitted', 'bg-info-soft border-info-line text-info'],
  AUTO_APPROVED: ['Auto-approved', 'bg-approve-soft border-approve-line text-approve'],
  APPROVED: ['Approved', 'bg-approve-soft border-approve-line text-approve'],
  AUTO_DENIED: ['Auto-denied', 'bg-deny-soft border-deny-line text-deny'],
  DENIED: ['Denied', 'bg-deny-soft border-deny-line text-deny'],
  PENDING_REVIEW: ['In review', 'bg-review-soft border-review-line text-review'],
  APPEALED: ['Appealed', 'bg-payer-soft border-payer-line text-payer-deep'],
  OPEN: ['Open', 'bg-review-soft border-review-line text-review'],
  UPHELD: ['Upheld', 'bg-deny-soft border-deny-line text-deny'],
  OVERTURNED: ['Overturned', 'bg-approve-soft border-approve-line text-approve'],
}

export function Status({ value }) {
  const [label, cls] = STATUS[value] || [value, 'bg-canvas border-rule text-ink-2']
  return <span className={`chip ${cls}`}>{label}</span>
}

export function Card({ title, eyebrow, action, children, className = '', bodyClass = 'p-4' }) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="card-head">
          <div>
            {eyebrow && <div className="eyebrow mb-0.5">{eyebrow}</div>}
            {title && <h3 className="text-sm font-semibold">{title}</h3>}
          </div>
          {action}
        </header>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  )
}

export function Kpi({ label, value, unit, sub, tone = 'ink' }) {
  const tones = {
    ink: 'text-ink',
    approve: 'text-approve',
    deny: 'text-deny',
    review: 'text-review',
    provider: 'text-provider-deep',
    payer: 'text-payer-deep',
  }
  return (
    <div className="card p-4">
      <div className="eyebrow">{label}</div>
      <div className={`mt-2 flex items-baseline gap-1 ${tones[tone]}`}>
        <span className="num text-[26px] font-semibold leading-none tracking-tight">{value}</span>
        {unit && <span className="num text-sm text-ink-3">{unit}</span>}
      </div>
      {sub && <div className="mt-1.5 text-2xs text-ink-3">{sub}</div>}
    </div>
  )
}

export function Empty({ icon: Icon = Info, title, children, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
      <div className="grid h-11 w-11 place-items-center rounded-full border border-rule bg-canvas">
        <Icon size={18} className="text-ink-3" strokeWidth={1.75} />
      </div>
      <div>
        <h4 className="text-sm font-semibold">{title}</h4>
        {children && <p className="mx-auto mt-1 max-w-sm text-[13px] text-ink-2">{children}</p>}
      </div>
      {action}
    </div>
  )
}

export function Spinner({ label = 'Loading' }) {
  return (
    <div className="flex items-center justify-center gap-2 py-14 text-ink-3">
      <Loader2 size={16} className="animate-spin" />
      <span className="text-[13px]">{label}</span>
    </div>
  )
}

export function Alert({ tone = 'deny', children, onDismiss }) {
  const tones = {
    deny: 'bg-deny-soft border-deny-line text-deny',
    approve: 'bg-approve-soft border-approve-line text-approve',
    review: 'bg-review-soft border-review-line text-review',
    info: 'bg-info-soft border-info-line text-info',
  }
  const Icon = tone === 'approve' ? Check : tone === 'info' ? Info : AlertCircle
  return (
    <div className={`flex items-start gap-2.5 rounded-md border px-3 py-2.5 text-[13px] ${tones[tone]}`}>
      <Icon size={15} className="mt-0.5 shrink-0" />
      <div className="flex-1">{children}</div>
      {onDismiss && (
        <button onClick={onDismiss} className="shrink-0 opacity-60 hover:opacity-100">
          <X size={14} />
        </button>
      )}
    </div>
  )
}

export function Field({ label, hint, error, children }) {
  return (
    <div>
      <label className="label">{label}</label>
      {children}
      {hint && !error && <p className="mt-1 text-2xs text-ink-3">{hint}</p>}
      {error && <p className="mt-1 text-2xs text-deny">{error}</p>}
    </div>
  )
}

export function Meter({ value, tone = 'ink', max = 1 }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  const tones = {
    ink: 'bg-ink', approve: 'bg-approve', deny: 'bg-deny',
    review: 'bg-review', provider: 'bg-provider', payer: 'bg-payer',
  }
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-rule">
      <div className={`h-full rounded-full ${tones[tone]}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export const pct = (v, digits = 0) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`

export const fmtDate = (iso) =>
  iso
    ? new Date(iso).toLocaleString(undefined, {
        day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
      })
    : '—'