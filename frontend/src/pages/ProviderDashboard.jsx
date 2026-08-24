import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  FileCheck2,
  Inbox,
  Loader2,
  Plus,
  ShieldAlert,
  UserCheck2,
} from 'lucide-react'

import { api } from '../lib/api'
import {
  DARK_TONE,
  accentTone,
  themeFor,
} from '../lib/portalTheme'

/* ============================================================
   HOSPITAL DASHBOARD — DARK / VIOLET

   Every surface on this page is a tinted dark panel. The shared
   light primitives in components/ui.jsx (Card, Meter, Empty,
   Spinner) are deliberately not used here — they render on a
   white `.card` surface that the rest of the app still depends
   on. The dark equivalents live at the bottom of this file.

   The page background itself is painted by Layout for the
   dashboard routes, so this page only draws panels on top.
============================================================ */

const T = themeFor('provider')

const TONE = {
  violet: accentTone(T),
  approve: DARK_TONE.approve,
  warn: DARK_TONE.warn,
  deny: DARK_TONE.deny,
  info: DARK_TONE.info,
  neutral: DARK_TONE.neutral,
}

const grad = (tone) =>
  `linear-gradient(90deg, ${TONE[tone].from}, ${TONE[tone].to})`

export default function ProviderDashboard() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api
      .get('/api/dashboard/provider')
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  /* ============================================================
     ERROR
  ============================================================ */

  if (error) {
    return (
      <Card tone="deny">
        <p
          className="text-[13px]"
          style={{ color: TONE.deny.text }}
        >
          {error}
        </p>
      </Card>
    )
  }

  /* ============================================================
     LOADING
  ============================================================ */

  if (!data) {
    return <Spinner label="Loading authorization activity" />
  }

  /* ============================================================
     EMPTY STATE
  ============================================================ */

  if (data.empty) {
    return (
      <div className="space-y-5">
        <Header />

        <Card bodyClass="p-0" tone="violet">
          <Empty
            icon={Inbox}
            title="No authorization requests yet"
          >
            Upload your first PA packet to start the automated
            medical-necessity workflow.
          </Empty>
        </Card>
      </div>
    )
  }

  const k = data.kpis

  /* ============================================================
     MAIN DASHBOARD
  ============================================================ */

  return (
    <div className="space-y-5">

      {/* =================================================
          HEADER
      ================================================= */}

      <Header
        subtitle="Live performance across your prior-authorization requests."
      />

      {/* =================================================
          KPI ROW
      ================================================= */}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">

        <GradientKpi
          icon={Inbox}
          tone="violet"
          label="Requests submitted"
          value={k.total_requests}
          sub={`${k.instant_decision_rate}% decided instantly`}
        />

        <GradientKpi
          icon={FileCheck2}
          tone="approve"
          label="Approval rate"
          value={`${k.approval_rate}%`}
          sub={`${k.denial_rate}% denied`}
        />

        <GradientKpi
          icon={UserCheck2}
          tone="warn"
          label="Pending human review"
          value={k.pending_review}
          sub={`${k.avg_processing_ms}ms avg · ${k.p95_processing_ms}ms p95`}
        />

        <GradientKpi
          icon={ShieldAlert}
          tone="info"
          label="Documents processed"
          value={k.documents_processed}
          sub={`${(
            k.avg_extraction_confidence * 100
          ).toFixed(1)}% avg extraction confidence`}
        />

      </div>

      {/* =================================================
          SECONDARY CARDS
      ================================================= */}

      <div className="grid gap-4 lg:grid-cols-3">

        {/* Appeal Risk */}

        <Card
          title="Appeal risk"
          eyebrow="Denied requests"
          tone="deny"
          className="lg:col-span-1"
        >
          <div
            className="num text-3xl font-semibold"
            style={{ color: T.text }}
          >
            {(
              k.mean_appeal_risk_on_denials * 100
            ).toFixed(1)}
            %
          </div>

          <p
            className="mt-1.5 text-[13px] leading-relaxed"
            style={{ color: T.text2 }}
          >
            Mean predicted probability that a denied request will be
            challenged.
          </p>

          <div className="mt-4">
            <Meter
              value={k.mean_appeal_risk_on_denials}
              tone="deny"
            />
          </div>
        </Card>

        {/* Top Denial Reasons */}

        <Card
          title="Top denial reasons"
          eyebrow="Medical-necessity criteria"
          tone="warn"
          className="lg:col-span-1"
        >
          {data.top_denial_reasons.length === 0 ? (
            <p
              className="py-8 text-center text-[13px]"
              style={{ color: T.text3 }}
            >
              No denial criteria recorded yet.
            </p>
          ) : (
            <div className="space-y-3">
              {data.top_denial_reasons.map((r) => (
                <div key={r.code}>
                  <div className="mb-1 flex justify-between gap-3">
                    <span
                      className="text-[13px]"
                      style={{ color: T.text }}
                    >
                      {r.label}
                    </span>

                    <span
                      className="num text-2xs"
                      style={{ color: T.text3 }}
                    >
                      {r.count}
                    </span>
                  </div>

                  <Meter
                    value={r.count}
                    max={
                      data.top_denial_reasons[0].count
                    }
                    tone="deny"
                  />
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Requests By Specialty */}

        <Card
          title="Requests by specialty"
          eyebrow="Clinical mix"
          tone="violet"
          className="lg:col-span-1"
        >
          {data.by_specialty.length === 0 ? (
            <p
              className="py-8 text-center text-[13px]"
              style={{ color: T.text3 }}
            >
              No specialty data yet.
            </p>
          ) : (
            <div className="space-y-3">
              {data.by_specialty.map((r) => (
                <div key={r.name}>
                  <div className="mb-1 flex justify-between gap-3">
                    <span
                      className="text-[13px]"
                      style={{ color: T.text }}
                    >
                      {r.name}
                    </span>

                    <span
                      className="num text-2xs"
                      style={{ color: T.text3 }}
                    >
                      {r.count}
                    </span>
                  </div>

                  <Meter
                    value={r.count}
                    max={data.by_specialty[0].count}
                    tone="violet"
                  />
                </div>
              ))}
            </div>
          )}
        </Card>

      </div>

      {/* =================================================
          RECENT ACTIVITY
      ================================================= */}

      <Card
        title="Recent 14-day activity"
        eyebrow="Live case trend"
        tone="violet"
        bodyClass="p-0"
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Date</Th>
                <Th align="right">Submitted</Th>
                <Th align="right">Approved</Th>
                <Th align="right">Denied</Th>
              </tr>
            </thead>

            <tbody>
              {data.trend.map((r) => (
                <Tr key={r.date}>
                  <Td>{r.date}</Td>

                  <Td align="right">
                    {r.submitted}
                  </Td>

                  <Td
                    align="right"
                    color={TONE.approve.text}
                  >
                    {r.approved}
                  </Td>

                  <Td
                    align="right"
                    color={TONE.deny.text}
                  >
                    {r.denied}
                  </Td>
                </Tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

    </div>
  )
}

/* ============================================================
   HEADER
============================================================ */

function Header({ subtitle }) {
  return (
    <div
      className="relative overflow-hidden rounded-2xl border px-6 py-6"
      style={{
        background: T.panel,
        borderColor: T.line,
        boxShadow: `0 24px 60px -40px ${T.accentGlow}`,
      }}
    >
      {/* Accent glow */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `
            radial-gradient(
              700px 220px at 12% 0%,
              rgba(167,139,250,0.20),
              transparent 70%
            ),
            radial-gradient(
              620px 220px at 92% 100%,
              rgba(124,58,237,0.22),
              transparent 70%
            )
          `,
        }}
      />

      <div className="relative flex flex-wrap items-end justify-between gap-4">
        <div>
          <div
            className="font-mono text-2xs font-semibold uppercase tracking-[0.14em]"
            style={{ color: T.accent }}
          >
            Hospital management
          </div>

          <h1
            className="mt-1.5 text-[26px] font-semibold tracking-tight"
            style={{ color: T.text }}
          >
            Authorization operations
          </h1>

          <p
            className="mt-1.5 text-[13px]"
            style={{ color: T.text2 }}
          >
            {subtitle ||
              'Submit a prior-authorization packet and the platform will extract, score, adjudicate and route it.'}
          </p>
        </div>

        <Link
          to="/hospital/new"
          className="group inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-[13.5px] font-semibold text-white transition-transform hover:scale-[1.02]"
          style={{
            backgroundImage: `linear-gradient(135deg, ${T.accent}, ${T.accentDeep})`,
            boxShadow: `0 12px 26px -10px ${T.accentGlow}`,
          }}
        >
          <Plus size={15} />
          New request
        </Link>
      </div>
    </div>
  )
}

/* ============================================================
   DARK PRIMITIVES
============================================================ */

function Card({
  title,
  eyebrow,
  tone = 'violet',
  className = '',
  bodyClass = '',
  children,
}) {
  return (
    <section
      className={`relative overflow-hidden rounded-2xl border ${className}`}
      style={{
        background: T.panel,
        borderColor: T.line,
        boxShadow: '0 20px 50px -34px rgba(0,0,0,0.9)',
      }}
    >
      {/* Tone rail */}
      <div
        className="h-[3px] w-full"
        style={{ backgroundImage: grad(tone) }}
      />

      {(title || eyebrow) && (
        <div className="flex items-baseline justify-between gap-3 px-5 pt-4">
          {title && (
            <h3
              className="text-[13.5px] font-semibold"
              style={{ color: T.text }}
            >
              {title}
            </h3>
          )}

          {eyebrow && (
            <span
              className="font-mono text-2xs uppercase tracking-[0.13em]"
              style={{ color: T.text3 }}
            >
              {eyebrow}
            </span>
          )}
        </div>
      )}

      <div className={bodyClass || 'px-5 pb-5 pt-3'}>
        {children}
      </div>
    </section>
  )
}

function GradientKpi({
  icon: Icon,
  tone,
  label,
  value,
  sub,
}) {
  const t = TONE[tone]

  return (
    <div
      className="relative overflow-hidden rounded-2xl border p-4 transition-transform duration-200 hover:-translate-y-0.5"
      style={{
        background: T.panel,
        borderColor: T.line,
        boxShadow: '0 20px 50px -34px rgba(0,0,0,0.9)',
      }}
    >
      {/* Tone wash */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `radial-gradient(420px 160px at 100% 0%, ${t.soft}, transparent 72%)`,
        }}
      />

      <div className="relative">
        <div
          className="grid h-9 w-9 place-items-center rounded-lg text-white"
          style={{
            backgroundImage: `linear-gradient(135deg, ${t.from}, ${t.to})`,
            boxShadow: `0 10px 22px -12px ${t.to}`,
          }}
        >
          <Icon size={16} />
        </div>

        <div
          className="mt-3 font-mono text-2xs uppercase tracking-[0.13em]"
          style={{ color: T.text3 }}
        >
          {label}
        </div>

        <div
          className="num mt-1 text-2xl font-semibold"
          style={{ color: t.text }}
        >
          {value}
        </div>

        {sub && (
          <div
            className="mt-0.5 text-[11px]"
            style={{ color: T.text3 }}
          >
            {sub}
          </div>
        )}
      </div>
    </div>
  )
}

function Meter({ value, max = 1, tone = 'violet' }) {
  const width =
    max > 0
      ? Math.max(0, Math.min(100, (value / max) * 100))
      : 0

  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full"
      style={{ background: T.track }}
    >
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{
          width: `${width}%`,
          backgroundImage: grad(tone),
        }}
      />
    </div>
  )
}

function Empty({ icon: Icon = Inbox, title, children }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      <div
        className="grid h-12 w-12 place-items-center rounded-full border"
        style={{
          background: T.accentSoft,
          borderColor: T.accentLine,
          color: T.accent,
        }}
      >
        <Icon size={20} strokeWidth={1.75} />
      </div>

      <h3
        className="text-[14.5px] font-semibold"
        style={{ color: T.text }}
      >
        {title}
      </h3>

      {children && (
        <p
          className="max-w-sm text-[13px] leading-relaxed"
          style={{ color: T.text2 }}
        >
          {children}
        </p>
      )}
    </div>
  )
}

function Spinner({ label = 'Loading' }) {
  return (
    <div className="flex items-center justify-center gap-2 py-24">
      <Loader2
        size={16}
        className="animate-spin"
        style={{ color: T.accent }}
      />

      <span
        className="text-[13px]"
        style={{ color: T.text2 }}
      >
        {label}
      </span>
    </div>
  )
}

/* ---------- table ---------- */

function Th({ children, align = 'left' }) {
  return (
    <th
      className="border-b px-4 py-3 font-mono text-2xs font-semibold uppercase tracking-[0.1em]"
      style={{
        borderColor: T.line,
        background: 'rgba(255,255,255,0.03)',
        color: T.text3,
        textAlign: align,
      }}
    >
      {children}
    </th>
  )
}

function Tr({ children }) {
  const [hot, setHot] = useState(false)

  return (
    <tr
      onMouseEnter={() => setHot(true)}
      onMouseLeave={() => setHot(false)}
      style={{
        background: hot ? T.accentSoft : 'transparent',
        transition: 'background 0.18s ease',
      }}
    >
      {children}
    </tr>
  )
}

function Td({ children, align = 'left', color }) {
  return (
    <td
      className="num border-b px-4 py-3.5 align-middle"
      style={{
        borderColor: T.line,
        color: color || T.text2,
        textAlign: align,
      }}
    >
      {children}
    </td>
  )
}
