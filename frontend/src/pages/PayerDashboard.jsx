import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Inbox, ArrowRight, Circle, Loader2 } from "lucide-react";

import {
  DARK_TONE,
  accentTone,
  darkAxis,
  darkGrid,
  darkTooltip,
  themeFor,
} from "../lib/portalTheme";

/* ============================================================
   PAYER DASHBOARD — DARK / BLUE

   Every surface is a tinted dark panel; nothing on this page
   renders on white. The page background itself is painted by
   Layout for the `/payer` route, so this file only draws the
   panels that sit on top of it.
============================================================ */

const T = themeFor("payer");

/* ============================== TONES ============================== */

const TONE = {
  payer: accentTone(T),
  review: DARK_TONE.info,
  approve: DARK_TONE.approve,
  deny: DARK_TONE.deny,
  warn: DARK_TONE.warn,
  neutral: DARK_TONE.neutral,
};

const grad = (tone) =>
  `linear-gradient(
    90deg,
    ${TONE[tone].from},
    ${TONE[tone].to}
  )`;

/* ============================== UI PRIMITIVES ============================== */

const PANEL = {
  background: T.panel,
  borderColor: T.line,
  boxShadow: "0 20px 50px -34px rgba(0,0,0,0.9)",
};

function Card({
  title,
  eyebrow,
  tone = "neutral",
  bodyClass = "",
  className = "",
  children,
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border transition-shadow ${className}`}
      style={PANEL}
    >
      {/* Tone rail */}
      <div
        className="h-[3px] w-full"
        style={{
          backgroundImage: grad(tone),
        }}
      />

      {(title || eyebrow) && (
        <div className="flex items-baseline justify-between gap-3 px-5 pt-4">
          <h3
            className="text-[13.5px] font-semibold"
            style={{ color: T.text }}
          >
            {title}
          </h3>

          {eyebrow && (
            <span
              className="text-[11px] font-medium uppercase tracking-wide"
              style={{ color: T.text3 }}
            >
              {eyebrow}
            </span>
          )}
        </div>
      )}

      <div className={bodyClass || "px-5 pb-5 pt-3"}>
        {children}
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  unit,
  sub,
  tone = "neutral",
  live = false,
}) {
  const t = TONE[tone];

  return (
    <div
      className="relative overflow-hidden rounded-2xl border p-4"
      style={PANEL}
    >
      <div
        className="absolute inset-x-0 top-0 h-[3px]"
        style={{
          backgroundImage: grad(tone),
        }}
      />

      {/* Tone wash */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: `radial-gradient(420px 160px at 100% 0%, ${t.soft}, transparent 72%)`,
        }}
      />

      <div className="relative">
        <div className="flex items-center gap-1.5">
          <span
            className="text-[12.5px] font-medium"
            style={{ color: T.text3 }}
          >
            {label}
          </span>

          {live && (
            <span className="relative flex h-1.5 w-1.5">
              <span
                className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
                style={{
                  backgroundColor: t.text,
                }}
              />

              <span
                className="relative inline-flex h-1.5 w-1.5 rounded-full"
                style={{
                  backgroundColor: t.text,
                }}
              />
            </span>
          )}
        </div>

        <div className="mt-1.5 flex items-baseline gap-1">
          <span
            className="font-mono text-[26px] font-semibold tracking-tight"
            style={{ color: t.text }}
          >
            {value}
          </span>

          {unit && (
            <span
              className="text-[13px] font-medium"
              style={{ color: T.text3 }}
            >
              {unit}
            </span>
          )}
        </div>

        {sub && (
          <p
            className="mt-1 text-[12px] leading-snug"
            style={{ color: T.text3 }}
          >
            {sub}
          </p>
        )}
      </div>
    </div>
  );
}

function Meter({
  value,
  max,
  tone = "neutral",
}) {
  const width =
    max > 0
      ? Math.min(100, (value / max) * 100)
      : 0;

  return (
    <div
      className="h-[6px] w-full overflow-hidden rounded-full"
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
  );
}

function Empty({
  icon: Icon = Inbox,
  title,
  children,
}) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-16 text-center">
      <div
        className="flex h-12 w-12 items-center justify-center rounded-full border"
        style={{
          background: T.accentSoft,
          borderColor: T.accentLine,
          color: T.accent,
        }}
      >
        <Icon
          size={20}
          strokeWidth={1.75}
        />
      </div>

      <h3
        className="text-[14.5px] font-semibold"
        style={{ color: T.text }}
      >
        {title}
      </h3>

      <p
        className="max-w-sm text-[13px] leading-relaxed"
        style={{ color: T.text2 }}
      >
        {children}
      </p>
    </div>
  );
}

function Spinner({ label }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-24">
      <Loader2
        size={26}
        className="animate-spin"
        style={{ color: T.accent }}
      />

      <p
        className="text-[13px]"
        style={{ color: T.text2 }}
      >
        {label}
      </p>
    </div>
  );
}

/* ============================== CHART STYLING ============================== */

const AXIS = darkAxis(T);

const TOOLTIP = darkTooltip(T);

const URGENCY_COLORS = [
  TONE.deny.text,
  TONE.warn.text,
  TONE.review.text,
  TONE.neutral.text,
];

/* ============================== MOCK DATA ============================== */

function fetchDashboardData() {
  const trend = Array.from(
    { length: 14 },
    (_, i) => {
      const d = new Date(2026, 7, 10 + i);

      const submitted =
        28 +
        Math.round(
          Math.sin(i / 2) * 6 +
            Math.random() * 4
        );

      const denied =
        4 +
        Math.round(Math.random() * 3);

      return {
        date: d.toISOString().slice(0, 10),
        submitted,
        approved:
          submitted -
          denied -
          Math.round(Math.random() * 3),
        denied,
      };
    }
  );

  return new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          empty: false,

          kpis: {
            total_requests: 4218,
            instant_decision_rate: 61,
            under_5s_rate: 47,
            pending_review: 312,
            unassigned: 58,
            my_queue: 24,
            auto_reassigned: 96,
            reviewer_override_rate: 12,
            open_appeals: 19,
            appeal_overturn_rate: 34,
            active_reviewers: 18,
            total_reviewers: 26,
          },

          trend,

          urgency_bands: [
            {
              band: "Emergent",
              count: 22,
            },
            {
              band: "Urgent",
              count: 64,
            },
            {
              band: "Routine",
              count: 158,
            },
            {
              band: "Low",
              count: 68,
            },
          ],

          reviewer_load: [
            {
              name: "Dr. Amara Okonkwo",
              specialty: "Cardiology",
              open_cases: 14,
              daily_capacity: 18,
              utilization: 78,
              is_available: true,
            },
            {
              name: "Dr. Priya Nair",
              specialty: "Oncology",
              open_cases: 20,
              daily_capacity: 20,
              utilization: 100,
              is_available: true,
            },
            {
              name: "Dr. Marcus Webb",
              specialty: "Orthopedics",
              open_cases: 6,
              daily_capacity: 16,
              utilization: 38,
              is_available: false,
            },
            {
              name: "Dr. Elena Vasquez",
              specialty: "Neurology",
              open_cases: 11,
              daily_capacity: 15,
              utilization: 73,
              is_available: true,
            },
          ],

          top_denial_reasons: [
            {
              code: "MED-NEC",
              label:
                "Medical necessity not established",
              count: 86,
            },
            {
              code: "PRIOR-STEP",
              label:
                "Step therapy not attempted",
              count: 61,
            },
            {
              code: "DOC-INC",
              label:
                "Incomplete clinical documentation",
              count: 44,
            },
            {
              code: "OUT-NET",
              label:
                "Out-of-network provider",
              count: 27,
            },
          ],
        }),
      500
    )
  );
}

/* ============================== PAGE ============================== */

export default function PayerDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

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
    );
  }

  /* ============================================================
     LOADING
  ============================================================ */

  if (!data) {
    return <Spinner label="Loading review activity" />;
  }

  /* ============================================================
     EMPTY STATE
  ============================================================ */

  if (data.empty) {
    return (
      <Card bodyClass="p-0" tone="payer">
        <Empty
          icon={Inbox}
          title="No cases have been submitted yet"
        >
          This dashboard aggregates live case data.
          Once a hospital account files a request, the
          queue, routing and override figures populate
          from it.
        </Empty>
      </Card>
    );
  }

  const k = data.kpis;

  /* ============================================================
     MAIN DASHBOARD
  ============================================================ */

  return (
    <div className="mx-auto max-w-7xl space-y-5">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <div
        className="relative overflow-hidden rounded-2xl border px-6 py-6"
        style={{
          ...PANEL,
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
                rgba(96,165,250,0.20),
                transparent 70%
              ),
              radial-gradient(
                620px 220px at 92% 100%,
                rgba(37,99,235,0.24),
                transparent 70%
              )
            `,
          }}
        />

        <div className="relative flex flex-wrap items-end justify-between gap-4">
          <div>
            <div
              className="text-[11px] font-semibold uppercase tracking-wide"
              style={{ color: T.accent }}
            >
              Insurance organization
            </div>

            <h1
              className="mt-1 text-[26px] font-semibold tracking-tight"
              style={{ color: T.text }}
            >
              Review operations
            </h1>

            <p
              className="mt-1 text-[13px]"
              style={{ color: T.text2 }}
            >
              Live view of intake, routing and reviewer decisions.
            </p>
          </div>

          <a
            href="/payer/queue"
            className="group inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-[13.5px] font-semibold text-white transition-transform hover:scale-[1.02]"
            style={{
              backgroundImage: `linear-gradient(135deg, ${T.accent}, ${T.accentDeep})`,
              boxShadow: `0 12px 26px -10px ${T.accentGlow}`,
            }}
          >
            Open my queue ({k.my_queue})

            <ArrowRight
              size={15}
              className="transition-transform group-hover:translate-x-0.5"
            />
          </a>
        </div>
      </div>

      {/* ======================================================
          KPI ROW 1
      ====================================================== */}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi
          label="Cases received"
          value={k.total_requests.toLocaleString()}
          tone="neutral"
        />

        <Kpi
          label="Cleared without a human"
          value={k.instant_decision_rate}
          unit="%"
          tone="approve"
          sub={`${k.under_5s_rate}% decided under 5s`}
        />

        <Kpi
          label="Waiting on a reviewer"
          value={k.pending_review}
          tone="review"
          sub={`${k.unassigned} unassigned`}
        />

        <Kpi
          label="In my queue"
          value={k.my_queue}
          tone="payer"
          live
        />
      </div>

      {/* ======================================================
          KPI ROW 2
      ====================================================== */}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi
          label="Auto-reassigned"
          value={k.auto_reassigned}
          tone="warn"
          sub="First-choice reviewer was unavailable or full"
        />

        <Kpi
          label="Reviewer override rate"
          value={k.reviewer_override_rate}
          unit="%"
          tone="warn"
          sub="Decisions against the engine's leaning"
        />

        <Kpi
          label="Open appeals"
          value={k.open_appeals}
          tone="deny"
          sub={`${k.appeal_overturn_rate}% of appeals overturned`}
        />

        <Kpi
          label="Reviewers taking cases"
          value={k.active_reviewers}
          tone="neutral"
          unit={`/ ${k.total_reviewers}`}
        />
      </div>

      {/* ======================================================
          CHARTS
      ====================================================== */}

      <div className="grid gap-5 lg:grid-cols-3">

        {/* Case volume */}
        <Card
          title="Case volume and outcomes"
          eyebrow="Last 14 days"
          tone="payer"
          className="lg:col-span-2"
        >
          <ResponsiveContainer
            width="100%"
            height={230}
          >
            <LineChart
              data={data.trend}
              margin={{
                top: 4,
                right: 8,
                left: -18,
                bottom: 0,
              }}
            >
              <CartesianGrid
                stroke={darkGrid}
                vertical={false}
              />

              <XAxis
                dataKey="date"
                tick={AXIS}
                tickLine={false}
                axisLine={{
                  stroke: T.line,
                }}
                tickFormatter={(d) =>
                  d.slice(5)
                }
              />

              <YAxis
                tick={AXIS}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />

              <Tooltip
                contentStyle={TOOLTIP.contentStyle}
                labelStyle={TOOLTIP.labelStyle}
                itemStyle={TOOLTIP.itemStyle}
                cursor={{
                  stroke: T.lineStrong,
                }}
              />

              <Line
                type="monotone"
                dataKey="submitted"
                stroke={TONE.neutral.text}
                strokeWidth={2}
                dot={false}
                name="Received"
              />

              <Line
                type="monotone"
                dataKey="approved"
                stroke={TONE.approve.text}
                strokeWidth={2.25}
                dot={false}
                name="Approved"
              />

              <Line
                type="monotone"
                dataKey="denied"
                stroke={TONE.deny.text}
                strokeWidth={2}
                dot={false}
                name="Denied"
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Urgency */}
        <Card
          title="Queue by clinical urgency"
          eyebrow="Pending cases"
          tone="deny"
        >
          <ResponsiveContainer
            width="100%"
            height={230}
          >
            <BarChart
              data={data.urgency_bands}
              layout="vertical"
              margin={{
                top: 0,
                right: 16,
                left: 8,
                bottom: 0,
              }}
            >
              <CartesianGrid
                stroke={darkGrid}
                horizontal={false}
              />

              <XAxis
                type="number"
                tick={AXIS}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />

              <YAxis
                type="category"
                dataKey="band"
                width={78}
                tick={{
                  fontSize: 11.5,
                  fill: T.text2,
                }}
                tickLine={false}
                axisLine={false}
              />

              <Tooltip
                contentStyle={TOOLTIP.contentStyle}
                labelStyle={TOOLTIP.labelStyle}
                itemStyle={TOOLTIP.itemStyle}
                cursor={{
                  fill: "rgba(255,255,255,0.05)",
                }}
              />

              <Bar
                dataKey="count"
                radius={[0, 6, 6, 0]}
                barSize={16}
              >
                {data.urgency_bands.map(
                  (b, i) => (
                    <Cell
                      key={b.band}
                      fill={URGENCY_COLORS[i]}
                    />
                  )
                )}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* ======================================================
          REVIEWER LOAD + DENIAL DRIVERS
      ====================================================== */}

      <div className="grid gap-5 lg:grid-cols-2">

        {/* Reviewer Load */}
        <Card
          title="Reviewer load"
          eyebrow="Live utilization"
          tone="review"
          bodyClass="p-0"
        >
          {data.reviewer_load.length === 0 ? (
            <p
              className="px-4 py-10 text-center text-[13px]"
              style={{ color: T.text3 }}
            >
              No reviewers registered.
            </p>
          ) : (
            <div>
              {data.reviewer_load.map((r) => (
                <div
                  key={r.name}
                  className="border-t px-5 py-3.5 first:border-t-0"
                  style={{ borderColor: T.line }}
                >
                  <div className="mb-1.5 flex items-baseline justify-between gap-3">
                    <div className="min-w-0">
                      <span
                        className="text-[13.5px] font-medium"
                        style={{ color: T.text }}
                      >
                        {r.name}
                      </span>

                      <span
                        className="ml-2 text-[11px]"
                        style={{ color: T.text3 }}
                      >
                        {r.specialty}
                      </span>
                    </div>

                    <span
                      className="shrink-0 font-mono text-[11.5px]"
                      style={{ color: T.text3 }}
                    >
                      {r.open_cases} / {r.daily_capacity}
                    </span>
                  </div>

                  <Meter
                    value={r.open_cases}
                    max={r.daily_capacity}
                    tone={
                      !r.is_available
                        ? "review"
                        : r.utilization > 80
                        ? "deny"
                        : "payer"
                    }
                  />

                  {!r.is_available && (
                    <p
                      className="mt-1.5 flex items-center gap-1 text-[11.5px]"
                      style={{ color: TONE.review.text }}
                    >
                      <Circle
                        size={6}
                        className="fill-current"
                      />
                      Paused — not receiving new cases
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Denial Drivers */}
        <Card
          title="Most common denial drivers"
          eyebrow="Failed criteria"
          tone="deny"
        >
          {data.top_denial_reasons.length === 0 ? (
            <p
              className="py-10 text-center text-[13px]"
              style={{ color: T.text3 }}
            >
              No denials yet.
            </p>
          ) : (
            <div className="space-y-3.5">
              {data.top_denial_reasons.map(
                (r) => (
                  <div key={r.code}>
                    <div className="mb-1 flex items-baseline justify-between gap-3">
                      <span
                        className="truncate text-[13px]"
                        style={{ color: T.text }}
                      >
                        {r.label}
                      </span>

                      <span
                        className="font-mono text-[11.5px]"
                        style={{ color: T.text3 }}
                      >
                        {r.count}
                      </span>
                    </div>

                    <Meter
                      value={r.count}
                      max={
                        data
                          .top_denial_reasons[0]
                          .count
                      }
                      tone="deny"
                    />
                  </div>
                )
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
