import { useEffect, useState } from "react";

import {
  AlertTriangle,
  ArrowLeft,
  Brain,
  CheckCircle2,
  FileText,
  FileWarning,
  Send,
  ShieldAlert,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import {
  Link,
  useNavigate,
  useParams,
} from "react-router-dom";

import { api } from "../lib/api";
import { humanizeAuditEvent } from '../lib/humanizeAudit'
import AIValidationCard from "../components/AIValidationCard";

import {
  Alert,
  Card,
  Empty,
  Field,
  Spinner,
  Status,
  fmtDate,
  pct,
} from "../components/ui";

import {
  AppealForecast,
  AttributionRail,
  DecisionLedger,
} from "../components/Explain";


/* The page background is painted once by Layout for the whole
   hospital portal, so these pages only draw panels on top. */


/* ============================================================
   REQUEST FILTERS
   ============================================================ */

const FILTERS = [
  ["", "All requests"],
  ["AUTO_APPROVED", "Auto-approved"],
  ["AUTO_DENIED", "Auto-denied"],
  ["PENDING_REVIEW", "In review"],
  ["APPROVED", "Approved"],
  ["DENIED", "Denied"],
  ["APPEALED", "Appealed"],
];


/* ============================================================
   REQUEST LIST
   ============================================================ */

export function RequestList() {
  const navigate = useNavigate();

  const [rows, setRows] = useState(null);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    setRows(null);
    setError(null);

    const query = filter
      ? `?status_filter=${encodeURIComponent(filter)}`
      : "";

    api
      .get(`/api/requests${query}`)
      .then(setRows)
      .catch((e) => {
        setError(e.message);
        setRows([]);
      });
  }, [filter]);

  if (!rows) {
    return <Spinner label="Loading requests" />;
  }

  return (
    <div>
      <div className="space-y-5">

      {/* PAGE HEADER */}
      <div className="relative overflow-hidden rounded-2xl border border-rule bg-gradient-to-br from-surface via-surface to-provider/15 px-6 py-6 shadow-card">

        {/* decorative accent wash */}
        <div className="pointer-events-none absolute -right-16 -top-16 h-56 w-56 rounded-full bg-gradient-to-br from-provider/40 via-provider/10 to-transparent blur-2xl" />

        <div className="relative flex flex-wrap items-end justify-between gap-4">

          <div>
            <div className="eyebrow flex items-center gap-1.5">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-gradient-to-br from-provider to-provider-deep" />
              Hospital management
            </div>

            <h1 className="mt-1 text-2xl font-semibold text-ink">
              Authorization requests
            </h1>

            <p className="mt-1.5 text-[13px] text-ink-2">
              Every submitted case, its decision source and
              the scores that supported the outcome.
            </p>
          </div>

          <Link
            to="/hospital/new"
            className="btn border-transparent bg-gradient-to-r from-provider to-provider-deep text-white shadow-sm shadow-provider/20 transition-all hover:shadow-md hover:shadow-provider/30 hover:brightness-105"
          >
            New request
          </Link>

        </div>

      </div>


      {/* ERROR */}
      {error && <Alert>{error}</Alert>}


      {/* FILTERS */}
      <div className="flex flex-wrap gap-1.5">

        {FILTERS.map(([value, label]) => (
          <button
            key={value}
            onClick={() => setFilter(value)}
            className={`btn transition-all duration-150 ${
              filter === value
                ? "border-transparent bg-gradient-to-r from-provider to-provider-deep text-white shadow-sm shadow-provider/30"
                : "btn-ghost hover:border-provider-line hover:bg-provider-soft"
            }`}
          >
            {label}
          </button>
        ))}

      </div>


      {/* REQUEST TABLE */}
      {rows.length === 0 ? (

        <Card bodyClass="p-0">

          <Empty
            icon={FileText}
            title="No requests found"
          >
            Submit a prior-authorization packet to
            populate this list.
          </Empty>

        </Card>

      ) : (

        <Card
          bodyClass="p-0"
          title={`${rows.length} request${
            rows.length === 1 ? "" : "s"
          }`}
        >

          <div className="overflow-x-auto">

            <table className="w-full">

              <thead>
                <tr>

                  <th className="th">
                    Case
                  </th>

                  <th className="th">
                    Diagnosis / therapy
                  </th>

                  <th className="th">
                    Status
                  </th>

                  <th className="th text-right">
                    Policy fit
                  </th>

                  <th className="th text-right">
                    Necessity
                  </th>

                  <th className="th text-right">
                    Appeal risk
                  </th>

                  <th className="th text-right">
                    Submitted
                  </th>

                </tr>
              </thead>


              <tbody>

                {rows.map((r) => (

                  <tr
                    key={r.id}
                    className="row-link cursor-pointer transition-colors duration-150"
                    onClick={() =>
                      navigate(
                        `/hospital/requests/${r.id}`
                      )
                    }
                  >

                    <td className="td">

                      <div className="num text-2xs">
                        {r.case_number}
                      </div>

                      <div className="mt-1 text-2xs text-ink-3">
                        {r.provider_specialty}
                      </div>

                    </td>


                    <td className="td">

                      <div className="font-medium">
                        {r.diagnosis}
                      </div>

                      <div className="text-2xs text-ink-3">
                        {r.requested_treatment} ·{" "}
                        {r.disease_severity}
                      </div>

                    </td>


                    <td className="td">
                      <Status value={r.status} />
                    </td>


                    <td className="td num text-right">
                      <ScoreBadge value={r.policy_fit_score} decimals={3} />
                    </td>


                    <td className="td num text-right">
                      <ScoreBadge value={r.necessity_score} format="pct" />
                    </td>


                    <td className="td num text-right">
                      <ScoreBadge
                        value={r.appeal_probability}
                        format="pct"
                        inverse
                      />
                    </td>


                    <td className="td text-right text-2xs text-ink-3">
                      {fmtDate(r.created_at)}
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>

        </Card>

      )}

      </div>
    </div>
  );
}


/* ============================================================
   SCORE BADGE
   Small gradient-tinted pill so scores are scannable at a
   glance without adding a second visual system to the table.
   ============================================================ */

function ScoreBadge({ value, decimals, format, inverse = false }) {
  if (value == null) {
    return <span className="text-ink-3">—</span>;
  }

  const numeric = format === "pct" ? value : value;
  const display =
    format === "pct" ? pct(value) : value.toFixed(decimals ?? 2);

  // 0 = weak/red, 1 = strong/green, inverse flips the meaning
  // (used for appeal risk, where low is good)
  const score = inverse ? 1 - numeric : numeric;

  const tone =
    score >= 0.7
      ? "bg-approve-soft text-approve ring-approve-line"
      : score >= 0.4
        ? "bg-review-soft text-review ring-review-line"
        : "bg-deny-soft text-deny ring-deny-line";

  return (
    <span
      className={`inline-flex min-w-[3.25rem] justify-center rounded-md px-2 py-0.5 text-2xs font-medium ring-1 ring-inset ${tone}`}
    >
      {display}
    </span>
  );
}


/* ============================================================
   REQUEST DETAIL
   ============================================================ */

export function RequestDetail() {

  const { id } = useParams();

  const navigate = useNavigate();


  const [data, setData] =
    useState(null);


  const [audit, setAudit] =
    useState([]);


  const [error, setError] =
    useState(null);


  const [appeal, setAppeal] =
    useState({
      rationale: "",
      new_documentation: false,
    });


  const [appealing, setAppealing] =
    useState(false);


  /* ==========================================================
     LOAD REQUEST
     ========================================================== */

  const load = () => {

    setError(null);


    api
      .get(`/api/requests/${id}`)
      .then((response) => {

        console.log(
          "REQUEST DETAILS:",
          response
        );

        console.log(
          "AI VALIDATION:",
          response?.agent_validation
        );

        setData(response);

      })
      .catch((e) => {

        setError(e.message);

      });


    api
      .get(`/api/requests/${id}/audit`)
      .then(setAudit)
      .catch(() => setAudit([]));

  };


  useEffect(() => {
    load();
  }, [id]);


  /* ==========================================================
     LOADING / ERROR
     ========================================================== */

  if (error && !data) {
    return <Alert>{error}</Alert>;
  }


  if (!data) {
    return (
      <Spinner
        label="Loading authorization case"
      />
    );
  }


  /* ==========================================================
     APPEAL
     ========================================================== */

  const canAppeal =
    data.decision === "DENIED" &&
    !(data.appeals || []).some(
      (a) => a.status === "OPEN"
    );


  const submitAppeal = async () => {

    setAppealing(true);

    setError(null);


    try {

      await api.post(
        `/api/requests/${id}/appeal`,
        appeal
      );


      setAppeal({
        rationale: "",
        new_documentation: false,
      });


      load();

    } catch (e) {

      setError(e.message);

    } finally {

      setAppealing(false);

    }

  };


  /* ==========================================================
     BANNER
     ========================================================== */

  const bannerText =
    data.decision_source === "REVIEWER" &&
    data.reviewer_notes
      ? data.reviewer_notes
      : data.criteria?.rationale;


  /* ==========================================================
     AI VALIDATION
     
     Supports multiple possible backend property names.
     Preferred property:
     
       data.agent_validation
     
     ========================================================== */

  const agentValidation =
    data.agent_validation ||
    data.validation_agent ||
    data.ai_validation ||
    data.agent_result ||
    null;


  return (

    <div>
      <div className="space-y-5">

      {/* ======================================================
         BACK
         ====================================================== */}

      <Link
        to="/hospital/requests"
        className="group inline-flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-provider-deep"
      >
        <ArrowLeft
          size={13}
          className="transition-transform group-hover:-translate-x-0.5"
        />
        Requests
      </Link>


      {/* ======================================================
         HEADER
         ====================================================== */}

      <div className="relative overflow-hidden rounded-2xl border border-rule bg-gradient-to-br from-surface via-surface to-provider/15 px-6 py-6 shadow-card">

        <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-gradient-to-br from-provider/40 via-provider/10 to-transparent blur-2xl" />

        <div className="relative flex flex-wrap items-start justify-between gap-5">

          <div>

            <div className="flex items-center gap-2">

              <span className="num text-2xs text-ink-3">
                {data.case_number}
              </span>

              <Status value={data.status} />

            </div>


            <h1 className="mt-2 text-2xl font-semibold text-ink">
              {data.diagnosis}
            </h1>


            <p className="mt-1 text-[13px] text-ink-2">

              {data.requested_treatment}

              {" · "}

              {data.features?.dose_category}

              {" · "}

              {data.features?.frequency}

              {" · "}

              {data.features?.route}

            </p>

          </div>


          {/* METRICS */}

          <div className="grid grid-cols-3 gap-5">

            <Metric
              label="Policy fit"
              value={
                data.policy_fit_score?.toFixed(3) ??
                "—"
              }
            />


            <Metric
              label="Necessity"
              value={
                pct(data.necessity_score)
              }
            />


            <Metric
              label="Processing"
              value={
                data.processing_ms != null
                  ? `${data.processing_ms} ms`
                  : "—"
              }
              muted
            />

          </div>

        </div>

      </div>


      {/* ======================================================
         DECISION BANNER
         ====================================================== */}

      {bannerText && (

        <Alert
          tone={
            data.decision === "APPROVED"
              ? "approve"
              : data.decision === "DENIED"
                ? "deny"
                : "review"
          }
        >
          {bannerText}
        </Alert>

      )}


      {/* ======================================================
         AI VALIDATION AGENT
         
         THIS IS THE NEW SECTION.
         
         It appears BEFORE Decision Ledger.
         ====================================================== */}

      {agentValidation && (

        <AIValidationCard
          validation={agentValidation}
          requestId={id}
        />

      )}


      {/* ======================================================
         AGENT NOT CONNECTED MESSAGE
         
         This helps you immediately see if FastAPI isn't
         returning the agent result.
         ====================================================== */}

      {!agentValidation && (

        <div
          className="relative overflow-hidden rounded-xl border border-dashed border-ruleStrong bg-surface px-5 py-4"
        >

          <div className="flex items-center gap-3">

            <div className="ai-icon bg-provider-soft text-provider-deep">
              <Brain size={20} />
            </div>

            <div>

              <div className="text-[13px] font-semibold text-ink">
                AI Validation Agent
              </div>

              <div className="text-2xs text-ink-3">
                Agent result is not included in the
                request API response yet.
              </div>

            </div>

          </div>

        </div>

      )}


      {/* ======================================================
         DECISION LEDGER + EXPLAINABLE AI
         ====================================================== */}

      <div className="grid gap-5 lg:grid-cols-2">

        <DecisionLedger
          criteria={data.criteria?.criteria}
          rationale={data.criteria?.rationale}
          necessityScore={
            data.necessity_score
          }
        />


        <AttributionRail
          explanation={data.explanation}
        />

      </div>


      {/* ======================================================
         APPEAL FORECAST
         ====================================================== */}

      <AppealForecast
        prediction={
          data.appeal_prediction
        }
      />


      {/* ======================================================
         DOCUMENTS
         ====================================================== */}

      <Card
        eyebrow="Clinical documentation"
        title="Uploaded documents"
      >

        {data.documents?.length ? (

          <div className="space-y-2">

            {data.documents.map((doc) => (

              <div
                key={doc.id}
                className="group flex items-center gap-3 rounded-md border border-rule bg-canvas px-3 py-3 transition-all hover:border-provider-line hover:bg-provider-soft"
              >

                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-provider-soft">
                  <FileText
                    size={16}
                    className="text-provider-deep"
                  />
                </div>


                <div className="min-w-0 flex-1">

                  <div className="truncate text-[13px] font-medium text-ink">
                    {doc.filename}
                  </div>


                  <div className="mt-1 flex items-center gap-2">

                    <span className="text-2xs text-ink-3">
                      {doc.page_count} pages
                    </span>

                    {doc.extraction_confidence != null && (
                      <ConfidenceBar
                        value={doc.extraction_confidence}
                      />
                    )}

                  </div>

                </div>

              </div>

            ))}

          </div>

        ) : (

          <p className="text-[13px] text-ink-3">
            No document attached.
          </p>

        )}

      </Card>


      {/* ======================================================
         APPEAL FORM
         ====================================================== */}

      {canAppeal && (

        <Card
          eyebrow="Appeal"
          title="Challenge this denial"
        >

          {error && (

            <div className="mb-4">
              <Alert>{error}</Alert>
            </div>

          )}


          <Field
            label="Appeal rationale"
            hint="Provide at least 10 characters. This becomes part of the audit record."
          >

            <textarea
              className="input transition-shadow focus:shadow-sm focus:shadow-provider/10"
              rows={5}
              value={appeal.rationale}
              placeholder="Explain why the denial should be reconsidered."
              onChange={(e) =>
                setAppeal({
                  ...appeal,
                  rationale:
                    e.target.value,
                })
              }
            />

          </Field>


          <label className="mt-4 flex cursor-pointer items-center gap-3">

            <input
              type="checkbox"
              checked={
                appeal.new_documentation
              }
              onChange={(e) =>
                setAppeal({
                  ...appeal,
                  new_documentation:
                    e.target.checked,
                })
              }
            />

            <span className="text-[13px]">
              New clinical documentation is being
              submitted
            </span>

          </label>


          <button
            onClick={submitAppeal}
            disabled={
              appealing ||
              appeal.rationale.trim()
                .length < 10
            }
            className="btn mt-4 border-transparent bg-gradient-to-r from-provider to-provider-deep text-white shadow-sm shadow-provider/20 transition-all hover:shadow-md hover:shadow-provider/30 hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none disabled:hover:brightness-100"
          >

            <Send size={14} />

            {appealing
              ? "Filing appeal…"
              : "File appeal"}

          </button>

        </Card>

      )}


      {/* ======================================================
         APPEAL HISTORY
         ====================================================== */}

      {data.appeals?.length > 0 && (

        <Card
          eyebrow="Appeal history"
          title="Appeals on this case"
        >

          <div className="space-y-3">

            {data.appeals.map((a) => (

              <div
                key={a.id}
                className="rounded-md border border-rule bg-canvas p-3"
              >

                <div className="flex items-center justify-between gap-3">

                  <Status value={a.status} />

                  <span className="text-2xs text-ink-3">
                    {fmtDate(a.created_at)}
                  </span>

                </div>


                <p className="mt-2 text-[13px] text-ink-2">
                  {a.rationale}
                </p>


                {a.outcome_notes && (

                  <p className="mt-2 text-[13px]">

                    <strong>
                      Outcome:
                    </strong>

                    {" "}

                    {a.outcome_notes}

                  </p>

                )}

              </div>

            ))}

          </div>

        </Card>

      )}


      {/* ======================================================
         AUDIT TRAIL
         ====================================================== */}

      <Card
        eyebrow="Audit trail"
        title="Case history"
      >

        {audit.length === 0 ? (

          <p className="text-[13px] text-ink-3">
            No audit events recorded.
          </p>

        ) : (

          <div className="relative divide-y divide-rule">

            {audit.map((event) => (

              <div
                key={event.id}
                className="relative py-3 pl-4 first:pt-0 last:pb-0"
              >

                <span className="absolute left-0 top-4 h-1.5 w-1.5 -translate-x-[3px] rounded-full bg-gradient-to-br from-provider to-provider-deep" />

                <div className="flex items-center justify-between gap-3">

                  <span className="font-mono text-2xs font-medium">
                    {event.action}
                  </span>


                  <span className="text-2xs text-ink-3">
                    {fmtDate(
                      event.created_at
                    )}
                  </span>

                </div>


                <div className="mt-1 text-2xs text-ink-3">
                  {event.actor_email ||
                    "System"}
                </div>

              </div>

            ))}

          </div>

        )}

      </Card>

      </div>
    </div>
  );
}


/* ============================================================
   CONFIDENCE BAR
   Compact inline gradient meter for extraction confidence,
   replaces the plain "xx.x% extraction confidence" text.
   ============================================================ */

function ConfidenceBar({ value }) {
  const percent = Math.round((value ?? 0) * 100);

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-1.5 w-14 overflow-hidden rounded-full bg-rule">
        <span
          className="block h-full rounded-full bg-gradient-to-r from-provider to-provider-deep"
          style={{ width: `${percent}%` }}
        />
      </span>
      <span className="text-2xs text-ink-3">
        {percent}% extracted
      </span>
    </span>
  );
}


/* ============================================================
   METRIC
   ============================================================ */

function Metric({ label, value, muted = false }) {

  return (

    <div className="text-right">

      <div className="eyebrow mb-1">
        {label}
      </div>

      <div
        className={
          muted
            ? "num text-lg font-semibold text-ink-2"
            : "num text-lg font-semibold text-ink"
        }
      >
        {value}
      </div>

    </div>

  );
}


/* ============================================================
   SUBMITTED VALUES
   ============================================================ */

const HIDE = /^_/;

export function SubmittedValues({
  features,
  documents,
}) {

  const entries =
    Object.entries(features || {})
      .filter(
        ([key]) => !HIDE.test(key)
      );


  return (

    <Card
      eyebrow="Case record"
      title="Submitted values"
      bodyClass="p-0"
    >

      {documents?.length > 0 && (

        <div className="border-b border-rule bg-canvas/60 px-4 py-3">

          {documents.map((d) => (

            <div
              key={d.id}
              className="flex items-center justify-between gap-3 text-[13px]"
            >

              <span className="truncate font-medium">
                {d.filename}
              </span>


              <span className="num shrink-0 text-2xs text-ink-3">

                {d.page_count}p

                {" · "}

                {pct(
                  d.extraction_confidence
                )}

                {" extracted"}

              </span>

            </div>

          ))}

        </div>

      )}


      <dl className="grid grid-cols-2 gap-x-6 gap-y-0 px-4 py-2">

        {entries.map(([key, value]) => (

          <div
            key={key}
            className="flex justify-between gap-3 border-b border-rule/60 py-1.5"
          >

            <dt className="truncate text-2xs text-ink-3">
              {key.replace(/_/g, " ")}
            </dt>


            <dd className="num shrink-0 text-2xs">
              {String(value)}
            </dd>

          </div>

        ))}

      </dl>

    </Card>

  );
}


/* ============================================================
   AUDIT TRAIL COMPONENT
   ============================================================ */

function AuditEntry({ e }) {
  const [showRaw, setShowRaw] = useState(false)
  const summary = humanizeAuditEvent(e)

  return (
    <li className="relative px-4 py-3 pl-6">
      <span className="absolute left-4 top-[1.4rem] h-1.5 w-1.5 rounded-full bg-gradient-to-br from-provider to-provider-deep" />
      <div className="flex items-baseline justify-between gap-3">
        <span className="num text-2xs font-medium uppercase tracking-wider">
          {e.action.replace(/_/g, ' ')}
        </span>
        <span className="shrink-0 text-2xs text-ink-3">{fmtDate(e.created_at)}</span>
      </div>
      {e.actor_email && (
        <div className="mt-0.5 text-2xs text-ink-3">{e.actor_email}</div>
      )}
      {summary && (
        <p className="mt-1.5 text-[13px] text-ink-2">{summary}</p>
      )}
      {e.detail && Object.keys(e.detail).length > 0 && (
        <>
          <button
            onClick={() => setShowRaw((s) => !s)}
            className="mt-1.5 text-2xs text-ink-3 underline decoration-dotted transition-colors hover:text-provider-deep"
          >
            {showRaw ? 'Hide raw details' : 'Show raw details'}
          </button>
          {showRaw && (
            <pre className="mt-2 overflow-x-auto rounded border border-rule bg-canvas px-2.5 py-2 text-2xs leading-relaxed text-ink-2">
              {JSON.stringify(e.detail, null, 1)}
            </pre>
          )}
        </>
      )}
    </li>
  )
}

export function AuditTrail({
  events = [],
}) {
  return (
    <Card
      eyebrow="Compliance"
      title="Audit trail"
      bodyClass="p-0"
    >
      {events.length === 0 ? (
        <p className="px-4 py-8 text-center text-[13px] text-ink-3">
          No events recorded.
        </p>
      ) : (
        <ol className="divide-y divide-rule">
          {events.map((e) => (
            <AuditEntry key={e.id} e={e} />
          ))}
        </ol>
      )}
    </Card>
  )
}