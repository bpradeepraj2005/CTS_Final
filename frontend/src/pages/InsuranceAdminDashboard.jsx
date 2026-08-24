import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  Clock3,
  FileText,
  LayoutDashboard,
  LogIn,
  Upload,
  Users,
  XCircle,
} from "lucide-react";

import { api } from "../lib/api";

import {
  Alert,
  Card,
  Empty,
  Spinner,
  Status,
  fmtDate,
} from "../components/ui";


export default function InsuranceAdminDashboard() {

  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [section, setSection] = useState("overview");


  const load = async () => {

    setError(null);

    try {

      const result =
        await api.get("/api/admin/dashboard");

      setData(result);

    } catch (err) {

      setError(
        err.message ||
        "Unable to load insurance administration data."
      );

    }

  };


  useEffect(() => {

    load();

    const timer = setInterval(
      load,
      15000
    );

    return () => clearInterval(timer);

  }, []);


  if (error && !data) {

    return (
      <div className="mx-auto max-w-[1100px] p-6">

        <BackButton onClick={() => navigate(-1)} />

        <div className="mt-4">
          <Alert>
            {error}
          </Alert>
        </div>

      </div>
    );

  }


  if (!data) {

    return (
      <div className="mx-auto max-w-[1100px] p-10">

        <Spinner
          label="Loading insurance administration"
        />

      </div>
    );

  }


  const overview =
    data.overview || {};

  const staff =
    data.staff || [];

  const cases =
    data.cases || [];

  const documents =
    data.documents || [];

  const logins =
    data.login_history || [];

  const completionRate =
    overview.total_cases
      ? Math.round((overview.completed_cases / overview.total_cases) * 100)
      : 0;

  const approvalRate =
    overview.completed_cases
      ? Math.round((overview.approved_cases / overview.completed_cases) * 100)
      : 0;


  const NAV = [
    { key: "overview", label: "Overview", icon: LayoutDashboard },
    { key: "staff", label: "Reviewers", icon: Users, count: staff.length },
    { key: "cases", label: "Cases", icon: FileText, count: cases.length },
    { key: "documents", label: "Documents", icon: Upload, count: documents.length },
    { key: "logins", label: "Login history", icon: LogIn, count: logins.length },
  ];


  return (
    <div className="min-h-screen w-full bg-canvas px-4 py-6 sm:px-6 lg:px-10 xl:px-14">

      {/* BACK */}

      <BackButton onClick={() => navigate(-1)} />

      {/* HEADER */}

      <div className="mt-3 flex flex-wrap items-start justify-between gap-4 fade-in" style={{ animationDelay: "40ms" }}>

        <div className="flex items-center gap-3.5">

          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-payer text-white shadow-[0_10px_24px_-10px_rgba(126,58,242,.55)]">
            <Activity size={22} />
          </div>

          <div>
            <div className="eyebrow">
              Insurance administration
            </div>

            <h1 className="mt-0.5 text-2xl font-bold tracking-tight text-ink">
              Administration dashboard
            </h1>

            <p className="mt-1 text-[13px] text-ink-2">
              {data.organization?.name}
              {" · "}
              Insurance / Payer organization
            </p>
          </div>

        </div>

        <span className="hidden items-center gap-1.5 chip border-approve-line bg-approve-soft text-approve sm:inline-flex">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-approve opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-approve" />
          </span>
          Live · refreshing every 15s
        </span>

      </div>


      {error && (
        <div className="mt-4">
          <Alert>
            {error}
          </Alert>
        </div>
      )}


      {/* SIDEBAR + CONTENT */}

      <div className="mt-6 grid gap-6 md:grid-cols-[220px_1fr] xl:grid-cols-[240px_1fr]">

        {/* SIDEBAR */}

        <div
          className="fade-in flex flex-col gap-4 md:sticky md:top-6 md:h-[calc(100vh-8rem)]"
          style={{ animationDelay: "80ms" }}
        >

          <nav className="flex gap-1.5 overflow-x-auto rounded-2xl border border-rule bg-surface p-2.5 pb-2.5 shadow-card md:flex-col md:overflow-visible">

            {NAV.map((item) => {
              const active = section === item.key;

              return (
                <button
                  key={item.key}
                  onClick={() => setSection(item.key)}
                  className={`group flex shrink-0 items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-[13px] font-medium transition-all duration-150 md:shrink ${
                    active
                      ? "bg-payer text-white shadow-[0_8px_18px_-8px_rgba(126,58,242,.55)]"
                      : "text-ink-2 hover:bg-payer-soft hover:text-ink"
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <item.icon
                      size={15}
                      className={active ? "text-white" : "text-ink-3 group-hover:text-payer-deep"}
                    />
                    {item.label}
                  </span>

                  {typeof item.count === "number" && (
                    <span
                      className={`num rounded-full px-1.5 py-0.5 text-2xs font-semibold ${
                        active
                          ? "bg-white/20 text-white"
                          : "bg-canvas text-ink-3"
                      }`}
                    >
                      {item.count}
                    </span>
                  )}
                </button>
              );
            })}

          </nav>

          {/* SIDEBAR FOOTER — fills remaining vertical space on desktop */}

          <div className="hidden flex-1 flex-col justify-end gap-4 md:flex">

            <div className="rounded-2xl border border-rule bg-surface p-4 shadow-card">
              <div className="eyebrow">Case completion</div>
              <div className="num mt-2 text-2xl font-bold text-ink">{completionRate}%</div>
              <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-rule">
                <div
                  className="h-full rounded-full bg-payer transition-all duration-500"
                  style={{ width: `${completionRate}%` }}
                />
              </div>
              <p className="mt-2 text-2xs text-ink-3">
                {overview.completed_cases ?? 0} of {overview.total_cases ?? 0} cases completed
              </p>
            </div>

            <div className="rounded-2xl border border-rule bg-surface p-4 shadow-card">
              <div className="eyebrow">Approval rate</div>
              <div className="num mt-2 text-2xl font-bold text-approve">{approvalRate}%</div>
              <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-rule">
                <div
                  className="h-full rounded-full bg-approve transition-all duration-500"
                  style={{ width: `${approvalRate}%` }}
                />
              </div>
              <p className="mt-2 text-2xs text-ink-3">
                {overview.approved_cases ?? 0} approved of {overview.completed_cases ?? 0} completed
              </p>
            </div>

            <div className="rounded-2xl border border-rule bg-payer-soft p-4">
              <div className="text-2xs font-semibold uppercase tracking-wide text-payer-deep">
                Organization
              </div>
              <div className="mt-1 text-sm font-medium text-ink">{data.organization?.name}</div>
              {data.organization?.created_at && (
                <p className="mt-1 text-2xs text-ink-3">
                  On platform since {fmtDate(data.organization.created_at)}
                </p>
              )}
            </div>

          </div>

        </div>


        {/* CONTENT */}

        <div key={section} className="fade-in space-y-6" style={{ animationDelay: "40ms" }}>

          {section === "overview" && (
            <OverviewSection
              organization={data.organization}
              overview={overview}
              recentCases={cases.slice(0, 6)}
              recentLogins={logins.slice(0, 6)}
            />
          )}

          {section === "staff" && (
            <Card
              eyebrow="People"
              title="Reviewer activity"
              bodyClass="p-0"
            >
              {staff.length === 0 ? (
                <Empty icon={Users} title="No reviewers found">
                  Insurance reviewer accounts will appear here after registration.
                </Empty>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr>
                        <th className="th">Reviewer</th>
                        <th className="th">Email</th>
                        <th className="th">Specialty</th>
                        <th className="th">Availability</th>
                        <th className="th">Last login</th>
                      </tr>
                    </thead>
                    <tbody>
                      {staff.map((member) => (
                        <tr key={member.id} className="transition-colors hover:bg-payer-soft">
                          <td className="td">
                            <div className="font-medium">{member.full_name}</div>
                            <div className="text-2xs text-ink-3">Insurance reviewer</div>
                          </td>
                          <td className="td">{member.email}</td>
                          <td className="td">{member.specialty || "—"}</td>
                          <td className="td">
                            {member.is_available ? (
                              <span className="chip bg-approve/10 text-approve">Available</span>
                            ) : (
                              <span className="chip bg-deny/10 text-deny">Unavailable</span>
                            )}
                          </td>
                          <td className="td text-2xs text-ink-3">
                            {member.last_login_at ? fmtDate(member.last_login_at) : "Never"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {section === "cases" && (
            <Card
              eyebrow="Cases"
              title="Reviewer case history"
              bodyClass="p-0"
            >
              {cases.length === 0 ? (
                <Empty icon={FileText} title="No cases yet">
                  Cases assigned to insurance reviewers will appear here.
                </Empty>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr>
                        <th className="th">Case</th>
                        <th className="th">Submitted by</th>
                        <th className="th">Diagnosis</th>
                        <th className="th">Treatment</th>
                        <th className="th">Status</th>
                        <th className="th">Submitted</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cases.map((item) => (
                        <tr key={item.id} className="transition-colors hover:bg-payer-soft">
                          <td className="td num text-2xs">{item.case_number}</td>
                          <td className="td">
                            {item.created_by ? (
                              <>
                                <div className="font-medium">{item.created_by.name}</div>
                                <div className="text-2xs text-ink-3">{item.created_by.email}</div>
                              </>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="td">{item.diagnosis || "—"}</td>
                          <td className="td">{item.requested_treatment || "—"}</td>
                          <td className="td">
                            <Status value={item.status || "SUBMITTED"} />
                          </td>
                          <td className="td text-2xs text-ink-3">{fmtDate(item.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {section === "documents" && (
            <Card
              eyebrow="Documents"
              title="Uploaded documentation"
              bodyClass="p-0"
            >
              {documents.length === 0 ? (
                <Empty icon={Upload} title="No documents">
                  Uploaded documentation will appear here.
                </Empty>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr>
                        <th className="th">File</th>
                        <th className="th">Request</th>
                        <th className="th">Pages</th>
                        <th className="th">Extraction</th>
                        <th className="th">Uploaded</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.map((document) => (
                        <tr key={document.id} className="transition-colors hover:bg-payer-soft">
                          <td className="td">
                            <div className="flex items-center gap-2">
                              <FileText size={16} className="text-payer-deep" />
                              <div>
                                <div className="font-medium">{document.filename}</div>
                                {document.uploaded_by_name && (
                                  <div className="text-2xs text-ink-3">
                                    Uploaded by {document.uploaded_by_name}
                                  </div>
                                )}
                              </div>
                            </div>
                          </td>
                          <td className="td num text-2xs">
                            {document.request_id ? document.request_id.slice(0, 8) : "Not linked"}
                          </td>
                          <td className="td">{document.page_count ?? "—"}</td>
                          <td className="td">
                            {document.extraction_confidence != null
                              ? `${Math.round(document.extraction_confidence * 100)}%`
                              : "—"}
                          </td>
                          <td className="td text-2xs text-ink-3">{fmtDate(document.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )}

          {section === "logins" && (
            <Card eyebrow="Security" title="Reviewer login history">
              {logins.length === 0 ? (
                <Empty icon={LogIn} title="No login history" />
              ) : (
                <div className="space-y-2">
                  {logins.slice(0, 20).map((event) => (
                    <div
                      key={event.id}
                      className="flex items-center justify-between rounded-lg border border-rule bg-canvas px-4 py-3 transition-colors hover:border-payer-line hover:bg-payer-soft/40"
                    >
                      <div className="flex items-center gap-3">
                        <div className="rounded-lg bg-payer-soft p-2">
                          <LogIn size={16} className="text-payer-deep" />
                        </div>
                        <div>
                          <div className="text-sm font-medium">{event.actor_email}</div>
                          <div className="text-2xs text-ink-3">Reviewer login</div>
                        </div>
                      </div>
                      <span className="text-2xs text-ink-3">{fmtDate(event.created_at)}</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}

        </div>

      </div>

      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
          animation: fadeInUp .45s cubic-bezier(.2,.7,.2,1) both;
        }
        @media (prefers-reduced-motion: reduce) {
          .fade-in { animation: none; }
        }
      `}</style>

    </div>
  );
}


/* ============================================================
   BACK BUTTON
============================================================ */

function BackButton({ onClick }) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-3 transition-colors hover:text-ink"
    >
      <ArrowLeft size={14} />
      Back
    </button>
  );
}


/* ============================================================
   OVERVIEW SECTION
============================================================ */

function OverviewSection({ organization, overview, recentCases = [], recentLogins = [] }) {
  return (
    <div className="space-y-6">

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="eyebrow">Organization</div>
            <h2 className="mt-1 text-lg font-semibold">{organization?.name}</h2>
            <p className="mt-1 text-[13px] text-ink-2">Insurance / Payer organization</p>
          </div>
          <div className="chip border-payer-line bg-payer-soft text-payer-deep">
            Insurance Admin
          </div>
        </div>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          icon={Users}
          label="Reviewers"
          value={overview.total_staff}
          description="Registered insurance staff"
        />
        <Metric
          icon={FileText}
          label="Cases"
          value={overview.total_cases}
          description="Cases assigned to reviewers"
        />
        <Metric
          icon={CheckCircle2}
          label="Completed"
          value={overview.completed_cases}
          description="Processed cases"
        />
        <Metric
          icon={Activity}
          label="Audit events"
          value={overview.total_audit_events}
          description="Recorded activity"
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SmallMetric icon={Clock3} label="Pending review" value={overview.pending_cases} />
        <SmallMetric icon={CheckCircle2} label="Approved" value={overview.approved_cases} tone="approve" />
        <SmallMetric icon={XCircle} label="Denied" value={overview.denied_cases} tone="deny" />
        <SmallMetric icon={Upload} label="Documents" value={overview.total_documents} />
      </div>

      {/* Fills remaining space below the metric grid */}
      <div className="grid gap-4 lg:grid-cols-2">

        <Card eyebrow="Cases" title="Recent cases" bodyClass="p-0">
          {recentCases.length === 0 ? (
            <Empty icon={FileText} title="No recent cases">
              Newly assigned cases will show up here.
            </Empty>
          ) : (
            <div className="divide-y divide-rule">
              {recentCases.map((item) => (
                <div key={item.id} className="flex items-center justify-between gap-3 px-5 py-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">
                      {item.diagnosis || item.case_number}
                    </div>
                    <div className="truncate text-2xs text-ink-3">
                      {item.created_by?.name || "—"} · {fmtDate(item.created_at)}
                    </div>
                  </div>
                  <Status value={item.status || "SUBMITTED"} />
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card eyebrow="Security" title="Recent logins" bodyClass="p-0">
          {recentLogins.length === 0 ? (
            <Empty icon={LogIn} title="No recent logins" />
          ) : (
            <div className="divide-y divide-rule">
              {recentLogins.map((event) => (
                <div key={event.id} className="flex items-center justify-between gap-3 px-5 py-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{event.actor_email}</div>
                    <div className="text-2xs text-ink-3">Reviewer login</div>
                  </div>
                  <span className="shrink-0 text-2xs text-ink-3">{fmtDate(event.created_at)}</span>
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
   METRIC
============================================================ */

function Metric({ icon: Icon, label, value, description }) {
  return (
    <div className="group rounded-2xl border border-rule bg-surface p-5 shadow-card transition-all duration-200 hover:-translate-y-0.5 hover:shadow-elevated">
      <div className="flex items-start justify-between">
        <div>
          <div className="eyebrow">{label}</div>
          <div className="num mt-2 text-3xl font-bold tracking-tight text-ink">{value ?? 0}</div>
          <p className="mt-1 text-2xs text-ink-3">{description}</p>
        </div>
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-payer-soft transition-transform duration-200 group-hover:scale-105">
          <Icon size={18} className="text-payer-deep" />
        </div>
      </div>
    </div>
  );
}


/* ============================================================
   SMALL METRIC
============================================================ */

function SmallMetric({ icon: Icon, label, value, tone }) {
  const toneClass =
    tone === "approve" ? "text-approve" : tone === "deny" ? "text-deny" : "text-payer-deep";

  return (
    <div className="rounded-xl border border-rule bg-surface px-4 py-3 transition-colors duration-200 hover:border-payer-line">
      <div className="flex items-center gap-3">
        <Icon size={17} className={toneClass} />
        <div>
          <div className="text-2xs text-ink-3">{label}</div>
          <div className="num text-lg font-semibold text-ink">{value ?? 0}</div>
        </div>
      </div>
    </div>
  );
}