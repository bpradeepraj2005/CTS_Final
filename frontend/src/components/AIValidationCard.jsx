import {
  AlertTriangle,
  CheckCircle2,
  FileWarning,
  Brain,
  ShieldAlert,
  ShieldCheck,
  Clock3,
  Siren,
  ArrowUp,
  Minus,
  ArrowDown,
  FilePlus2,
} from "lucide-react";

import { useNavigate } from "react-router-dom";

/*
|--------------------------------------------------------------------------
| Determine Human Review Priority
|--------------------------------------------------------------------------
|
| Priority is determined in this order:
|
| 1. Backend-provided severity
| 2. Backend-provided priority
| 3. Backend-provided severity_score
| 4. Fallback calculation from validation findings
|
| Expected backend values can be:
|
| severity: "critical" | "high" | "medium" | "low"
| priority: "critical" | "high" | "medium" | "low"
| severity_score: 0 - 1
|
|--------------------------------------------------------------------------
*/

function normalizePriority(value) {
  if (!value) return null;

  const normalized = String(value)
    .trim()
    .toLowerCase()
    .replace(/[\s_-]+/g, "");

  if (
    normalized === "critical" ||
    normalized === "urgent" ||
    normalized === "emergency" ||
    normalized === "p1"
  ) {
    return "CRITICAL";
  }

  if (
    normalized === "high" ||
    normalized === "highpriority" ||
    normalized === "p2"
  ) {
    return "HIGH";
  }

  if (
    normalized === "medium" ||
    normalized === "moderate" ||
    normalized === "standard" ||
    normalized === "p3"
  ) {
    return "MEDIUM";
  }

  if (
    normalized === "low" ||
    normalized === "routine" ||
    normalized === "p4"
  ) {
    return "LOW";
  }

  return null;
}


/*
|--------------------------------------------------------------------------
| Fallback severity calculation
|--------------------------------------------------------------------------
|
| This is used when the backend has not yet returned a severity.
|
| IMPORTANT:
| The backend value should eventually be the source of truth.
|
*/

function calculateFallbackPriority(validation) {
  const {
    inconsistencies = [],
    missing_context = [],
    documentation_needed = [],
    reasoning = "",
    human_review_required = false,
  } = validation;

  const text = [
    ...inconsistencies,
    ...missing_context,
    ...documentation_needed,
    reasoning,
  ]
    .join(" ")
    .toLowerCase();

  /*
   * CRITICAL indicators
   */
  const criticalKeywords = [
    "life threatening",
    "life-threatening",
    "emergency",
    "acute deterioration",
    "critical condition",
    "severe bleeding",
    "stroke",
    "sepsis",
    "cardiac arrest",
    "respiratory failure",
    "unstable",
    "immediate intervention",
    "urgent intervention",
    "red flag",
    "focal deficit",
    "neurologic deficit",
  ];

  const hasCriticalFinding = criticalKeywords.some((keyword) =>
    text.includes(keyword)
  );

  if (hasCriticalFinding) {
    return "CRITICAL";
  }


  /*
   * HIGH priority indicators
   */
  const highKeywords = [
    "severe",
    "high risk",
    "high-risk",
    "urgent",
    "significant inconsistency",
    "major inconsistency",
    "contraindication",
    "adverse event",
    "rapid progression",
    "persistent",
    "worsening",
    "missing clinical justification",
    "medical necessity cannot be assessed",
  ];

  const hasHighFinding = highKeywords.some((keyword) =>
    text.includes(keyword)
  );

  if (
    hasHighFinding ||
    inconsistencies.length >= 2 ||
    (human_review_required && missing_context.length >= 5)
  ) {
    return "HIGH";
  }


  /*
   * MEDIUM priority
   */
  if (
    human_review_required ||
    missing_context.length >= 2 ||
    documentation_needed.length >= 1 ||
    inconsistencies.length >= 1
  ) {
    return "MEDIUM";
  }


  /*
   * LOW priority
   */
  return "LOW";
}


/*
|--------------------------------------------------------------------------
| Priority configuration
|--------------------------------------------------------------------------
*/

const PRIORITY_CONFIG = {
  CRITICAL: {
    label: "CRITICAL",
    description: "Immediate human review required",
    className: "priority-critical",
    icon: Siren,
  },

  HIGH: {
    label: "HIGH",
    description: "Priority human review required",
    className: "priority-high",
    icon: ArrowUp,
  },

  MEDIUM: {
    label: "MEDIUM",
    description: "Standard human review required",
    className: "priority-medium",
    icon: Minus,
  },

  LOW: {
    label: "LOW",
    description: "Low-priority review",
    className: "priority-low",
    icon: ArrowDown,
  },
};


/*
|--------------------------------------------------------------------------
| Main Component
|--------------------------------------------------------------------------
*/
export default function AIValidationCard({
  validation,
  requestId,
}) {
  const navigate = useNavigate();

  if (!validation) {
    return null;
  }
  if (!validation) {
    return null;
  }


  /*
   * Extract validation result
   */
const {
  contextually_complete,
  consistency_check,
  missing_context = [],
  inconsistencies = [],
  documentation_needed = [],
  reasoning,
  human_review_required,
  confidence = 0,
  severity,
  priority,
  severity_score,

  // Critic Agent result
  critic_agent,
} = validation;


  /*
   * Normalize arrays so UI does not crash if backend
   * returns null.
   */
  const safeMissingContext = Array.isArray(missing_context)
    ? missing_context
    : [];

  const safeInconsistencies = Array.isArray(inconsistencies)
    ? inconsistencies
    : [];

  const safeDocumentation = Array.isArray(documentation_needed)
    ? documentation_needed
    : [];


  /*
   * Confidence
   */
  const confidencePercent = Math.round(
    Math.max(0, Math.min(1, Number(confidence) || 0)) * 100
  );


  /*
   * Consistency
   */
  const consistencyPassed =
    String(consistency_check || "").toUpperCase() === "PASS";


  /*
   * Determine priority
   *
   * Backend priority is preferred.
   */
  const backendPriority =
    normalizePriority(priority) ||
    normalizePriority(severity);


  /*
   * If backend does not provide priority,
   * calculate a fallback.
   */
  const reviewPriority =
    backendPriority ||
    calculateFallbackPriority(validation);


  /*
   * Severity score
   *
   * If backend sends severity_score, use it.
   * Otherwise derive an approximate score from priority.
   */
  let severityScore = Number(severity_score);

  if (Number.isNaN(severityScore)) {
    const fallbackScores = {
      CRITICAL: 0.95,
      HIGH: 0.80,
      MEDIUM: 0.60,
      LOW: 0.30,
    };

    severityScore = fallbackScores[reviewPriority] || 0;
  }

  severityScore = Math.max(
    0,
    Math.min(1, severityScore)
  );

  const severityPercent = Math.round(
    severityScore * 100
  );


  /*
   * Priority UI configuration
   */
  const priorityConfig =
    PRIORITY_CONFIG[reviewPriority] ||
    PRIORITY_CONFIG.MEDIUM;

  const PriorityIcon = priorityConfig.icon;


  /*
   * Human review message
   */
  const reviewMessage =
    reviewPriority === "CRITICAL"
      ? "Immediate human intervention is recommended because the request contains potentially critical clinical concerns."
      : reviewPriority === "HIGH"
      ? "This request should be reviewed before lower-priority cases because the validation findings may affect medical-necessity assessment."
      : reviewPriority === "MEDIUM"
      ? "This request requires human verification before a final authorization decision."
      : "Human review is recommended, but this request can be handled after higher-priority cases.";


  return (
    <div className="ai-validation-card">

      {/* ============================================================
          HEADER
      ============================================================ */}

      <div className="ai-validation-header">

        <div className="ai-validation-title">

          <div className="ai-icon">
            <Brain size={22} />
          </div>

          <div>
            <h2>AI Validation Agent</h2>

            <p>
              Contextual completeness, consistency & severity analysis
            </p>
          </div>

        </div>


        {/* Validation status */}

        {human_review_required ? (
          <div className="review-badge">
            <AlertTriangle size={16} />
            Human Review Required
          </div>
        ) : (
          <div className="complete-badge">
            <CheckCircle2 size={16} />
            Validation Passed
          </div>
        )}

      </div>
{/* ============================================================
    CRITIC AGENT
============================================================ */}

{critic_agent && (
  <div
    className={`mb-5 rounded-xl border px-4 py-3 ${
      critic_agent.status === "PASS"
        ? "border-approve-line bg-approve-soft"
        : critic_agent.status === "WARNING"
        ? "border-review-line bg-review-soft"
        : "border-deny-line bg-deny-soft"
    }`}
  >
    <div className="flex items-start justify-between gap-4">

      <div className="flex items-start gap-3">

        <div
          className={`mt-0.5 ${
            critic_agent.status === "PASS"
              ? "text-approve"
              : critic_agent.status === "WARNING"
              ? "text-review"
              : "text-deny"
          }`}
        >
          {critic_agent.status === "PASS" ? (
            <ShieldCheck size={20} />
          ) : (
            <ShieldAlert size={20} />
          )}
        </div>

        <div>
          <div className="text-[13px] font-semibold text-ink">
            Critic Agent
          </div>

          <div className="mt-1 text-2xs text-ink-2">
            {critic_agent.agent_working
              ? "Primary validation agent output verified."
              : "Primary validation agent output could not be verified."}
          </div>

          {critic_agent.reasoning && (
            <div className="mt-1 text-2xs text-ink-3">
              {critic_agent.reasoning}
            </div>
          )}
        </div>

      </div>

      <div
        className={`rounded-full px-2.5 py-1 text-2xs font-semibold ${
          critic_agent.status === "PASS"
            ? "bg-approve/15 text-approve"
            : critic_agent.status === "WARNING"
            ? "bg-review/15 text-review"
            : "bg-deny/15 text-deny"
        }`}
      >
        {critic_agent.status}
      </div>

    </div>

    {Array.isArray(critic_agent.issues) &&
      critic_agent.issues.length > 0 && (
        <ul className="mt-3 ml-7 list-disc text-2xs text-ink-2">
          {critic_agent.issues.map((issue, index) => (
            <li key={index}>{issue}</li>
          ))}
        </ul>
      )}

  </div>
)}


      {/* ============================================================
          PRIORITY SECTION
      ============================================================ */}

      {human_review_required && (

        <div
          className={`human-review-priority ${priorityConfig.className}`}
        >

          <div className="priority-icon">
            <PriorityIcon size={24} />
          </div>


          <div className="priority-content">

            <div className="priority-top">

              <div>
                <span className="priority-label">
                  HUMAN REVIEW PRIORITY
                </span>

                <strong>
                  {priorityConfig.label}
                </strong>
              </div>


              <div className="severity-score">

                <span>
                  Severity
                </span>

                <strong>
                  {severityPercent}%
                </strong>

              </div>

            </div>


            <p>
              {priorityConfig.description}
            </p>

            <div className="priority-message">
              <Clock3 size={15} />
              {reviewMessage}
            </div>

          </div>

        </div>

      )}


      {/* ============================================================
          SUMMARY STATS
      ============================================================ */}

      <div className="ai-validation-stats">

        {/* Contextual completeness */}

        <div className="ai-stat">

          <span>
            Contextual Completeness
          </span>

          <strong
            className={
              contextually_complete
                ? "status-success"
                : "status-warning"
            }
          >
            {contextually_complete
              ? "Complete"
              : "Incomplete"}
          </strong>

        </div>


        {/* Consistency */}

        <div className="ai-stat">

          <span>
            Consistency Check
          </span>

          <strong
            className={
              consistencyPassed
                ? "status-success"
                : "status-warning"
            }
          >
            {consistency_check || "UNKNOWN"}
          </strong>

        </div>


        {/* Confidence */}

        <div className="ai-stat">

          <span>
            AI Confidence
          </span>

          <strong>
            {confidencePercent}%
          </strong>

        </div>


        {/* Severity */}

        <div className="ai-stat">

          <span>
            Clinical Severity
          </span>

          <strong
            className={`severity-text ${priorityConfig.className}`}
          >
            {reviewPriority}
          </strong>

        </div>


        {/* Review priority */}

        <div className="ai-stat">

          <span>
            Review Priority
          </span>

          <strong
            className={`priority-text ${priorityConfig.className}`}
          >
            {reviewPriority}
          </strong>

        </div>


        {/* Severity score */}

        <div className="ai-stat">

          <span>
            Severity Score
          </span>

          <strong>
            {severityPercent}%
          </strong>

        </div>

      </div>


      {/* ============================================================
          MISSING CONTEXT
      ============================================================ */}

      {safeMissingContext.length > 0 && (

  <div className="ai-section">

    <div className="flex items-start justify-between gap-4">

      <div className="section-heading">

        <AlertTriangle size={18} />

        <h3>
          Missing Context
        </h3>

      </div>

      <button
        type="button"
        onClick={() =>
          navigate(
            `/hospital/new?from=${encodeURIComponent(
              requestId || ""
            )}`
          )
        }
        className="btn btn-ghost shrink-0"
      >
        <FilePlus2 size={14} />
        Add Missing Context
      </button>

    </div>

          <ul>
            {safeMissingContext.map(
              (item, index) => (
                <li key={index}>
                  {item}
                </li>
              )
            )}
          </ul>

        </div>

      )}


      {/* ============================================================
          INCONSISTENCIES
      ============================================================ */}

      {safeInconsistencies.length > 0 && (

        <div className="ai-section">

          <div className="section-heading">

            <ShieldAlert size={18} />

            <h3>
              Inconsistencies Detected
            </h3>

          </div>

          <ul>
            {safeInconsistencies.map(
              (item, index) => (
                <li key={index}>
                  {item}
                </li>
              )
            )}
          </ul>

        </div>

      )}


      {/* ============================================================
          DOCUMENTATION
      ============================================================ */}

      {safeDocumentation.length > 0 && (

        <div className="ai-section">

          <div className="section-heading">

            <FileWarning size={18} />

            <h3>
              Documentation Needed
            </h3>

          </div>

          <ul>
            {safeDocumentation.map(
              (item, index) => (
                <li key={index}>
                  {item}
                </li>
              )
            )}
          </ul>

        </div>

      )}


      {/* ============================================================
          AI REASONING
      ============================================================ */}

      {reasoning && (

        <div className="ai-reasoning">

          <strong>
            AI Reasoning
          </strong>

          <p>
            {reasoning}
          </p>

        </div>

      )}


      {/* ============================================================
          HUMAN REVIEW ALERT
      ============================================================ */}

      {human_review_required && (

        <div
          className={`human-review-alert ${priorityConfig.className}`}
        >

          <PriorityIcon size={20} />

          <div>

            <strong>
              {reviewPriority} Priority Human Review
            </strong>

            <p>
              {reviewMessage}
            </p>

          </div>

        </div>

      )}

    </div>
  );
}