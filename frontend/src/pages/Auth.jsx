import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Loader2,
  ShieldCheck,
  Building2,
  Lock,
  Mail,
  User,
  CheckCircle2,
} from "lucide-react";

import { useAuth } from "../lib/auth";
import { Alert, Field } from "../components/ui";

/* =========================================================
   DESIGN TOKENS — DARK THEME
   (neutral tokens shared by both portals — text, surfaces,
   inputs. Portal-tinted backgrounds live in PAGE_TOKENS below.)
========================================================= */

const C = {
  surface: "#141B2A",
  surfaceAlt: "#101725",
  inputBg: "#0C111C",
  inputFocusBg: "#111827",
  panel: "rgba(255,255,255,0.05)",

  text: "#E8EDF7",
  text2: "#A7B4CC",
  text3: "#7A87A1",
};

/* =========================================================
   PORTAL THEMES

   hospital / provider -> violet
   insurance / payer   -> blue

   `main` is the filled-button tone (white text sits on it);
   `bright` is the accent used for text and icons, which needs
   the light end of the ramp to carry on a dark panel.
========================================================= */

const THEME = {
  provider: {
    name: "Hospital management",
    home: "/hospital",

    main: "#8B5CF6",
    deep: "#6D28D9",
    bright: "#A78BFA",

    soft: "rgba(139,92,246,0.16)",
    line: "rgba(167,139,250,0.32)",

    Icon: Building2,

    tagline:
      "Submit and track prior authorizations without the phone tag.",

    points: [
      "Real-time case status",
      "Direct messaging with payers",
      "Full audit trail on every request",
    ],
  },

  payer: {
    name: "Insurance organization",
    home: "/payer",

    main: "#3B82F6",
    deep: "#1D4ED8",
    bright: "#60A5FA",

    soft: "rgba(59,130,246,0.16)",
    line: "rgba(96,165,250,0.32)",

    Icon: ShieldCheck,

    tagline:
      "Review, decide, and manage appeals from a single queue.",

    points: [
      "Configurable review workflows",
      "SLA tracking on every case",
      "Exportable decision history",
    ],
  },
};



/* =========================================================
   PAGE-LEVEL BACKGROUND TOKENS — PORTAL AWARE

   These drive the tinted page background, dot texture, ambient
   gradient glow, aside panel gradient, and card shadow.
   `provider` runs violet, `payer` runs blue.
========================================================= */

const PAGE_TOKENS = {
  provider: {
    bg: "#0B0714",
    border: "#2A2144",
    borderStrong: "#3D3161",
    dot: "rgba(196,181,253,.09)",
    glow: "rgba(139,92,246,0.20)",
    asideGradient:
      "linear-gradient(145deg, #16102A 0%, #100A1E 45%, #0A0714 100%)",
    cardShadow: "0 28px 64px -34px rgba(0,0,0,0.9)",
  },

  payer: {
    bg: "#060B18",
    border: "#1D2A44",
    borderStrong: "#2C3E60",
    dot: "rgba(147,197,253,.09)",
    glow: "rgba(59,130,246,0.20)",
    asideGradient:
      "linear-gradient(145deg, #0D1A32 0%, #091124 45%, #060B18 100%)",
    cardShadow: "0 28px 64px -34px rgba(0,0,0,0.9)",
  },
};

/* =========================================================
   SMALL SHARED PIECES
========================================================= */

function AuthMotionStyles() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&display=swap');

      .font-display {
        font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
      }

      .font-body {
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
      }

      @keyframes authFadeUp {
        from {
          opacity: 0;
          transform: translateY(14px);
        }

        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes authFloatA {
        0%, 100% {
          transform: translate(0, 0) scale(1);
        }

        50% {
          transform: translate(20px, -24px) scale(1.06);
        }
      }

      @keyframes authFloatB {
        0%, 100% {
          transform: translate(0, 0) scale(1);
        }

        50% {
          transform: translate(-18px, 18px) scale(1.08);
        }
      }

      @keyframes authShake {
        10%, 90% {
          transform: translateX(-1px);
        }

        20%, 80% {
          transform: translateX(2px);
        }

        30%, 50%, 70% {
          transform: translateX(-4px);
        }

        40%, 60% {
          transform: translateX(4px);
        }
      }

      .auth-panel-enter {
        animation: authFadeUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
      }

      .auth-field-enter {
        opacity: 0;
        animation: authFadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
      }

      .auth-alert-enter {
        animation: authShake 0.45s ease-in-out both;
      }

      .auth-blob {
        animation: authFloatA 11s ease-in-out infinite;
      }

      .auth-blob-b {
        animation: authFloatB 13s ease-in-out infinite;
      }

      .auth-input {
        width: 100%;
        border-radius: 0.6rem;
        border: 1px solid var(--field-border, #2A2144);
        background: ${C.inputBg};
        color: ${C.text};
        padding: 0.65rem 0.85rem;
        font-size: 0.875rem;
        outline: none;
        transition:
          border-color .15s ease,
          box-shadow .15s ease,
          background .15s ease;
      }

      .auth-input::placeholder {
        color: ${C.text3};
      }

      .auth-input:focus {
        border-color: var(--accent, #8B5CF6);
        background: ${C.inputFocusBg};
        box-shadow:
          0 0 0 3px
          var(--accent-soft, rgba(139,92,246,.22));
      }

      @media (prefers-reduced-motion: reduce) {
        .auth-panel-enter,
        .auth-field-enter,
        .auth-alert-enter,
        .auth-blob,
        .auth-blob-b {
          animation: none !important;
        }
      }
    `}</style>
  );
}

/* =========================================================
   FIELD ICON
========================================================= */

function FieldIcon({ icon: Icon }) {
  return (
    <Icon
      size={15}
      strokeWidth={2}
      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2"
      style={{ color: C.text3 }}
    />
  );
}

function IconField({ icon, children }) {
  return (
    <div className="relative [&_input]:pl-9">
      <div className="relative">{children}</div>
      <FieldIcon icon={icon} />
    </div>
  );
}

/* =========================================================
   SUBMIT BUTTON
========================================================= */

function SubmitButton({
  busy,
  idleLabel,
  busyLabel,
  main,
  deep,
}) {
  const [hover, setHover] = useState(false);

  return (
    <button
      type="submit"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="group relative w-full overflow-hidden rounded-lg border px-5 py-2.5 text-sm font-semibold transition-all duration-200 ease-out disabled:opacity-70"
      style={{
        borderColor: main,
        background: hover && !busy ? deep : main,
        color: "#FFFFFF",
        transform:
          hover && !busy ? "translateY(-1px)" : "none",
        boxShadow:
          hover && !busy
            ? `0 10px 24px -14px ${main}`
            : "none",
      }}
      disabled={busy}
    >
      <span className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/30 to-transparent transition-transform duration-700 ease-out group-hover:translate-x-full" />

      <span className="relative inline-flex items-center justify-center gap-2">
        {busy && <Loader2 size={15} className="animate-spin" />}

        {busy ? busyLabel : idleLabel}
      </span>
    </button>
  );
}

/* =========================================================
   SHELL
========================================================= */

function Shell({
  portal,
  title,
  subtitle,
  children,
  footer,
}) {
  const t = THEME[portal];
  const P = PAGE_TOKENS[portal];
  const Icon = t.Icon;

  return (
    <div
      className="font-body relative min-h-screen overflow-hidden"
      style={{
        background: P.bg,
        "--accent": t.main,
        "--accent-soft": t.soft,
        "--field-border": P.border,
      }}
    >
      <AuthMotionStyles />

      {/* =================================================
          BACKGROUND DOT TEXTURE
      ================================================= */}

      <div
        className="pointer-events-none absolute inset-0 opacity-[0.45]"
        style={{
          backgroundImage: `radial-gradient(circle, ${P.dot} 1px, transparent 1px)`,
          backgroundSize: "22px 22px",
        }}
      />

      {/* =================================================
          MAIN GRADIENT (portal-tinted)
      ================================================= */}

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: `
            radial-gradient(
              ellipse 900px 500px at 50% -10%,
              ${P.glow},
              transparent 65%
            )
          `,
        }}
      />

      {/* =================================================
          SECONDARY GRADIENT
      ================================================= */}

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: `
            radial-gradient(
              ellipse 650px 450px at 100% 80%,
              ${t.soft},
              transparent 65%
            )
          `,
        }}
      />

      <div
        className="relative grid min-h-screen lg:grid-cols-[minmax(0,1fr)_minmax(0,560px)]"
      >
        {/* =================================================
            LEFT — INFORMATION PANEL
        ================================================= */}

        <aside
          className="relative hidden overflow-hidden lg:flex lg:flex-col lg:p-14 xl:p-16"
          style={{
            background: P.asideGradient,
            borderRight: `1px solid ${P.border}`,
          }}
        >
          {/* Portal-colored animated blob */}
          <div
            className="auth-blob absolute -left-24 -top-24 h-80 w-80 rounded-full blur-3xl"
            style={{
              background: t.main,
              opacity: 0.13,
            }}
          />

          {/* Portal-colored animated blob */}
          <div
            className="auth-blob-b absolute -bottom-32 right-0 h-96 w-96 rounded-full blur-3xl"
            style={{
              background: t.main,
              opacity: 0.10,
            }}
          />

          {/* Extra ambient glow */}
          <div
            className="pointer-events-none absolute left-1/2 top-1/2 h-80 w-80 -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl"
            style={{
              background: t.soft,
            }}
          />

          {/* =================================================
              LEFT HEADER
          ================================================= */}

          <div className="relative flex items-center justify-between">
            <Link
              to="/"
              className="inline-flex items-center gap-1.5 text-[13px] transition-colors hover:opacity-70"
              style={{
                color: C.text2,
              }}
            >
              <ArrowLeft size={13} />
              Choose a different portal
            </Link>

            <div
              className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[12px] font-medium backdrop-blur"
              style={{
                borderColor: t.line,
                background: C.panel,
                color: C.text2,
              }}
            >
              <span className="relative flex h-1.5 w-1.5">
                <span
                  className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
                  style={{
                    background: t.bright,
                  }}
                />

                <span
                  className="relative inline-flex h-1.5 w-1.5 rounded-full"
                  style={{
                    background: t.bright,
                  }}
                />
              </span>

              {t.name} portal
            </div>
          </div>

          {/* =================================================
              LEFT CONTENT
          ================================================= */}

          <div className="relative flex flex-1 flex-col justify-center">
            <div
              className="flex h-14 w-14 items-center justify-center rounded-2xl border backdrop-blur"
              style={{
                borderColor: t.line,
                background: C.panel,
                boxShadow: `0 14px 32px -18px ${t.main}`,
              }}
            >
              <Icon
                size={24}
                strokeWidth={1.75}
                style={{
                  color: t.bright,
                }}
              />
            </div>

            <h2
              className="font-display mt-7 max-w-md text-[32px] font-semibold leading-[1.15] tracking-tight xl:text-[36px]"
              style={{
                color: C.text,
              }}
            >
              {t.tagline}
            </h2>

            <p
              className="mt-4 max-w-sm text-[15px] leading-relaxed"
              style={{
                color: C.text2,
              }}
            >
              Built for teams who move authorizations and claims between
              hospitals and payers every day.
            </p>

            {/* Points */}
            <ul className="mt-10 space-y-4">
              {t.points.map((point) => (
                <li
                  key={point}
                  className="flex items-center gap-3 text-[15px]"
                  style={{
                    color: C.text2,
                  }}
                >
                  <span
                    className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full"
                    style={{
                      background: t.soft,
                      color: t.bright,
                    }}
                  >
                    <CheckCircle2 size={14} />
                  </span>

                  {point}
                </li>
              ))}
            </ul>

            {/* Trust badges */}
            <div
              className="mt-12 flex items-center gap-6 border-t pt-8"
              style={{
                borderColor: P.border,
              }}
            >
              {[
                "99.9% uptime",
                "SOC 2 Type II",
                "HIPAA-ready",
              ].map((tag) => (
                <span
                  key={tag}
                  className="text-[11px] font-medium uppercase tracking-wide"
                  style={{
                    color: C.text3,
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          <p
            className="relative text-[12px]"
            style={{
              color: C.text3,
            }}
          >
            Access is limited to verified staff accounts. All activity is
            logged for audit purposes.
          </p>
        </aside>

        {/* =================================================
            RIGHT — FORM
        ================================================= */}

        <div
          className="relative grid place-items-center px-6 py-12 sm:px-10"
          style={{
            background: P.bg,
          }}
        >
          {/* Mobile Back */}
          <Link
            to="/"
            className="mb-6 inline-flex items-center gap-1.5 text-[13px] lg:hidden"
            style={{
              color: C.text2,
            }}
          >
            <ArrowLeft size={13} />
            Choose a different portal
          </Link>

          <div className="relative w-full max-w-[440px]">
            {/* =================================================
                FORM CARD
            ================================================= */}

            <div
              className="auth-panel-enter relative overflow-hidden rounded-2xl border p-8 sm:p-9"
              style={{
                borderColor: P.border,
                background: C.surface,
                boxShadow: P.cardShadow,
              }}
            >
              {/* Top Gradient */}
              <div
                className="absolute inset-x-0 top-0 h-[3px]"
                style={{
                  background: `
                    linear-gradient(
                      to right,
                      transparent,
                      ${t.bright},
                      transparent
                    )
                  `,
                }}
              />

              {/* Portal Badge */}
              <div
                className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium"
                style={{
                  borderColor: t.line,
                  background: t.soft,
                  color: t.bright,
                }}
              >
                {t.name}
              </div>

              {/* Title */}
              <h1
                className="font-display mt-5 text-[28px] font-semibold tracking-tight"
                style={{
                  color: C.text,
                }}
              >
                {title}
              </h1>

              {/* Subtitle */}
              <p
                className="mt-2 text-[14px] leading-relaxed"
                style={{
                  color: C.text2,
                }}
              >
                {subtitle}
              </p>

              {/* Form */}
              <div className="mt-7 space-y-4">
                {children}
              </div>

              {/* Security Note */}
              <div
                className="mt-7 flex items-center gap-2 border-t pt-5 text-[12px]"
                style={{
                  borderColor: P.border,
                  color: C.text3,
                }}
              >
                <ShieldCheck size={14} className="shrink-0" />

                Your credentials are encrypted in transit and never stored in
                plain text.
              </div>
            </div>

            {/* Footer */}
            <p
              className="mt-5 text-center text-[13px]"
              style={{
                color: C.text2,
              }}
            >
              {footer}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   ANIMATED ERROR ALERT
========================================================= */

function AnimatedAlert({
  children,
  onDismiss,
}) {
  return (
    <div className="auth-alert-enter">
      <Alert onDismiss={onDismiss}>
        {children}
      </Alert>
    </div>
  );
}

/* =========================================================
   NORMAL STAFF SIGN IN
========================================================= */

export function SignIn({ portal }) {
  const t = THEME[portal];

  const { login } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();

    setBusy(true);
    setError(null);

    try {
      await login({
        ...form,
        portal,
      });

      navigate(t.home, {
        replace: true,
      });
    } catch (err) {
      setError(
        err?.message || "Unable to sign in."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell
      portal={portal}
      title="Sign in"
      subtitle="Use your staff account registered to this portal."
      footer={
        <>
          No staff account yet?{" "}
          <Link
            to={
              portal === "provider"
                ? "/hospital/staff/signup"
                : "/payer/staff/signup"
            }
            className="font-medium hover:underline"
            style={{
              color: t.bright,
            }}
          >
            Create one
          </Link>
        </>
      }
    >
      <form
        onSubmit={submit}
        className="space-y-4"
      >
        {error && (
          <AnimatedAlert
            onDismiss={() => setError(null)}
          >
            {error}
          </AnimatedAlert>
        )}

        {/* Email */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "40ms",
          }}
        >
          <Field label="Work email">
            <IconField icon={Mail}>
              <input
                className="auth-input"
                type="email"
                required
                autoFocus
                autoComplete="email"
                value={form.email}
                onChange={(e) =>
                  setForm({
                    ...form,
                    email: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Password */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "100ms",
          }}
        >
          <Field label="Password">
            <IconField icon={Lock}>
              <input
                className="auth-input"
                type="password"
                required
                autoComplete="current-password"
                value={form.password}
                onChange={(e) =>
                  setForm({
                    ...form,
                    password: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Submit */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "160ms",
          }}
        >
          <SubmitButton
            busy={busy}
            idleLabel="Sign in"
            busyLabel="Signing in…"
            main={t.main}
            deep={t.deep}
          />
        </div>
      </form>
    </Shell>
  );
}

/* =========================================================
   ADMIN SIGN IN
========================================================= */

export function AdminSignIn({ portal }) {
  const t = THEME[portal];

  const adminRole =
    portal === "provider"
      ? "PROVIDER_ADMIN"
      : "PAYER_ADMIN";

  const adminHome =
    portal === "provider"
      ? "/hospital/admin"
      : "/payer/admin";

  const signupPath =
    portal === "provider"
      ? "/hospital/admin/signup"
      : "/payer/admin/signup";

  const { login, logout } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();

    setBusy(true);
    setError(null);

    try {
      const user = await login({
        ...form,
        portal,
      });

      if (user?.role !== adminRole) {
        logout();

        const message =
          portal === "provider"
            ? "This account does not have hospital administrator access."
            : "This account does not have insurance administrator access.";

        setError(message);
        return;
      }

      navigate(adminHome, {
        replace: true,
      });
    } catch (err) {
      setError(
        err?.message || "Unable to sign in."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell
      portal={portal}
      title="Admin sign in"
      subtitle={
        portal === "provider"
          ? "Sign in using your hospital administrator account."
          : "Sign in using your insurance administrator account."
      }
      footer={
        <>
          Don't have an admin account?{" "}
          <Link
            to={signupPath}
            className="font-medium hover:underline"
            style={{
              color: t.bright,
            }}
          >
            Create admin account
          </Link>
        </>
      }
    >
      <form
        onSubmit={submit}
        className="space-y-4"
      >
        {error && (
          <AnimatedAlert
            onDismiss={() => setError(null)}
          >
            {error}
          </AnimatedAlert>
        )}

        {/* Email */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "40ms",
          }}
        >
          <Field label="Work email">
            <IconField icon={Mail}>
              <input
                className="auth-input"
                type="email"
                required
                autoFocus
                autoComplete="email"
                value={form.email}
                onChange={(e) =>
                  setForm({
                    ...form,
                    email: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Password */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "100ms",
          }}
        >
          <Field label="Password">
            <IconField icon={Lock}>
              <input
                className="auth-input"
                type="password"
                required
                autoComplete="current-password"
                value={form.password}
                onChange={(e) =>
                  setForm({
                    ...form,
                    password: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Submit */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "160ms",
          }}
        >
          <SubmitButton
            busy={busy}
            idleLabel="Sign in as admin"
            busyLabel="Signing in…"
            main={t.main}
            deep={t.deep}
          />
        </div>
      </form>
    </Shell>
  );
}

/* =========================================================
   HOSPITAL STAFF SIGN UP
========================================================= */

export function SignUpProvider() {
  const t = THEME.provider;

  const { signupProvider } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    full_name: "",
    email: "",
    password: "",
    organization_name: "",
  });

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();

    setBusy(true);
    setError(null);

    try {
      await signupProvider(form);

      navigate("/hospital", {
        replace: true,
      });
    } catch (err) {
      setError(
        err?.message || "Unable to create account."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell
      portal="provider"
      title="Create a hospital staff account"
      subtitle="Create an account to submit authorization requests and manage hospital cases."
      footer={
        <>
          Already registered?{" "}
          <Link
            to="/hospital/staff/signin"
            className="font-medium hover:underline"
            style={{
              color: t.bright,
            }}
          >
            Sign in
          </Link>
        </>
      }
    >
      <form
        onSubmit={submit}
        className="space-y-4"
      >
        {error && (
          <AnimatedAlert
            onDismiss={() => setError(null)}
          >
            {error}
          </AnimatedAlert>
        )}

        {/* Full name */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "40ms",
          }}
        >
          <Field label="Full name">
            <IconField icon={User}>
              <input
                className="auth-input"
                required
                autoFocus
                value={form.full_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    full_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Organization */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "90ms",
          }}
        >
          <Field label="Hospital or clinic">
            <IconField icon={Building2}>
              <input
                className="auth-input"
                required
                value={form.organization_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    organization_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Email */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "140ms",
          }}
        >
          <Field label="Work email">
            <IconField icon={Mail}>
              <input
                className="auth-input"
                type="email"
                required
                autoComplete="email"
                value={form.email}
                onChange={(e) =>
                  setForm({
                    ...form,
                    email: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Password */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "190ms",
          }}
        >
          <Field
            label="Password"
            hint="At least 8 characters."
          >
            <IconField icon={Lock}>
              <input
                className="auth-input"
                type="password"
                required
                minLength={8}
                autoComplete="new-password"
                value={form.password}
                onChange={(e) =>
                  setForm({
                    ...form,
                    password: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Submit */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "240ms",
          }}
        >
          <SubmitButton
            busy={busy}
            idleLabel="Create account"
            busyLabel="Creating account…"
            main={t.main}
            deep={t.deep}
          />
        </div>
      </form>
    </Shell>
  );
}

/* =========================================================
   INSURANCE / PAYER STAFF SIGN UP
========================================================= */

export function SignUpPayer() {
  const t = THEME.payer;

  const { signupPayer } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    full_name: "",
    organization_name: "",
    email: "",
    password: "",
  });

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();

    setBusy(true);
    setError(null);

    try {
      await signupPayer({
        full_name: form.full_name.trim(),
        organization_name: form.organization_name.trim(),
        email: form.email.trim(),
        password: form.password,
      });

      navigate("/payer", {
        replace: true,
      });
    } catch (err) {
      setError(
        err?.message ||
          "Unable to create insurance staff account."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell
      portal="payer"
      title="Create an insurance staff account"
      subtitle="Create an account to review authorization cases and handle appeals."
      footer={
        <>
          Already registered?{" "}
          <Link
            to="/payer/staff/signin"
            className="font-medium hover:underline"
            style={{
              color: t.bright,
            }}
          >
            Sign in
          </Link>
        </>
      }
    >
      <form
        onSubmit={submit}
        className="space-y-4"
      >
        {error && (
          <AnimatedAlert
            onDismiss={() => setError(null)}
          >
            {error}
          </AnimatedAlert>
        )}

        {/* Full name */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "40ms",
          }}
        >
          <Field label="Full name">
            <IconField icon={User}>
              <input
                className="auth-input"
                required
                autoFocus
                value={form.full_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    full_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Organization */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "90ms",
          }}
        >
          <Field label="Insurance organization">
            <IconField icon={Building2}>
              <input
                className="auth-input"
                required
                value={form.organization_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    organization_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Email */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "140ms",
          }}
        >
          <Field label="Work email">
            <IconField icon={Mail}>
              <input
                className="auth-input"
                type="email"
                required
                autoComplete="email"
                value={form.email}
                onChange={(e) =>
                  setForm({
                    ...form,
                    email: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Password */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "190ms",
          }}
        >
          <Field
            label="Password"
            hint="At least 8 characters."
          >
            <IconField icon={Lock}>
              <input
                className="auth-input"
                type="password"
                required
                minLength={8}
                autoComplete="new-password"
                value={form.password}
                onChange={(e) =>
                  setForm({
                    ...form,
                    password: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Submit */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "240ms",
          }}
        >
          <SubmitButton
            busy={busy}
            idleLabel="Create account"
            busyLabel="Creating account…"
            main={t.main}
            deep={t.deep}
          />
        </div>
      </form>
    </Shell>
  );
}

/* =========================================================
   ADMIN ACCOUNT CREATION
========================================================= */

export function SignUpAdmin({ portal }) {
  const t = THEME[portal];

  const { signupAdmin } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    full_name: "",
    email: "",
    password: "",
    organization_name: "",
  });

  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const isHospital = portal === "provider";

  const submit = async (e) => {
    e.preventDefault();

    setBusy(true);
    setError(null);

    try {
      await signupAdmin({
        ...form,
        portal,
      });

      navigate(
        isHospital
          ? "/hospital/admin"
          : "/payer/admin",
        {
          replace: true,
        }
      );
    } catch (err) {
      setError(
        err?.message ||
          "Unable to create admin account."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Shell
      portal={portal}
      title={
        isHospital
          ? "Create hospital admin account"
          : "Create insurance admin account"
      }
      subtitle={
        isHospital
          ? "Create an administrator account to access hospital-wide operations and audit information."
          : "Create an administrator account to access insurance-wide operations and audit information."
      }
      footer={
        <>
          Already have an admin account?{" "}
          <Link
            to={
              isHospital
                ? "/hospital/admin/signin"
                : "/payer/admin/signin"
            }
            className="font-medium hover:underline"
            style={{
              color: t.bright,
            }}
          >
            Sign in
          </Link>
        </>
      }
    >
      <form
        onSubmit={submit}
        className="space-y-4"
      >
        {error && (
          <AnimatedAlert
            onDismiss={() => setError(null)}
          >
            {error}
          </AnimatedAlert>
        )}

        {/* Full name */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "40ms",
          }}
        >
          <Field label="Full name">
            <IconField icon={User}>
              <input
                className="auth-input"
                required
                autoFocus
                value={form.full_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    full_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Organization */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "90ms",
          }}
        >
          <Field
            label={
              isHospital
                ? "Hospital or clinic"
                : "Insurance organization"
            }
          >
            <IconField icon={Building2}>
              <input
                className="auth-input"
                required
                value={form.organization_name}
                onChange={(e) =>
                  setForm({
                    ...form,
                    organization_name: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Email */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "140ms",
          }}
        >
          <Field label="Work email">
            <IconField icon={Mail}>
              <input
                className="auth-input"
                type="email"
                required
                autoComplete="email"
                value={form.email}
                onChange={(e) =>
                  setForm({
                    ...form,
                    email: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Password */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "190ms",
          }}
        >
          <Field
            label="Password"
            hint="At least 8 characters."
          >
            <IconField icon={Lock}>
              <input
                className="auth-input"
                type="password"
                required
                minLength={8}
                autoComplete="new-password"
                value={form.password}
                onChange={(e) =>
                  setForm({
                    ...form,
                    password: e.target.value,
                  })
                }
              />
            </IconField>
          </Field>
        </div>

        {/* Submit */}
        <div
          className="auth-field-enter"
          style={{
            animationDelay: "240ms",
          }}
        >
          <SubmitButton
            busy={busy}
            idleLabel="Create admin account"
            busyLabel="Creating admin account…"
            main={t.main}
            deep={t.deep}
          />
        </div>
      </form>
    </Shell>
  );
}