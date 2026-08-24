import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Building2,
  ShieldCheck,
  UserCog,
  Users,
} from "lucide-react";

import { Link } from "react-router-dom";

/* ============================================================
   DESIGN TOKENS — DARK THEME

   provider (Hospital) = violet, payer (Insurance) = blue.
   `*Deep` is the brighter end of each ramp — on a dark ground
   that is the emphatic direction — and `*Press` the darker step
   kept for filled buttons carrying white text.
============================================================ */

const C = {
  bg: "#0A0F1B",
  surface: "#141B2A",
  surfaceAlt: "#101725",
  border: "#232C40",
  borderStrong: "#33405C",

  text: "#E8EDF7",
  text2: "#A7B4CC",
  text3: "#7A87A1",

  provider: "#8B5CF6",
  providerDeep: "#A78BFA",
  providerPress: "#6D28D9",
  providerSoft: "rgba(139,92,246,0.16)",
  providerLine: "rgba(167,139,250,0.35)",

  payer: "#3B82F6",
  payerDeep: "#60A5FA",
  payerPress: "#1D4ED8",
  payerSoft: "rgba(59,130,246,0.16)",
  payerLine: "rgba(96,165,250,0.35)",
};

const chip =
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium";

function PageStyle() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

      .font-display {
        font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
      }

      .font-body {
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
      }

      .font-tag {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
      }

      @keyframes fadeUp {
        from {
          opacity: 0;
          transform: translateY(14px);
        }

        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .fade-enter {
        animation: fadeUp .6s cubic-bezier(.2,.7,.2,1) both;
      }

      @media (prefers-reduced-motion: reduce) {
        .fade-enter {
          animation: none !important;
        }
      }
    `}</style>
  );
}

export default function PortalSelection({ portal }) {
  const hospital = portal === "hospital";

  const config = hospital
    ? {
        eyebrow: "Hospital portal",
        title: "Choose your access",
        description:
          "Select whether you are accessing the hospital administration portal or the staff authorization workspace.",
        icon: Building2,

        adminTitle: "Admin Portal",
        adminDescription:
          "Monitor staff activity, submitted cases, audit history and hospital-wide authorization operations.",

        staffTitle: "Staff Portal",
        staffDescription:
          "Create authorization requests, upload documents and manage existing hospital cases.",

        adminTo: "/hospital/admin/signin",
        staffTo: "/hospital/staff/signin",

        accent: "provider",
      }
    : {
        eyebrow: "Insurance portal",
        title: "Choose your access",
        description:
          "Select whether you are accessing the insurance administration portal or the reviewer workspace.",
        icon: ShieldCheck,

        adminTitle: "Admin Portal",
        adminDescription:
          "Monitor reviewer activity, case assignments, decisions and the complete insurance audit history.",

        staffTitle: "Staff Portal",
        staffDescription:
          "Review authorization cases, handle appeals and make adjudication decisions.",

        adminTo: "/payer/admin/signin",
        staffTo: "/payer/staff/signin",

        accent: "payer",
      };

  const PortalIcon = config.icon;

  const provider = config.accent === "provider";

  const main = provider ? C.provider : C.payer;
  /* Accent text sits on a dark panel, so it uses the bright end
     of the ramp rather than the mid tone used for fills. */
  const bright = provider ? C.providerDeep : C.payerDeep;
  const soft = provider ? C.providerSoft : C.payerSoft;
  const line = provider ? C.providerLine : C.payerLine;

  return (
    <div
      className="font-body relative min-h-screen overflow-hidden"
      style={{
        background: C.bg,
        color: C.text,
      }}
    >
      <PageStyle />

      {/* =====================================================
          BACKGROUND
      ===================================================== */}

      {/* Faint dot texture */}
      <div
        className="pointer-events-none fixed inset-0 -z-50 opacity-[0.55]"
        style={{
          backgroundImage:
            "radial-gradient(rgba(196,181,253,.08) 1px, transparent 1px)",
          backgroundSize: "28px 28px",
        }}
      />

      {/* Portal accent glow */}
      <div
        className="pointer-events-none fixed inset-0 -z-40"
        style={{
          background: `radial-gradient(
            ellipse 900px 500px at 50% -10%,
            ${soft},
            transparent 60%
          )`,
        }}
      />

      {/* Secondary ambient glow */}
      <div
        className="pointer-events-none fixed inset-0 -z-40"
        style={{
          background:
            "radial-gradient(ellipse 650px 450px at 10% 60%, rgba(139,92,246,0.12), transparent 65%)",
        }}
      />

      {/* =====================================================
          HEADER
      ===================================================== */}

      <header
        className="sticky top-0 z-10 backdrop-blur-xl"
        style={{
          background: "rgba(10,15,27,0.90)",
          borderBottom: `1px solid ${C.border}`,
          boxShadow: "0 10px 28px -18px rgba(0,0,0,.85)",
        }}
      >
        <div className="mx-auto flex h-16 max-w-[1100px] items-center px-6">
          <Link
            to="/"
            className="flex items-center gap-2 text-sm font-medium transition-all duration-200 hover:-translate-x-0.5 hover:opacity-80"
            style={{ color: C.text2 }}
          >
            <ArrowLeft size={16} />
            Back
          </Link>
        </div>
      </header>

      {/* =====================================================
          MAIN
      ===================================================== */}

      <main className="relative mx-auto max-w-[1100px] px-6 py-16">
        {/* =================================================
            PAGE TITLE
        ================================================= */}

        <div className="fade-enter mx-auto max-w-2xl text-center">
          {/* Portal Icon */}
          <div
            className="mx-auto grid h-14 w-14 place-items-center rounded-2xl text-white shadow-lg transition-transform duration-300 hover:scale-105"
            style={{
              background: `linear-gradient(135deg, ${main}, ${
                provider ? C.providerDeep : C.payerDeep
              })`,
              boxShadow: `0 12px 30px -12px ${main}`,
            }}
          >
            <PortalIcon size={26} />
          </div>

          {/* Portal Label */}
          <div
            className={`font-tag mx-auto mt-6 w-fit ${chip}`}
            style={{
              borderColor: line,
              background: soft,
              color: bright,
            }}
          >
            {config.eyebrow}
          </div>

          {/* Title */}
          <h1
            className="font-display mt-4 text-3xl font-bold tracking-tight sm:text-4xl"
            style={{ color: C.text }}
          >
            {config.title}
          </h1>

          {/* Description */}
          <p
            className="mx-auto mt-4 max-w-xl text-sm leading-6"
            style={{ color: C.text2 }}
          >
            {config.description}
          </p>
        </div>

        {/* =================================================
            PORTAL CARDS
        ================================================= */}

        <div className="mt-12 grid gap-5 md:grid-cols-2">
          <PortalOption
            icon={UserCog}
            title={config.adminTitle}
            description={config.adminDescription}
            to={config.adminTo}
            accent={config.accent}
            admin
            delay={80}
          />

          <PortalOption
            icon={Users}
            title={config.staffTitle}
            description={config.staffDescription}
            to={config.staffTo}
            accent={config.accent}
            delay={160}
          />
        </div>
      </main>
    </div>
  );
}

/* =========================================================
   PORTAL OPTION
========================================================= */

function PortalOption({
  icon: Icon,
  title,
  description,
  to,
  accent,
  admin = false,
  delay = 0,
}) {
  const provider = accent === "provider";

  const main = provider ? C.provider : C.payer;
  const deep = provider ? C.providerDeep : C.payerDeep;
  /* Accent text on a dark panel uses the bright end of the ramp. */
  const bright = deep;
  const soft = provider ? C.providerSoft : C.payerSoft;
  const line = provider ? C.providerLine : C.payerLine;

  const [hover, setHover] = useState(false);

  return (
    <Link
      to={to}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="fade-enter group relative overflow-hidden rounded-2xl border p-7 transition-all duration-300"
      style={{
        animationDelay: `${delay}ms`,

        borderColor: hover ? line : C.border,

        background: `linear-gradient(
          180deg,
          ${soft} 0%,
          ${C.surface} 42%,
          ${C.surface} 100%
        )`,

        transform: hover ? "translateY(-4px)" : "none",

        boxShadow: hover
          ? `0 30px 60px -26px ${line}`
          : "0 18px 40px -30px rgba(0,0,0,.8)",
      }}
    >
      {/* =================================================
          CARD AMBIENT GLOW
      ================================================= */}

      <div
        className="pointer-events-none absolute right-0 top-0 h-40 w-40 rounded-full blur-3xl transition-opacity duration-300"
        style={{
          background: main,
          opacity: hover ? 0.16 : 0.08,
        }}
      />

      {/* =================================================
          CARD CONTENT
      ================================================= */}

      <div className="relative">
        {/* Top Row */}
        <div className="flex items-start justify-between">
          {/* Icon */}
          <div
            className="grid h-12 w-12 place-items-center rounded-xl text-white transition-transform duration-300 group-hover:scale-105"
            style={{
              background: `linear-gradient(135deg, ${main}, ${deep})`,
              boxShadow: `0 10px 24px -12px ${main}`,
            }}
          >
            <Icon size={22} />
          </div>

          {/* Administration Badge */}
          {admin && (
            <span
              className={`font-tag ${chip}`}
              style={{
                borderColor: line,
                background: soft,
                color: bright,
              }}
            >
              Administration
            </span>
          )}
        </div>

        {/* Title */}
        <h2
          className="font-display mt-7 text-xl font-bold tracking-tight"
          style={{ color: C.text }}
        >
          {title}
        </h2>

        {/* Description */}
        <p
          className="mt-3 text-sm leading-6"
          style={{ color: C.text2 }}
        >
          {description}
        </p>

        {/* Divider + Continue */}
        <div
          className="mt-7 flex items-center justify-between pt-5"
          style={{
            borderTop: `1px solid ${C.border}`,
          }}
        >
          <span
            className="text-sm font-semibold"
            style={{ color: bright }}
          >
            Continue
          </span>

          <span
            className="grid h-8 w-8 place-items-center rounded-full transition-all duration-200 group-hover:translate-x-0.5"
            style={{
              background: soft,
              color: bright,
            }}
          >
            <ArrowRight size={15} />
          </span>
        </div>
      </div>
    </Link>
  );
}