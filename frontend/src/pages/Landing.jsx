import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  Building2,
  CheckCircle2,
  Clock,
  FileCheck2,
  FileText,
  Linkedin,
  Lock,
  Mail,
  MapPin,
  MessageSquareText,
  Phone,
  Plus,
  Route,
  Shield,
  ShieldCheck,
  Sparkles,
  Twitter,
  UserCheck,
  Workflow,
} from "lucide-react";
import { Link } from "react-router-dom";

/* ============================================================
   DESIGN TOKENS — DARK THEME

   Nothing here is white. `surface` is the panel a card sits on,
   `surfaceAlt` the deeper band behind it.

   provider (Hospital) = violet, payer (Insurance) = blue.

   `*Deep` is the BRIGHTER shade, not the darker one — on a dark
   ground the emphatic end of a ramp is the light end, so
   anywhere the old light theme reached for a 700-weight colour
   to make text stand out, it now reaches up instead of down.
   `*Press` keeps a genuinely darker step for the one place that
   still needs it: a filled button that carries white text.
============================================================ */

const C = {
  bg: "linear-gradient(135deg, #120A21 0%, #0D0719 38%, #090B18 72%, #070A14 100%)",
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
  approve: "#34D399",
  approveSoft: "rgba(52,211,153,0.12)",
  approveLine: "rgba(52,211,153,0.32)",
};

const btn =
  "inline-flex items-center justify-center gap-2 rounded-lg border px-5 py-2.5 text-sm font-semibold transition-all duration-200";
const chip =
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium";

/* A faint dot-grid used as a shared texture so the green backdrop never
   reads as flat — applied once at the page level. */
function GlobalTexture() {
  return (
    <div
      className="pointer-events-none fixed inset-0 -z-50 opacity-[0.45]"
      style={{
        backgroundImage:
          "radial-gradient(rgba(196,181,253,.07) 1px, transparent 1px)",
        backgroundSize: "28px 28px",
      }}
    />
  );
}

/* ============================================================
   GLOBAL STYLE
============================================================ */

function LandingStyle() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

      .font-display { font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif; }
      .font-body { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
      .font-tag { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes floatSlow {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
      }
      @keyframes floatSlowDelayed {
        0%, 100% { transform: translateY(0) rotate(2.5deg); }
        50% { transform: translateY(-7px) rotate(2.5deg); }
      }
      @keyframes floatDoc {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
      }
      @keyframes trackPulse {
        0%   { transform: translateX(0); }
        45%  { transform: translateX(calc(100% - 1.1rem)); }
        55%  { transform: translateX(calc(100% - 1.1rem)); }
        100% { transform: translateX(0); }
      }
      @keyframes trackFill {
        0%   { width: 0%; }
        45%  { width: 100%; }
        100% { width: 100%; }
      }
      @keyframes ringPulse {
        0%   { box-shadow: 0 0 0 0 rgba(167,139,250,.45); }
        70%  { box-shadow: 0 0 0 8px rgba(167,139,250,0); }
        100% { box-shadow: 0 0 0 0 rgba(167,139,250,0); }
      }
      @keyframes ringExpand {
        0%   { transform: translate(-50%, -50%) scale(0.6); opacity: .5; }
        100% { transform: translate(-50%, -50%) scale(1.9); opacity: 0; }
      }
      @keyframes shieldDrift {
        0%, 100% { transform: translateY(0) rotate(-2deg); }
        50% { transform: translateY(-10px) rotate(2deg); }
      }

      .reveal {
        opacity: 0;
        transform: translateY(18px);
        transition: opacity .6s cubic-bezier(.2,.7,.2,1), transform .6s cubic-bezier(.2,.7,.2,1);
      }
      .reveal.is-visible { opacity: 1; transform: translateY(0); }
      .hero-enter { animation: fadeUp .8s cubic-bezier(.2,.7,.2,1) both; }

      @media (prefers-reduced-motion: reduce) {
        .reveal { opacity: 1 !important; transform: none !important; transition: none !important; }
        .hero-enter { animation: none !important; }
        *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; }
      }
    `}</style>
  );
}

/* ============================================================
   MOTION UTILITIES
============================================================ */

function useInView(options = { threshold: 0.15 }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      setVisible(true);
      return;
    }

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setVisible(true);
        observer.disconnect();
      }
    }, options);

    observer.observe(node);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return [ref, visible];
}

function Reveal({ as: Tag = "div", delay = 0, className = "", children }) {
  const [ref, visible] = useInView();
  return (
    <Tag
      ref={ref}
      className={`reveal ${visible ? "is-visible" : ""} ${className}`}
      style={{ transitionDelay: visible ? `${delay}ms` : "0ms" }}
    >
      {children}
    </Tag>
  );
}

function useCountUp(target, visible, duration = 1200) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!visible) return;
    const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) {
      setValue(target);
      return;
    }

    let start = null;
    const step = (ts) => {
      if (start === null) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) requestAnimationFrame(step);
    };

    const frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [visible, target, duration]);

  return value;
}

/* A small centered eyebrow that sits ON a divider line, so section
   breaks read as a deliberate label rather than dead space. */
function SectionRule({ label }) {
  return (
    <div className="relative flex items-center justify-center">
      <div className="absolute inset-x-0 top-1/2 h-px" style={{ background: C.border }} />
      <span
        className="font-tag relative px-4 text-[10px] font-semibold uppercase tracking-[.16em]"
        style={{ background: C.bg, color: C.text3 }}
      >
        {label}
      </span>
    </div>
  );
}

/* ============================================================
   PAGE
============================================================ */

export default function Landing() {
  return (
    <div className="font-body relative min-h-screen" style={{ background: C.bg }}>
      <LandingStyle />
      <GlobalTexture />
      <SiteHeader />

      <main>
        <Hero />
        <SectionRule label="Two portals, one workflow" />
        <PortalSection />
        <SectionRule label="How it works" />
        <WorkflowSection />
        <TrustSection />
        <MetricsSection />
      </main>

      <SiteFooter />
    </div>
  );
}

/* ============================================================
   HEADER
============================================================ */

function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className="sticky top-0 z-30 backdrop-blur-xl transition-shadow duration-300"
      style={{
        background: "linear-gradient(180deg, rgba(12,8,22,0.92), rgba(9,11,24,0.82))",
        borderBottom: `1px solid ${scrolled ? C.borderStrong : "transparent"}`,
        boxShadow: scrolled ? "0 10px 30px -18px rgba(0,0,0,.85)" : "none",
      }}
    >
      <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <div
            className="grid h-9 w-9 place-items-center rounded-lg text-white transition-transform duration-300 hover:rotate-6"
            style={{ background: `linear-gradient(135deg, ${C.provider}, ${C.payer})` }}
          >
            <ShieldCheck size={19} />
          </div>
          <div>
            <div className="font-display text-sm font-bold tracking-tight" style={{ color: C.text }}>
              PriorAuth AI
            </div>
            <div className="font-tag text-[10px] font-medium uppercase tracking-[.14em]" style={{ color: C.text3 }}>
              Authorization intelligence
            </div>
          </div>
        </div>

        <nav className="hidden items-center gap-8 md:flex">
          <a href="#workflow" className="text-[13px] font-medium transition-colors hover:opacity-80" style={{ color: C.text2 }}>
            How it works
          </a>
          <a href="#trust" className="text-[13px] font-medium transition-colors hover:opacity-80" style={{ color: C.text2 }}>
            Security &amp; compliance
          </a>
          <a href="#contact" className="text-[13px] font-medium transition-colors hover:opacity-80" style={{ color: C.text2 }}>
            Contact
          </a>
        </nav>

        <div className="flex items-center gap-3">
          <span className={`hidden sm:inline-flex ${chip}`} style={{ borderColor: C.approveLine, background: C.approveSoft, color: C.approve }}>
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: C.approve }} />
              <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: C.approve }} />
            </span>
            Platform operational
          </span>

          <Link to="/hospital/portal" className={btn} style={{ borderColor: C.borderStrong, background: C.surface, color: C.text }}>
            Sign in
          </Link>
        </div>
      </div>
    </header>
  );
}

/* ============================================================
   HERO
============================================================ */

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div
        className="absolute inset-0 -z-10"
        style={{ background: `radial-gradient(ellipse 900px 500px at 50% -5%, ${C.providerSoft}, transparent 60%)` }}
      />
      <div
        className="absolute inset-0 -z-10"
        style={{ background: `radial-gradient(ellipse 700px 400px at 85% 30%, ${C.payerSoft}, transparent 60%)` }}
      />

      {/* Wide relative frame so the side clusters have room to sit outside
          the 860px reading column without being clipped. */}
      <div className="relative mx-auto max-w-[1440px]">
        <HeroLeftCluster />
        <HeroRightCluster />

        <div className="relative mx-auto max-w-[860px] px-6 pb-14 pt-14 text-center lg:pb-16 lg:pt-16">
          <div
            className="hero-enter font-tag inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[.12em]"
            style={{ animationDelay: "0ms", borderColor: C.providerLine, background: C.providerSoft, color: C.providerDeep }}
          >
            Built for hospital and insurance teams
          </div>

          <h1
            className="hero-enter font-display mx-auto mt-6 max-w-[760px] text-[2.75rem] font-bold leading-[1.08] tracking-[-.03em] sm:text-6xl"
            style={{ animationDelay: "80ms", color: C.text }}
          >
            Prior authorization,
            <br />
            <span style={{ background: `linear-gradient(90deg, ${C.providerDeep}, ${C.payerDeep})`, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
              decided in minutes.
            </span>
          </h1>

          <p className="hero-enter mx-auto mt-6 max-w-xl text-[15px] leading-7" style={{ animationDelay: "160ms", color: C.text2 }}>
            Submit a request, see exactly where it stands, and get a clear
            answer — without days of phone calls and faxed paperwork.
          </p>

          <div className="hero-enter mt-8 flex flex-wrap items-center justify-center gap-4" style={{ animationDelay: "240ms" }}>
            <Link
              to="/hospital/signup"
              className={`${btn} group`}
              style={{ borderColor: C.provider, background: `linear-gradient(135deg, ${C.provider}, ${C.payer})`, color: "#FFFFFF" }}
              onMouseEnter={(e) => (e.currentTarget.style.filter = "brightness(0.92)")}
              onMouseLeave={(e) => (e.currentTarget.style.filter = "brightness(1)")}
            >
              Request a demo
              <ArrowRight size={15} className="transition-transform duration-200 group-hover:translate-x-0.5" />
            </Link>
            <a href="#workflow" className="text-sm font-semibold transition-colors hover:opacity-80" style={{ color: C.text2 }}>
              See how it works
            </a>
          </div>

          <div className="hero-enter mx-auto mt-10 max-w-sm" style={{ animationDelay: "320ms" }}>
            <ClaimTrack />
          </div>

          <dl className="mx-auto mt-10 grid max-w-lg grid-cols-3 gap-6 pt-6" style={{ borderTop: `1px solid ${C.border}` }}>
            <div>
              <dt className="font-tag text-[10px] font-medium uppercase tracking-wide" style={{ color: C.text3 }}>Requests handled</dt>
              <dd className="font-display mt-1 text-2xl font-bold tabular-nums" style={{ color: C.text }}>1.2M+</dd>
            </div>
            <div>
              <dt className="font-tag text-[10px] font-medium uppercase tracking-wide" style={{ color: C.text3 }}>Faster answers</dt>
              <dd className="font-display mt-1 text-2xl font-bold tabular-nums" style={{ color: C.text }}>40%</dd>
            </div>
            <div>
              <dt className="font-tag text-[10px] font-medium uppercase tracking-wide" style={{ color: C.text3 }}>Organizations</dt>
              <dd className="font-display mt-1 text-2xl font-bold tabular-nums" style={{ color: C.text }}>310+</dd>
            </div>
          </dl>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------
   HERO — LEFT CLUSTER
   A faint shield-and-cross watermark with three floating
   workflow cards drifting over it, echoing the provider side
   of the product (document intake → validation → routing).
------------------------------------------------------------ */

function HeroLeftCluster() {
  const cards = [
    {
      icon: FileCheck2,
      title: "Upload Documents",
      text: "Clinical notes, labs, imaging & more",
      top: "14%",
      left: "6%",
      delay: "0s",
    },
    {
      icon: BrainCircuit,
      title: "AI Validation",
      text: "Checks completeness & guideline match",
      top: "44%",
      left: "13%",
      delay: "1.1s",
    },
    {
      icon: Route,
      title: "Smart Routing",
      text: "Right payer. Right team. Right time.",
      top: "74%",
      left: "5%",
      delay: "2.2s",
    },
  ];

  return (
    <div className="pointer-events-none absolute left-0 top-0 hidden h-[640px] w-[30%] xl:block" aria-hidden="true">
      {/* Bold curved gradient stroke threading behind the cards */}
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 420 640" fill="none" preserveAspectRatio="none">
        <path
          d="M -20 60 C 140 40, 120 220, 260 260 C 380 292, 200 420, 300 520 C 350 570, 300 590, 260 610"
          stroke="url(#leftLine)"
          strokeWidth="2.5"
          fill="none"
        />
        <defs>
          <linearGradient id="leftLine" x1="0" y1="0" x2="0" y2="640" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor={C.providerDeep} stopOpacity="0" />
            <stop offset="0.5" stopColor={C.providerDeep} stopOpacity="0.9" />
            <stop offset="1" stopColor={C.payerDeep} stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>

      {/* Watermark shield with a medical cross, drifting slowly */}
      <div
        className="absolute left-[2%] top-[6%] h-28 w-28"
        style={{ animation: "shieldDrift 7s ease-in-out infinite" }}
      >
        <Shield size={112} strokeWidth={1.6} style={{ color: C.providerDeep, opacity: 0.9 }} />
        <Plus
          size={34}
          strokeWidth={2}
          className="absolute left-1/2 top-[42%] -translate-x-1/2 -translate-y-1/2"
          style={{ color: C.providerDeep, opacity: 0.9 }}
        />
      </div>

      {cards.map((card, i) => (
        <div
          key={card.title}
          className="absolute w-[200px] rounded-xl border-2 p-4"
          style={{
            top: card.top,
            left: card.left,
            borderColor: C.providerLine,
            background: `linear-gradient(160deg, ${C.surface}, ${C.surfaceAlt})`,
            boxShadow: `0 20px 45px -22px rgba(0,0,0,.95), 0 0 0 1px ${C.providerSoft}`,
            animation: `floatSlow ${5.5 + i * 0.6}s ease-in-out infinite`,
            animationDelay: card.delay,
          }}
        >
          <div className="flex items-start gap-2.5">
            <div
              className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-white"
              style={{ background: `linear-gradient(135deg, ${C.provider}, ${C.providerDeep})` }}
            >
              <card.icon size={16} />
            </div>
            <div>
              <div className="font-display text-[13px] font-bold leading-tight" style={{ color: C.text }}>
                {card.title}
              </div>
              <div className="mt-1 text-[11.5px] font-medium leading-[1.4]" style={{ color: C.text2 }}>
                {card.text}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------
   HERO — RIGHT CLUSTER
   A tilted "product" panel showing a live request overview,
   plus a floating approved-document card breaking its edge —
   the payer side of the same workflow, mid-decision.
------------------------------------------------------------ */

function HeroRightCluster() {
  const timeline = [
    { label: "Request submitted", meta: "2m ago" },
    { label: "Under clinical review", meta: "3m ago" },
    { label: "Approved", meta: "4m 12s" },
  ];

  return (
    <div className="pointer-events-none absolute right-0 top-0 hidden h-[640px] w-[34%] xl:block" aria-hidden="true">
      <div
        className="absolute right-[6%] top-[10%] w-[330px] rounded-2xl border p-5"
        style={{
          borderColor: C.border,
          background: `linear-gradient(165deg, ${C.surface}, ${C.surfaceAlt})`,
          boxShadow: "0 40px 80px -30px rgba(0,0,0,.9)",
          transform: "rotate(2.5deg)",
          animation: "floatSlowDelayed 9s ease-in-out infinite",
        }}
      >
        <div className="flex items-center justify-between">
          <span className="font-tag text-[11px] font-semibold uppercase tracking-[.1em]" style={{ color: C.text3 }}>
            Request Overview
          </span>
          <div className="flex items-center gap-1.5" style={{ color: C.text3 }}>
            <Sparkles size={13} />
          </div>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2">
          <div className="rounded-lg border p-2.5" style={{ borderColor: C.border, background: C.bg }}>
            <div className="text-[9.5px] font-medium" style={{ color: C.text3 }}>Overall Status</div>
            <div
              className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-bold"
              style={{ color: C.approve }}
            >
              <CheckCircle2 size={12} />
              Approved
            </div>
          </div>
          <div className="rounded-lg border p-2.5" style={{ borderColor: C.border, background: C.bg }}>
            <div className="text-[9.5px] font-medium" style={{ color: C.text3 }}>Turnaround Time</div>
            <div className="font-display mt-1.5 text-[13px] font-bold" style={{ color: C.text }}>4m 12s</div>
            <div className="text-[9.5px] font-medium" style={{ color: C.approve }}>-35% faster</div>
          </div>
          <div className="rounded-lg border p-2.5" style={{ borderColor: C.border, background: C.bg }}>
            <div className="text-[9.5px] font-medium" style={{ color: C.text3 }}>Success Rate</div>
            <div className="font-display mt-1.5 text-[13px] font-bold" style={{ color: C.text }}>98%</div>
            <div className="text-[9.5px] font-medium" style={{ color: C.approve }}>+12% this month</div>
          </div>
        </div>

        <div className="mt-4 flex items-start gap-4 border-t pt-4" style={{ borderColor: C.border }}>
          <div className="flex-1">
            <div className="font-tag text-[10px] font-semibold uppercase tracking-wide" style={{ color: C.text3 }}>
              Activity Timeline
            </div>
            <ul className="mt-2.5 space-y-2">
              {timeline.map((item) => (
                <li key={item.label} className="flex items-center justify-between gap-3 text-[11px]">
                  <span className="flex items-center gap-1.5" style={{ color: C.text2 }}>
                    <CheckCircle2 size={11} style={{ color: C.approve }} />
                    {item.label}
                  </span>
                  <span className="font-tag shrink-0" style={{ color: C.text3 }}>{item.meta}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Small static donut built from a conic-gradient, no chart lib needed */}
          <div
            className="grid h-14 w-14 shrink-0 place-items-center rounded-full"
            style={{ background: `conic-gradient(${C.payer} 0% 65%, ${C.provider} 65% 100%)` }}
          >
            <div className="grid h-8 w-8 place-items-center rounded-full" style={{ background: C.surfaceAlt }}>
              <BarChart3 size={13} style={{ color: C.text3 }} />
            </div>
          </div>
        </div>
      </div>

      {/* Floating approved-document card, overlapping the panel's lower edge */}
      <div
        className="absolute bottom-[12%] right-[16%]"
        style={{ animation: "floatDoc 6.5s ease-in-out infinite", animationDelay: "0.6s" }}
      >
        <div className="relative">
          {/* Concentric circular ripples, staggered so only one is ever prominent */}
          <span
            className="absolute left-1/2 top-1/2 h-16 w-16 rounded-full"
            style={{ background: C.providerSoft, animation: "ringExpand 3.2s ease-out infinite" }}
          />
          <span
            className="absolute left-1/2 top-1/2 h-16 w-16 rounded-full"
            style={{ background: C.providerSoft, animation: "ringExpand 3.2s ease-out infinite", animationDelay: "1.6s" }}
          />

          <div
            className="relative grid h-20 w-16 place-items-center rounded-2xl border"
            style={{
              borderColor: C.providerLine,
              background: `linear-gradient(160deg, ${C.surface}, ${C.surfaceAlt})`,
              boxShadow: `0 25px 50px -20px ${C.providerLine}`,
            }}
          >
            <FileText size={20} style={{ color: C.providerDeep }} />
            <div
              className="absolute -bottom-2 -right-2 grid h-7 w-7 place-items-center rounded-full border-2"
              style={{ borderColor: C.bg, background: C.approve, color: "#07130D" }}
            >
              <CheckCircle2 size={14} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ClaimTrack() {
  const stages = ["Submitted", "Reviewing", "Decided"];
  return (
    <div
      className="rounded-2xl border p-5 text-left"
      style={{ borderColor: C.border, background: C.surface, boxShadow: "0 24px 55px -28px rgba(0,0,0,.85)", animation: "floatSlow 6s ease-in-out infinite" }}
    >
      <div className="flex items-center justify-between">
        <span className="font-tag text-[11px] font-semibold uppercase tracking-[.1em]" style={{ color: C.text3 }}>Claim #48213</span>
        <span className="font-tag inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold" style={{ background: C.approveSoft, color: C.approve }}>
          <CheckCircle2 size={11} />
          Approved · 4m 12s
        </span>
      </div>

      <div className="relative mt-5 h-1.5 overflow-hidden rounded-full" style={{ background: C.surfaceAlt }}>
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ background: `linear-gradient(90deg, ${C.provider}, ${C.payer})`, animation: "trackFill 5.2s cubic-bezier(.4,0,.2,1) infinite" }} />
        <div className="absolute inset-y-0 left-0 h-1.5 w-1.5 rounded-full" style={{ background: C.providerDeep, animation: "trackPulse 5.2s cubic-bezier(.4,0,.2,1) infinite, ringPulse 1.6s ease-out infinite" }} />
      </div>

      <div className="mt-3 grid grid-cols-3 text-[11px] font-medium" style={{ color: C.text3 }}>
        {stages.map((stage) => (
          <span key={stage} className={stage === "Submitted" ? "text-left" : stage === "Decided" ? "text-right" : "text-center"}>{stage}</span>
        ))}
      </div>
    </div>
  );
}

/* ============================================================
   PORTAL CARDS
============================================================ */

function PortalSection() {
  return (
    <section style={{ background: C.bg }}>
      <div className="mx-auto max-w-[1200px] px-6 py-14 lg:px-8 lg:py-16">
        <Reveal className="mx-auto max-w-xl text-center">
          <h2 className="font-display text-2xl font-bold tracking-tight sm:text-3xl" style={{ color: C.text }}>
            Choose your portal
          </h2>
          <p className="mt-2 text-[14px]" style={{ color: C.text2 }}>
            One shared workflow, built separately for each side of the desk.
          </p>
        </Reveal>

        <div className="mx-auto mt-10 grid max-w-[860px] gap-5 sm:grid-cols-2">
          <Reveal delay={80}>
            <PortalCard
              to="/hospital/portal"
              signupTo="/hospital/signup"
              icon={Building2}
              eyebrow="Hospital"
              title="Submit a request"
              description="Upload paperwork once — we handle the data entry."
              accent="provider"
              points={[
                { icon: FileCheck2, text: "No manual re-typing" },
                { icon: Clock, text: "Real-time status, no phone calls" },
                { icon: CheckCircle2, text: "Instant answers on simple cases" },
              ]}
            />
          </Reveal>

          <Reveal delay={160}>
            <PortalCard
              to="/payer/portal"
              signupTo="/payer/signup"
              icon={ShieldCheck}
              eyebrow="Insurance"
              title="Review requests"
              description="See what needs attention first, and why."
              accent="payer"
              points={[
                { icon: Clock, text: "Urgent cases rise to the top" },
                { icon: UserCheck, text: "Routed to the right specialist" },
                { icon: MessageSquareText, text: "Plain-language reasoning" },
              ]}
            />
          </Reveal>
        </div>
      </div>
    </section>
  );
}

function PortalCard({ to, signupTo, icon: Icon, eyebrow, title, description, accent, points }) {
  const provider = accent === "provider";
  const main = provider ? C.provider : C.payer;
  const deep = provider ? C.providerDeep : C.payerDeep;
  const press = provider ? C.providerPress : C.payerPress;
  const soft = provider ? C.providerSoft : C.payerSoft;
  const line = provider ? C.providerLine : C.payerLine;
  const [hover, setHover] = useState(false);
  const [btnHover, setBtnHover] = useState(false);

  return (
    <section
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="group relative flex h-full flex-col overflow-hidden rounded-2xl border transition-all duration-300"
      style={{
        borderColor: hover ? line : C.border,
        background: `linear-gradient(180deg, ${soft} 0%, ${C.surface} 30%)`,
        transform: hover ? "translateY(-4px)" : "none",
        boxShadow: hover
          ? `0 30px 60px -26px ${line}`
          : "0 18px 40px -30px rgba(0,0,0,.8)",
      }}
    >
      <div className="flex flex-1 flex-col p-6">
        <div className="flex items-center gap-3">
          <div
            className="grid h-11 w-11 shrink-0 place-items-center rounded-xl text-white transition-transform duration-300 group-hover:scale-105"
            style={{ background: `linear-gradient(135deg, ${main}, ${deep})` }}
          >
            <Icon size={19} />
          </div>
          <div>
            <span className="font-tag block text-[10px] font-semibold uppercase tracking-wide" style={{ color: deep }}>{eyebrow}</span>
            <h3 className="font-display text-lg font-bold leading-tight tracking-tight" style={{ color: C.text }}>{title}</h3>
          </div>
        </div>

        <p className="mt-3 text-[13px] leading-5" style={{ color: C.text2 }}>{description}</p>

        <ul className="mt-5 space-y-2.5 pt-5" style={{ borderTop: `1px solid ${C.border}` }}>
          {points.map((point) => (
            <li key={point.text} className="flex items-center gap-2.5 text-[13px] leading-5" style={{ color: C.text2 }}>
              <point.icon size={14} className="shrink-0" style={{ color: deep }} />
              {point.text}
            </li>
          ))}
        </ul>

        <div className="mt-6 flex-1" />

        <Link
          to={to}
          onMouseEnter={() => setBtnHover(true)}
          onMouseLeave={() => setBtnHover(false)}
          className="group/btn relative flex items-center justify-between overflow-hidden rounded-xl border px-4 py-3.5 transition-all duration-200"
          style={{
            borderColor: main,
            background: btnHover
              ? `linear-gradient(135deg, ${deep}, ${main})`
              : `linear-gradient(135deg, ${main}, ${press})`,
            transform: btnHover ? "translateY(-1px)" : "none",
          }}
        >
          <span className="font-display text-[15px] font-bold tracking-tight" style={{ color: "#FFFFFF" }}>
            {eyebrow} Portal
          </span>
          <span
            className="grid h-7 w-7 place-items-center rounded-full transition-transform duration-200 group-hover/btn:translate-x-0.5"
            style={{ background: "rgba(255,255,255,0.22)" }}
          >
            <ArrowRight size={14} style={{ color: "#FFFFFF" }} />
          </span>
        </Link>

        <Link
          to={signupTo}
          className="mt-3 text-center text-[12px] font-medium transition-colors hover:opacity-80"
          style={{ color: C.text3 }}
        >
          New here? <span style={{ color: deep }}>Create an account</span>
        </Link>
      </div>
    </section>
  );
}

/* ============================================================
   WORKFLOW
============================================================ */

function WorkflowSection() {
  const steps = [
    { icon: FileCheck2, title: "01 — Extract", text: "Structured fields — patient, diagnosis, requested treatment, coding — are pulled from the submitted packet in seconds." },
    { icon: BrainCircuit, title: "02 — Evaluate", text: "The request is checked against medical-necessity criteria and scored for urgency and appeal risk, with the reasoning shown alongside." },
    { icon: Workflow, title: "03 — Route & decide", text: "Straightforward cases resolve automatically. Complex ones are routed to the reviewer with the right specialty." },
  ];

  const [lineRef, lineVisible] = useInView({ threshold: 0.4 });

  return (
    <section id="workflow" style={{ background: C.bg }}>
      <div className="mx-auto max-w-[1200px] px-6 py-14 lg:px-8 lg:py-16">
        <div ref={lineRef} className="relative grid gap-8 md:grid-cols-3">
          <div
            className="absolute left-0 right-0 top-6 hidden h-px origin-left transition-transform duration-[1200ms] ease-out md:block"
            style={{ background: C.border, transform: lineVisible ? "scaleX(1)" : "scaleX(0)" }}
          />
          <div
            className="absolute left-0 right-0 top-6 hidden h-px origin-left transition-transform duration-[1400ms] ease-out md:block"
            style={{ background: `linear-gradient(90deg, ${C.provider}, ${C.payer})`, transform: lineVisible ? "scaleX(1)" : "scaleX(0)", transitionDelay: "150ms" }}
          />

          {steps.map((step, i) => (
            <Reveal key={step.title} delay={i * 120} className="relative">
              <div className="relative z-10 grid h-12 w-12 place-items-center rounded-xl border transition-transform duration-300 hover:-translate-y-1 hover:scale-105" style={{ borderColor: C.border, background: C.surface, color: C.providerDeep }}>
                <step.icon size={20} />
              </div>
              <h3 className="font-tag mt-5 text-sm font-bold uppercase tracking-wide" style={{ color: C.text3 }}>{step.title}</h3>
              <p className="mt-2 text-[13px] leading-6" style={{ color: C.text2 }}>{step.text}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ============================================================
   TRUST / COMPLIANCE
============================================================ */

function TrustSection() {
  const badges = [
    { label: "HIPAA compliant", icon: ShieldCheck },
    { label: "SOC 2 Type II", icon: Lock },
    { label: "256-bit encryption", icon: FileCheck2 },
    { label: "99.95% uptime SLA", icon: Clock },
  ];

  return (
    <section id="trust" style={{ borderTop: `1px solid ${C.border}`, background: C.surfaceAlt }}>
      <div className="mx-auto max-w-[1200px] px-6 py-10 lg:px-8">
        <Reveal className="flex flex-wrap items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg" style={{ background: C.surface, color: C.text2 }}>
              <Lock size={17} />
            </div>
            <div>
              <div className="font-display text-sm font-bold" style={{ color: C.text }}>Built to hospital and insurance security standards</div>
              <div className="text-[13px]" style={{ color: C.text2 }}>Every case is encrypted in transit and at rest, with a full audit trail.</div>
            </div>
          </div>

          <div className="flex flex-wrap gap-2.5">
            {badges.map((badge) => (
              <span key={badge.label} className={`font-tag ${chip}`} style={{ borderColor: C.border, background: C.surface, color: C.text2 }}>
                <badge.icon size={12} className="mr-1 inline-block -mt-0.5" />
                {badge.label}
              </span>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}

/* ============================================================
   METRICS
============================================================ */

function MetricsSection() {
  const metrics = [
    { value: 1.2, suffix: "M+", decimals: 1, label: "Authorizations processed annually" },
    { value: 40, suffix: "%", decimals: 0, label: "Faster time-to-decision" },
    { value: 98.4, suffix: "%", decimals: 1, label: "Field extraction accuracy" },
    { value: 310, suffix: "+", decimals: 0, label: "Hospital and insurance organizations" },
  ];

  const [ref, visible] = useInView({ threshold: 0.3 });

  return (
    <section style={{ background: C.bg }}>
      <div className="mx-auto max-w-[1200px] px-6 py-14 lg:px-8 lg:py-16">
        <Reveal>
          <div
            ref={ref}
            className="grid gap-8 rounded-2xl border p-8 sm:grid-cols-2 lg:grid-cols-4 lg:p-10"
            style={{ borderColor: C.border, background: `linear-gradient(135deg, ${C.providerSoft}, ${C.surface} 40%, ${C.payerSoft})` }}
          >
            {metrics.map((metric) => (
              <MetricStat key={metric.label} metric={metric} visible={visible} />
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function MetricStat({ metric, visible }) {
  const animated = useCountUp(metric.value, visible);
  return (
    <div>
      <div className="font-display text-3xl font-bold tracking-tight tabular-nums" style={{ color: C.text }}>
        {animated.toFixed(metric.decimals)}{metric.suffix}
      </div>
      <div className="mt-1.5 text-[13px]" style={{ color: C.text2 }}>{metric.label}</div>
    </div>
  );
}

/* ============================================================
   FOOTER
============================================================ */

function SiteFooter() {
  return (
    <footer id="contact" style={{ borderTop: `1px solid ${C.border}`, background: C.surfaceAlt }}>
      <div className="mx-auto max-w-[1200px] px-6 py-12 lg:px-8">
        <div className="grid gap-10 lg:grid-cols-[1.3fr_1fr_1fr_1.2fr]">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="grid h-9 w-9 place-items-center rounded-lg text-white" style={{ background: `linear-gradient(135deg, ${C.provider}, ${C.payer})` }}>
                <ShieldCheck size={19} />
              </div>
              <div className="font-display text-sm font-bold tracking-tight" style={{ color: C.text }}>PriorAuth AI</div>
            </div>

            <p className="mt-4 max-w-xs text-[13px] leading-6" style={{ color: C.text2 }}>
              Authorization intelligence for hospital and insurance teams —
              built to make medical-necessity decisions faster and easier
              to explain.
            </p>

            <div className="mt-5 flex items-center gap-3">
              <a href="https://linkedin.com" aria-label="LinkedIn" className="grid h-8 w-8 place-items-center rounded-lg border transition-all duration-200 hover:-translate-y-0.5" style={{ borderColor: C.border, color: C.text3 }}>
                <Linkedin size={15} />
              </a>
              <a href="https://twitter.com" aria-label="Twitter" className="grid h-8 w-8 place-items-center rounded-lg border transition-all duration-200 hover:-translate-y-0.5" style={{ borderColor: C.border, color: C.text3 }}>
                <Twitter size={15} />
              </a>
            </div>
          </div>

          <div>
            <div className="font-tag text-[10px] font-semibold uppercase tracking-wide" style={{ color: C.text3 }}>Product</div>
            <ul className="mt-4 space-y-2.5 text-[13px]" style={{ color: C.text2 }}>
              <li><a href="#workflow" className="hover:opacity-80">How it works</a></li>
              <li><Link to="/hospital/portal" className="hover:opacity-80">Hospital portal</Link></li>
              <li><Link to="/payer/portal" className="hover:opacity-80">Insurance portal</Link></li>
              <li><a href="#trust" className="hover:opacity-80">Security</a></li>
            </ul>
          </div>

          <div>
            <div className="font-tag text-[10px] font-semibold uppercase tracking-wide" style={{ color: C.text3 }}>Company</div>
            <ul className="mt-4 space-y-2.5 text-[13px]" style={{ color: C.text2 }}>
              <li><a href="#" className="hover:opacity-80">About</a></li>
              <li><a href="#" className="hover:opacity-80">Careers</a></li>
              <li><a href="#" className="hover:opacity-80">Press</a></li>
              <li><a href="#" className="hover:opacity-80">Privacy policy</a></li>
            </ul>
          </div>

          <div>
            <div className="font-tag text-[10px] font-semibold uppercase tracking-wide" style={{ color: C.text3 }}>Contact</div>
            <ul className="mt-4 space-y-3 text-[13px]" style={{ color: C.text2 }}>
              <li className="flex items-start gap-2.5">
                <MapPin size={15} className="mt-0.5 shrink-0" style={{ color: C.text3 }} />
                548 Market Street, Suite 91000<br />San Francisco, CA 94104
              </li>
              <li className="flex items-center gap-2.5">
                <Phone size={15} className="shrink-0" style={{ color: C.text3 }} />
                <a href="tel:+18005550142" className="hover:opacity-80">+1 (800) 555-0142</a>
              </li>
              <li className="flex items-center gap-2.5">
                <Mail size={15} className="shrink-0" style={{ color: C.text3 }} />
                <a href="mailto:hello@priorauth.ai" className="hover:opacity-80">hello@priorauth.ai</a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-3 pt-6 text-[11px] sm:flex-row" style={{ borderTop: `1px solid ${C.border}`, color: C.text3 }}>
          <span>© 2026 PriorAuth AI, Inc. All rights reserved.</span>
          <span>HIPAA compliant · SOC 2 Type II certified</span>
        </div>
      </div>
    </footer>
  );
}











