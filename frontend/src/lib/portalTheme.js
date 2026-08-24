/* =========================================================
   PORTAL THEME — DARK

   The signed-in portals run on a dark canvas. Nothing here is
   white: every surface is a tinted dark panel and every border
   is a low-alpha accent line, so a "box" never reads as a sheet
   of paper dropped on the page.

   Each portal carries its own accent hue:

     hospital / provider  ->  violet
     insurance / payer    ->  blue

   Shared by Layout, Chatbot, ProviderDashboard and
   PayerDashboard so the chrome and the page agree on one
   palette instead of each file inventing its own.
   ========================================================= */

export const PORTAL_THEME = {

  /* ---------- HOSPITAL / PROVIDER — VIOLET ---------- */

  provider: {
    key: "provider",

    accent: "#A78BFA",
    accentDeep: "#7C3AED",
    accentBright: "#C4B5FD",

    accentSoft: "rgba(167,139,250,0.13)",
    accentLine: "rgba(167,139,250,0.32)",
    accentGlow: "rgba(124,58,237,0.45)",

    text: "#EFEAFB",
    text2: "#B3A6D6",
    text3: "#8A7CAC",

    line: "rgba(167,139,250,0.14)",
    lineStrong: "rgba(167,139,250,0.26)",

    /* Panels are translucent so the page glow shows through
       instead of sitting behind an opaque block. */
    panel: "rgba(255,255,255,0.045)",
    panelRaised: "rgba(255,255,255,0.075)",
    panelHover: "rgba(255,255,255,0.10)",

    track: "rgba(167,139,250,0.16)",

    headerBg: "rgba(12,8,22,0.86)",

    sidebar: `
      radial-gradient(
        ellipse 420px 320px at 20% 0%,
        rgba(124,58,237,0.32),
        transparent 70%
      ),
      radial-gradient(
        ellipse 420px 360px at 90% 22%,
        rgba(167,139,250,0.15),
        transparent 72%
      ),
      linear-gradient(
        160deg,
        #150E28 0%,
        #110A20 52%,
        #0B0716 100%
      )
    `,

    page: `
      radial-gradient(
        ellipse 1100px 620px at 0% 0%,
        rgba(124,58,237,0.26),
        transparent 68%
      ),
      radial-gradient(
        ellipse 900px 560px at 100% 12%,
        rgba(167,139,250,0.14),
        transparent 66%
      ),
      radial-gradient(
        ellipse 1000px 520px at 55% 100%,
        rgba(91,33,182,0.22),
        transparent 70%
      ),
      linear-gradient(
        160deg,
        #120A21 0%,
        #0D0719 48%,
        #0A0614 100%
      )
    `,
  },

  /* ---------- INSURANCE / PAYER — BLUE ---------- */

  payer: {
    key: "payer",

    accent: "#60A5FA",
    accentDeep: "#2563EB",
    accentBright: "#93C5FD",

    accentSoft: "rgba(96,165,250,0.13)",
    accentLine: "rgba(96,165,250,0.32)",
    accentGlow: "rgba(37,99,235,0.45)",

    text: "#E9F1FF",
    text2: "#A6BCDA",
    text3: "#7B90B0",

    line: "rgba(96,165,250,0.14)",
    lineStrong: "rgba(96,165,250,0.26)",

    panel: "rgba(255,255,255,0.045)",
    panelRaised: "rgba(255,255,255,0.075)",
    panelHover: "rgba(255,255,255,0.10)",

    track: "rgba(96,165,250,0.16)",

    headerBg: "rgba(6,11,24,0.86)",

    sidebar: `
      radial-gradient(
        ellipse 420px 320px at 20% 0%,
        rgba(37,99,235,0.34),
        transparent 70%
      ),
      radial-gradient(
        ellipse 420px 360px at 90% 22%,
        rgba(96,165,250,0.15),
        transparent 72%
      ),
      linear-gradient(
        160deg,
        #0C1730 0%,
        #091124 52%,
        #060C1A 100%
      )
    `,

    page: `
      radial-gradient(
        ellipse 1100px 620px at 0% 0%,
        rgba(37,99,235,0.26),
        transparent 68%
      ),
      radial-gradient(
        ellipse 900px 560px at 100% 12%,
        rgba(96,165,250,0.14),
        transparent 66%
      ),
      radial-gradient(
        ellipse 1000px 520px at 55% 100%,
        rgba(30,64,175,0.24),
        transparent 70%
      ),
      linear-gradient(
        160deg,
        #0A1428 0%,
        #071021 48%,
        #050B18 100%
      )
    `,
  },
};

export const themeFor = (portal) =>
  portal === "payer"
    ? PORTAL_THEME.payer
    : PORTAL_THEME.provider;

/* =========================================================
   SEMANTIC TONES

   Tuned for dark surfaces — the light-mode 600/700 shades go
   muddy on a dark panel, so text and marks use the 300/400
   band and only gradients reach for the deep end.
   ========================================================= */

export const DARK_TONE = {
  approve: {
    from: "#6EE7B7",
    to: "#059669",
    text: "#34D399",
    soft: "rgba(16,185,129,0.12)",
    line: "rgba(52,211,153,0.30)",
  },

  deny: {
    from: "#FDA4AF",
    to: "#E11D48",
    text: "#FB7185",
    soft: "rgba(244,63,94,0.12)",
    line: "rgba(251,113,133,0.30)",
  },

  warn: {
    from: "#FCD34D",
    to: "#D97706",
    text: "#FBBF24",
    soft: "rgba(245,158,11,0.12)",
    line: "rgba(251,191,36,0.30)",
  },

  info: {
    from: "#67E8F9",
    to: "#0891B2",
    text: "#22D3EE",
    soft: "rgba(8,145,178,0.12)",
    line: "rgba(34,211,238,0.30)",
  },

  neutral: {
    from: "#CBD5E1",
    to: "#64748B",
    text: "#94A3B8",
    soft: "rgba(148,163,184,0.10)",
    line: "rgba(148,163,184,0.24)",
  },
};

/* A tone entry built from a portal accent, so `tone="accent"`
   means violet on the hospital side and blue on the payer
   side without either page hard-coding a hex. */
export const accentTone = (t) => ({
  from: t.accentBright,
  to: t.accentDeep,
  text: t.accent,
  soft: t.accentSoft,
  line: t.accentLine,
});

/* =========================================================
   RECHARTS
   ========================================================= */

export const darkAxis = (t) => ({
  fontSize: 11,
  fill: t.text3,
  fontFamily: "JetBrains Mono",
});

export const darkGrid = "rgba(148,163,184,0.10)";

export const darkTooltip = (t) => ({
  contentStyle: {
    fontSize: 12,
    borderRadius: 10,
    backgroundColor:
      t.key === "payer" ? "#0C1728" : "#160F28",
    border: `1px solid ${t.lineStrong}`,
    boxShadow: "0 18px 40px -18px rgba(0,0,0,0.85)",
    fontFamily: "Inter",
    padding: "8px 12px",
    color: t.text,
  },
  labelStyle: {
    color: t.text2,
    marginBottom: 2,
  },
  itemStyle: {
    color: t.text,
  },
});
