/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],

  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif"
        ],
        display: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "sans-serif"
        ],
        mono: [
          "JetBrains Mono",
          "SFMono-Regular",
          "Consolas",
          "monospace"
        ]
      },

      fontSize: {
        "2xs": [
          "10px",
          {
            lineHeight: "14px"
          }
        ]
      },

      /* ===================================================
         DARK PALETTE

         The whole product runs dark. Nothing here is white:
         `surface` is the panel a card sits on and `canvas` is
         the deeper ground behind it, so a box always reads as
         raised rather than as a sheet of paper.

         Portal accents:
           provider (hospital)  -> violet
           payer    (insurance) -> blue

         Every `-soft` / `-line` value is a solid hex rather
         than an rgba() string so Tailwind's `/opacity`
         modifiers (bg-provider-soft/40, bg-approve/10) still
         compile.
         =================================================== */
      colors: {
        canvas: "#0C111C",
        surface: "#141B2A",
        ink: "#E8EDF7",
        "ink-2": "#A7B4CC",
        "ink-3": "#7A87A1",

        rule: "#232C40",
        ruleStrong: "#33405C",

        provider: "#8B5CF6",
        "provider-deep": "#A78BFA",
        "provider-soft": "#221A38",
        "provider-line": "#4C3A7A",

        payer: "#3B82F6",
        "payer-deep": "#60A5FA",
        "payer-soft": "#12203A",
        "payer-line": "#2E4A7A",

        approve: "#34D399",
        "approve-soft": "#0F2A22",
        "approve-line": "#1E5C48",

        deny: "#FB7185",
        "deny-soft": "#2C1520",
        "deny-line": "#6B2438",

        review: "#FBBF24",
        "review-soft": "#2B2110",
        "review-line": "#6B4E17",

        info: "#38BDF8",
        "info-soft": "#0E2333",
        "info-line": "#1C4E68"
      },

      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,.35), 0 14px 34px -18px rgba(0,0,0,.75)",
        soft: "0 14px 44px -18px rgba(0,0,0,.7)",
        elevated: "0 26px 64px -22px rgba(0,0,0,.85)"
      },

      borderRadius: {
        xl: "14px",
        "2xl": "18px"
      },

      animation: {
        "fade-in": "fadeIn .25s ease-out",
        "slide-up": "slideUp .3s ease-out",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite"
      },

      keyframes: {
        fadeIn: {
          "0%": {
            opacity: "0"
          },
          "100%": {
            opacity: "1"
          }
        },

        slideUp: {
          "0%": {
            opacity: "0",
            transform: "translateY(8px)"
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0)"
          }
        },

        pulseSoft: {
          "0%, 100%": {
            opacity: ".55"
          },
          "50%": {
            opacity: "1"
          }
        }
      }
    }
  },

  plugins: []
}