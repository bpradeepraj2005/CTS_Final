import { useEffect, useState } from "react";
import Chatbot from "./Chatbot";

import {
  Bell,
  ChevronDown,
  CircleDot,
  LogOut,
  Menu,
  PauseCircle,
  ShieldCheck,
  X,
  ListChecks,
  AlertTriangle,
} from "lucide-react";

import {
  NavLink,
  Outlet,
  useNavigate,
} from "react-router-dom";

import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { themeFor } from "../lib/portalTheme";


/* =========================================================
   PORTAL SHELL

   The chrome — header, sidebar, chat widget — is dark on both
   portals; the accent hue is what tells them apart (violet for
   the hospital side, blue for the payer side). Colours come
   from src/lib/portalTheme.js so the dashboards inside can
   paint against exactly the same palette.

   The full-bleed page background is painted here, once, on the
   shell root — every page inside <Outlet /> draws its panels on
   top of it rather than carrying a background of its own. That
   keeps the gutter around the content column the same colour as
   the content area, which a page background applied from inside
   <main>'s padding cannot do.
   ========================================================= */

export default function Layout({ portal, nav }) {
  const { user, logout, refresh } = useAuth();
  const navigate = useNavigate();

  const [busy, setBusy] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [signOutHot, setSignOutHot] = useState(false);

  const isPayer = portal === "payer";

  const t = themeFor(portal);


  /* =====================================================
     REFRESH USER
     ===================================================== */

  useEffect(() => {
    refresh().catch(() => {});
  }, []);


  /* =====================================================
     REVIEWER AVAILABILITY
     ===================================================== */

  const toggleAvailability = async () => {
    if (!user) return;

    setBusy(true);

    try {
      await api.patch(
        "/api/auth/availability",
        {
          is_available: !user.is_available,

          unavailable_reason: user.is_available
            ? "On another case"
            : null,
        }
      );

      await refresh();

    } finally {
      setBusy(false);
    }
  };


  /* =====================================================
     LOGOUT
     ===================================================== */

  const handleLogout = () => {
    logout();
    navigate("/");
  };


  /* Shared look for the small square chrome buttons. */
  const chromeButton = {
    background: t.panel,
    borderColor: t.line,
    color: t.text2,
  };

  const dangerHover = {
    background: "rgba(244,63,94,0.14)",
    borderColor: "rgba(251,113,133,0.35)",
    color: "#FB7185",
  };


  return (
    <div
      className="min-h-screen"
      style={{
        background: t.page,
        backgroundSize: "cover",
        backgroundAttachment: "fixed",
      }}
    >

      {/* =================================================
          HEADER
         ================================================= */}

      <header
        className="sticky top-0 z-40 border-b backdrop-blur-xl"
        style={{
          background: t.headerBg,
          borderColor: t.line,
        }}
      >

        <div className="flex h-16 items-center justify-between px-4 lg:px-6">

          {/* LEFT SIDE */}
          <div className="flex items-center gap-3">

            {/* MOBILE MENU */}
            <button
              className="grid h-10 w-10 place-items-center rounded-lg border transition-colors lg:hidden"
              style={chromeButton}
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={18} />
            </button>


            {/* LOGO */}
            <div className="flex items-center gap-3">

              <div
                className="grid h-10 w-10 place-items-center rounded-xl text-white"
                style={{
                  backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
                  boxShadow: `0 10px 26px -10px ${t.accentGlow}`,
                }}
              >
                <ShieldCheck size={20} />
              </div>


              <div className="hidden sm:block">

                <div
                  className="text-sm font-bold tracking-tight"
                  style={{ color: t.text }}
                >
                  PriorAuth AI
                </div>

                <div
                  className="text-[10px] font-medium uppercase tracking-[.12em]"
                  style={{ color: t.text3 }}
                >

                  {isPayer
                    ? "Payer intelligence"
                    : "Hospital management"}

                </div>

              </div>

            </div>

          </div>


          {/* =================================================
              HEADER RIGHT
             ================================================= */}

          <div className="flex items-center gap-2">


            {/* REVIEWER AVAILABILITY */}

            {isPayer && user && (

              <button
                onClick={toggleAvailability}
                disabled={busy}
                className="hidden items-center gap-2 rounded-lg border px-3 py-2 text-xs font-semibold transition disabled:opacity-60 sm:flex"
                style={
                  user.is_available
                    ? {
                        background: "rgba(16,185,129,0.12)",
                        borderColor: "rgba(52,211,153,0.30)",
                        color: "#34D399",
                      }
                    : {
                        background: t.panel,
                        borderColor: t.line,
                        color: t.text3,
                      }
                }
              >

                {user.is_available ? (
                  <CircleDot size={14} />
                ) : (
                  <PauseCircle size={14} />
                )}

                {user.is_available
                  ? "Available"
                  : "Unavailable"}

              </button>

            )}


            {/* NOTIFICATIONS */}

            <button
              className="grid h-10 w-10 place-items-center rounded-lg border transition-colors"
              style={chromeButton}
              title="Notifications"
            >
              <Bell size={17} />
            </button>


            {/* USER */}

            {user && (

              <div className="hidden items-center gap-2 pl-2 sm:flex">

                <div
                  className="grid h-9 w-9 place-items-center rounded-full text-xs font-bold text-white"
                  style={{
                    backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
                  }}
                >
                  {user.full_name
                    ?.charAt(0)
                    ?.toUpperCase() || "U"}
                </div>


                <div className="max-w-[150px]">

                  <div
                    className="truncate text-xs font-semibold"
                    style={{ color: t.text }}
                  >
                    {user.full_name}
                  </div>

                  <div
                    className="truncate text-[10px]"
                    style={{ color: t.text3 }}
                  >
                    {user.organization_name}
                  </div>

                </div>


                <ChevronDown
                  size={14}
                  style={{ color: t.text3 }}
                />

              </div>

            )}


            {/* LOGOUT */}

            <button
              onClick={handleLogout}
              onMouseEnter={() => setSignOutHot(true)}
              onMouseLeave={() => setSignOutHot(false)}
              className="grid h-10 w-10 place-items-center rounded-lg border transition-colors"
              style={
                signOutHot
                  ? dangerHover
                  : chromeButton
              }
              title="Sign out"
            >
              <LogOut size={16} />
            </button>

          </div>

        </div>

      </header>


      {/* =====================================================
          MAIN BODY
         ===================================================== */}

      <div className="flex">


        {/* =================================================
            DESKTOP SIDEBAR
           ================================================= */}

        <aside
          className="sticky top-16 hidden h-[calc(100vh-64px)] w-64 shrink-0 border-r lg:block"
          style={{
            background: t.sidebar,
            backgroundAttachment: "fixed",
            backgroundSize: "cover",
            borderColor: t.line,
          }}
        >

          <Sidebar
            nav={nav}
            portal={portal}
            user={user}
          />

        </aside>


        {/* =================================================
            MOBILE SIDEBAR
           ================================================= */}

        {mobileOpen && (

          <div className="fixed inset-0 z-50 lg:hidden">

            {/* OVERLAY */}

            <div
              className="absolute inset-0 backdrop-blur-sm"
              style={{ background: "rgba(2,4,10,0.66)" }}
              onClick={() => setMobileOpen(false)}
            />


            {/* SIDEBAR */}

            <aside
              className="relative h-full w-80 shadow-elevated"
              style={{
                background: t.sidebar,
                backgroundSize: "cover",
              }}
            >

              <div
                className="flex h-16 items-center justify-between border-b px-5"
                style={{ borderColor: t.line }}
              >

                <div
                  className="font-bold"
                  style={{ color: t.text }}
                >
                  Navigation
                </div>


                <button
                  onClick={() => setMobileOpen(false)}
                  className="grid h-9 w-9 place-items-center rounded-lg border"
                  style={chromeButton}
                  aria-label="Close navigation"
                >
                  <X size={17} />
                </button>

              </div>


              <Sidebar
                nav={nav}
                portal={portal}
                user={user}
                onNavigate={() => setMobileOpen(false)}
              />

            </aside>

          </div>

        )}


        {/* =================================================
            PAGE CONTENT
           ================================================= */}

        <main className="min-w-0 flex-1">

          <div className="mx-auto max-w-[1500px] px-4 py-6 lg:px-8 lg:py-8">

            <div className="page-enter">

              <Outlet />

            </div>


            {/* CHATBOT */}

            <Chatbot portal={portal} />

          </div>

        </main>

      </div>

    </div>
  );
}


/* =========================================================
   SIDEBAR
   ========================================================= */

function Sidebar({
  nav,
  portal,
  user,
  onNavigate,
}) {

  const isPayer = portal === "payer";

  const t = themeFor(portal);

  /* NavLink's className callback can't reach a hover state, so
     the non-active hover tint rides on inline handlers instead
     of a Tailwind `hover:` class — the panel colours are
     per-portal values, not utility classes. */
  const [hovered, setHovered] = useState(null);


  return (

    <div className="flex h-full flex-col overflow-y-auto px-3 py-5">


      {/* =================================================
          WORKSPACE
         ================================================= */}

      <div
        className="mb-5 rounded-xl border p-4 backdrop-blur-sm"
        style={{
          background: t.panel,
          borderColor: t.line,
        }}
      >

        <div
          className="text-[10px] font-semibold uppercase tracking-[.13em]"
          style={{ color: t.text3 }}
        >
          Workspace
        </div>


        <div
          className="mt-2 text-sm font-bold"
          style={{ color: t.text }}
        >

          {isPayer
            ? "Payer operations"
            : "Hospital operations"}

        </div>


        <div
          className="mt-1 text-xs"
          style={{ color: t.text3 }}
        >

          {user?.organization_name ||
            "Organization"}

        </div>

      </div>


      {/* =================================================
          NAVIGATION
         ================================================= */}

      <nav className="space-y-1">

        {nav.map((item) => {

          const Icon = item.icon;

          /*
           * Detect the Human Review Queue.
           *
           * This allows us to give the queue a special
           * visual treatment without changing routing.
           */

          const isReviewQueue =
            isPayer &&
            (
              item.to === "/payer/queue" ||
              item.label?.toLowerCase().includes("review queue")
            );


          return (

            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={onNavigate}
              onMouseEnter={() => setHovered(item.to)}
              onMouseLeave={() => setHovered(null)}

              className="group relative flex items-center gap-3 rounded-lg border px-3 py-2.5 text-[13px] font-semibold transition"

              style={({ isActive }) =>
                isActive
                  ? {
                      background: t.accentSoft,
                      borderColor: t.accentLine,
                      color: t.accentBright,
                    }
                  : {
                      background:
                        hovered === item.to
                          ? t.panelHover
                          : "transparent",
                      borderColor: "transparent",
                      color:
                        hovered === item.to
                          ? t.text
                          : t.text2,
                    }
              }
            >

              {/* ICON */}

              <Icon
                size={17}
                strokeWidth={1.9}
              />


              {/* LABEL */}

              <span className="flex-1">

                {isReviewQueue
                  ? "Human Review Queue"
                  : item.label}

              </span>


              {/* =================================================
                  HUMAN REVIEW INDICATOR
                 ================================================= */}

              {isReviewQueue && (

                <span
                  className="flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide"
                  style={{
                    background: "rgba(245,158,11,0.14)",
                    borderColor: "rgba(251,191,36,0.32)",
                    color: "#FBBF24",
                  }}
                  title="Cases requiring human review"
                >
                  <AlertTriangle size={10} />

                  Review

                </span>

              )}

            </NavLink>

          );

        })}

      </nav>


      {/* =================================================
          HUMAN REVIEW INFORMATION
         ================================================= */}

      {isPayer && (

        <div
          className="mt-5 rounded-xl border p-4"
          style={{
            background: "rgba(245,158,11,0.09)",
            borderColor: "rgba(251,191,36,0.26)",
          }}
        >

          <div className="flex items-center gap-2">

            <div
              className="grid h-7 w-7 place-items-center rounded-lg"
              style={{
                background: "rgba(245,158,11,0.18)",
                color: "#FCD34D",
              }}
            >

              <ListChecks size={15} />

            </div>


            <div>

              <div
                className="text-xs font-bold"
                style={{ color: "#FCD34D" }}
              >
                Human Review
              </div>

              <div
                className="text-[10px]"
                style={{ color: "rgba(252,211,77,0.72)" }}
              >
                Priority-based cases
              </div>

            </div>

          </div>


          <p
            className="mt-3 text-[10px] leading-4"
            style={{ color: "rgba(253,230,138,0.68)" }}
          >

            AI-flagged requests requiring human
            verification are organized in the
            review queue based on clinical severity.

          </p>

        </div>

      )}


      {/* =================================================
          HELP
         ================================================= */}

      <div
        className="mt-auto rounded-xl border p-4 backdrop-blur-sm"
        style={{
          background: t.panel,
          borderColor: t.line,
        }}
      >

        <div className="flex items-center gap-2">

          <div
            className="h-2 w-2 rounded-full"
            style={{
              background: t.accent,
              boxShadow: `0 0 10px ${t.accent}`,
            }}
          />

          <span
            className="text-xs font-semibold"
            style={{ color: t.text }}
          >
            Need help?
          </span>

        </div>


        <p
          className="mt-2 text-[10px] leading-4"
          style={{ color: t.text3 }}
        >

          Upload your documents and we'll help
          you complete the request.

        </p>

      </div>

    </div>

  );
}
