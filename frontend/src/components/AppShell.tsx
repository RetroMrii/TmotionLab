import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router";
import { InterfaceIcon } from "./InterfaceIcon";

export function LensMark() {
  return (
    <svg
      width="30"
      height="30"
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <ellipse
        cx="16"
        cy="16"
        rx="7"
        ry="13"
        transform="rotate(40 16 16)"
        stroke="currentColor"
      />
      <ellipse
        cx="16"
        cy="16"
        rx="7"
        ry="13"
        transform="rotate(-40 16 16)"
        stroke="currentColor"
      />
      <circle cx="16" cy="16" r="3" stroke="currentColor" />
    </svg>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [theme, setTheme] = useState(
    document.documentElement.dataset.theme || "dark",
  );
  const [open, setOpen] = useState(false);
  const [mobile, setMobile] = useState(false);
  const nav = useRef<HTMLElement>(null);
  const explore = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  const exploreRegion = useRef<HTMLDivElement>(null);
  const expanded = useRef(false);
  const pinned = useRef(false);
  const suppressHover = useRef(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const dismissExplore = useCallback(() => {
    clearTimeout(closeTimer.current);
    pinned.current = false;
    expanded.current = false;
    setOpen(false);
  }, []);
  function showExplore() {
    clearTimeout(closeTimer.current);
    expanded.current = true;
    setOpen(true);
  }
  const first = useRef(true);
  useEffect(() => {
    function pointer(event: PointerEvent) {
      document.documentElement.dataset.input = "pointer";
      if (!exploreRegion.current?.contains(event.target as Node)) dismissExplore();
      if (!nav.current?.contains(event.target as Node)) {
        setMobile(false);
      }
    }
    function keyboard(event: KeyboardEvent) {
      document.documentElement.dataset.input = "keyboard";
      document.getAnimations().forEach((animation) => {
        const target = (animation.effect as KeyframeEffect | null)?.target;
        // Keyboard interactions retain their instant behavior, but do not stop the approved ambient artwork.
        if (!(target instanceof Element && target.closest(".emotion-signal"))) animation.cancel();
      });
      if (event.key === "Escape") {
        if (open) {
          dismissExplore();
          suppressHover.current = true;
          explore.current?.focus();
        } else if (mobile) {
          setMobile(false);
          menu.current?.focus();
        }
      }
    }
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("keydown", keyboard);
    };
  }, [open, mobile, dismissExplore]);
  useEffect(() => {
    const breakpoint = window.matchMedia("(max-width: 560px)");
    const resize = () => {
      const wasInSecondaryNav = document.activeElement?.closest(".secondary-nav");
      dismissExplore();
      setMobile(false);
      if (wasInSecondaryNav) (breakpoint.matches ? menu : explore).current?.focus();
    };
    breakpoint.addEventListener("change", resize);
    return () => {
      breakpoint.removeEventListener("change", resize);
      clearTimeout(closeTimer.current);
    };
  }, [dismissExplore]);
  useEffect(() => {
    document.title = `${location.pathname.startsWith("/research") ? "Research" : location.pathname === "/about" ? "About" : location.pathname === "/compare" ? "Compare" : "Reading Studio"} · TmotionLab`;
    if (first.current) {
      first.current = false;
      return;
    }
    const id = requestAnimationFrame(() => {
      dismissExplore();
      setMobile(false);
      const target = location.hash
        ? document.getElementById(location.hash.slice(1))
        : document.getElementById("main");
      target?.focus({ preventScroll: true });
      if (location.hash) target?.scrollIntoView({ behavior: "instant" });
      else window.scrollTo({ top: 0, behavior: "instant" });
    });
    return () => cancelAnimationFrame(id);
  }, [location.pathname, location.hash, dismissExplore]);
  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    setTheme(next);
    try {
      localStorage.setItem("tmotionlab-theme", next);
    } catch {
      /* private storage is optional */
    }
  }
  const analyze = location.pathname === "/" || location.pathname === "/predict";
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="site-header">
        <Link to="/" className="brand" aria-label="TmotionLab home">
          <LensMark />
          <span>TmotionLab</span>
        </Link>
        <nav
          ref={nav}
          aria-label="Main navigation"
          onClick={(event) => {
            if ((event.target as Element).closest("a")) {
              dismissExplore();
              setMobile(false);
            }
          }}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node)) {
              dismissExplore();
              setMobile(false);
            }
          }}
        >
          <Link
            className={analyze ? "nav-link active" : "nav-link"}
            aria-current={analyze ? "page" : undefined}
            to="/"
          >
            Analyze
          </Link>
          <div className={`secondary-nav ${mobile ? "mobile-open" : ""}`}>
            <div className="explore-wrap" ref={exploreRegion}
              onPointerEnter={(event) => {
                if (event.pointerType !== "mouse" || !matchMedia("(hover: hover) and (pointer: fine)").matches) return;
                if (!suppressHover.current) showExplore();
              }}
              onPointerLeave={(event) => {
                if (event.pointerType !== "mouse") return;
                suppressHover.current = false;
                clearTimeout(closeTimer.current);
                closeTimer.current = setTimeout(() => {
                  if (!pinned.current && !exploreRegion.current?.contains(document.activeElement)) dismissExplore();
                }, 150);
              }}
              onBlur={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget as Node)) dismissExplore();
              }}>
              <button
                ref={explore}
                className={`nav-link ${location.pathname.startsWith("/research") ? "active" : ""}`}
                aria-expanded={open}
                aria-controls="explore-menu"
                onClick={(event) => {
                  if (expanded.current && (pinned.current || event.detail === 0)) {
                    dismissExplore();
                    suppressHover.current = true;
                  } else {
                    pinned.current = true;
                    suppressHover.current = false;
                    showExplore();
                  }
                }}
              >
                Explore
              </button>
              <div
                id="explore-menu"
                className="explore-menu"
                data-open={open}
                inert={!open}
                aria-hidden={!open}
              >
                {[
                  [
                    "/research",
                    "Research overview",
                    "The study behind the reading.",
                  ],
                  [
                    "/research/experiments",
                    "Experiment explorer",
                    "Twelve runs. Inspect the evidence.",
                  ],
                ].map(([to, title, description], i) => (
                  <Link
                    key={to}
                    to={to}
                    aria-current={location.pathname === to ? "page" : undefined}
                  >
                    <span className="menu-number">0{i + 1}</span>
                    <span>
                      <strong>{title}</strong>
                      <small>{description}</small>
                    </span>
                    <InterfaceIcon name="arrow" />
                  </Link>
                ))}
              </div>
            </div>
            <Link
              className={`nav-link ${location.pathname === "/about" ? "active" : ""}`}
              aria-current={location.pathname === "/about" ? "page" : undefined}
              to="/about"
            >
              About
            </Link>
          </div>
          <button
            className="theme-toggle icon-button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            <svg
              viewBox="0 0 24 24"
              width="19"
              height="19"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
            </svg>
          </button>
          <button
            className="mobile-toggle icon-button"
            ref={menu}
            aria-label="Toggle navigation"
            aria-expanded={mobile}
            onClick={() => {
              setMobile(!mobile);
              dismissExplore();
            }}
          >
            <InterfaceIcon name={mobile ? "close" : "menu"} />
          </button>
        </nav>
      </header>
      <main id="main" tabIndex={-1}>
        {children}
      </main>
      <footer className="site-footer">
        <Link to="/" className="brand">
          <LensMark />
          <span>TmotionLab</span>
        </Link>
        <p>A model perspective. Always human context.</p>
        <Link to="/about">
          About the project
        </Link>
      </footer>
    </>
  );
}
