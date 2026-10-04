"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Runs", match: (p: string) => p === "/" || p.startsWith("/investigate") },
  { href: "/new", label: "New task", match: (p: string) => p.startsWith("/new") },
  { href: "/compare", label: "Compare", match: (p: string) => p.startsWith("/compare") },
  { href: "/results", label: "Results", match: (p: string) => p.startsWith("/results") },
  { href: "/label", label: "Label", match: (p: string) => p.startsWith("/label") },
];

export function TopBar({ mode }: { mode: string | null }) {
  const pathname = usePathname() ?? "/";
  return (
    <header className="topbar">
      <Link href="/" className="brand" aria-label="blackbox home">
        <span className="brand-mark" aria-hidden />
        blackbox
      </Link>
      <nav className="nav" aria-label="Primary">
        {NAV.map((item) => (
          <Link key={item.href} href={item.href} aria-current={item.match(pathname) ? "page" : undefined}>
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="topbar-right">
        <ModeBadge mode={mode} />
      </div>
    </header>
  );
}

function ModeBadge({ mode }: { mode: string | null }) {
  if (!mode) return <span className="badge badge-neutral">Connecting…</span>;
  const live = mode === "live";
  const title =
    mode === "live"
      ? "Replays may call the configured LLM provider."
      : mode === "offline"
        ? "Replays run against a local model or deterministic stand-ins."
        : "Only recorded responses are served. Nothing calls an LLM.";
  return (
    <span className={`badge ${live ? "badge-warn" : "badge-outline"}`} title={title}>
      <span className="dot" style={{ background: live ? "var(--warn)" : "var(--text-3)", margin: 0 }} />
      {mode.toUpperCase()}
    </span>
  );
}
