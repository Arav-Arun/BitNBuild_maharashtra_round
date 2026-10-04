"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Box } from "lucide-react";

const NAV = [
  { href: "/", label: "Runs", match: (p: string) => p === "/" || p.startsWith("/investigate") },
  { href: "/new", label: "New task", match: (p: string) => p.startsWith("/new") },
  { href: "/compare", label: "Compare", match: (p: string) => p.startsWith("/compare") },
  { href: "/results", label: "Results", match: (p: string) => p.startsWith("/results") },
  { href: "/label", label: "Label", match: (p: string) => p.startsWith("/label") },
];

export function TopBar() {
  const pathname = usePathname() ?? "/";
  return (
    <header className="topbar">
      <Link href="/" className="brand" aria-label="blackbox home">
        <Box className="brand-mark" aria-hidden="true" strokeWidth={1.8} />
        blackbox
      </Link>
      <nav className="nav" aria-label="Primary">
        {NAV.map((item) => (
          <Link key={item.href} href={item.href} aria-current={item.match(pathname) ? "page" : undefined}>
            {item.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
