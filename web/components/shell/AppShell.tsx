"use client";
import { useEffect, useState } from "react";
import { TopBar } from "./TopBar";
import { UNREACHABLE_HINT, api } from "../../lib/api";
import type { AppHealth } from "../../lib/contract";
import { AppContext, type AppState } from "./AppContext";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AppState>({
    staticBundle: false,
    mode: null,
    reachable: true,
    status: "Connecting to recorder",
  });
  useEffect(() => {
    api<AppHealth>("/health")
      .then((h) => setState({
        mode: h.mode,
        staticBundle: h.static_bundle,
        reachable: true,
        status: h.static_bundle
          ? "Static recorded showcase · writes disabled"
          : `${h.counts.runs.toLocaleString()} recorded runs, ${h.counts.steps.toLocaleString()} steps`,
      }))
      .catch(() => setState({
        mode: null,
        staticBundle: false,
        reachable: false,
        status: `API unavailable. ${UNREACHABLE_HINT}`,
      }));
  }, []);
  return <AppContext.Provider value={state}><div className="shell"><TopBar/><main className="main">{children}</main></div></AppContext.Provider>;
}
