"use client";
import { useEffect, useState } from "react";
import { TopBar } from "./TopBar";
import { api } from "../../lib/api";
import { AppContext } from "./AppContext";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<string | null>(null);
  const [staticBundle, setStaticBundle] = useState(false);
  const [status, setStatus] = useState("Connecting to recorder");
  useEffect(() => { api<{mode:string;status:string;static_bundle:boolean;counts:{runs:number;steps:number}}>('/health')
    .then(h => { setMode(h.mode); setStaticBundle(h.static_bundle); setStatus(h.static_bundle
      ? "Static recorded showcase · writes disabled"
      : `${h.counts.runs.toLocaleString()} runs · ${h.counts.steps.toLocaleString()} recorded steps`); })
    .catch(() => setStatus("API unavailable · start with make dev-api")); }, []);
  return <AppContext.Provider value={{staticBundle,mode}}><div className="shell"><TopBar mode={mode}/><main className="main">{children}</main><footer className="statusbar"><span><i className={`dot ${mode ? "online" : "offline"}`} />{status}</span><span className="spacer"/><span>Black Box · local recorder</span></footer></div></AppContext.Provider>;
}
