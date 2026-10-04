import {
  AlertTriangle,
  CircleDot,
  Database,
  Flag,
  ShieldCheck,
  ShieldQuestion,
  ShieldX,
  Sparkles,
  Wrench,
} from "lucide-react";

import type { StepKind } from "../types";

const VERDICT_HELP: Record<string, string> = {
  VERIFIED: "The fix passed reliably more often than the unchanged control (95% intervals do not overlap).",
  REFUTED: "A known-good edit reached the outcome but did not improve it.",
  INCONCLUSIVE: "The fix and control intervals overlap. More samples or a better edit are needed.",
};

export function VerdictBadge({
  verdict,
  preview,
}: {
  verdict: string | null | undefined;
  preview?: boolean;
}) {
  if (preview) {
    return (
      <span className="badge badge-outline" title="Run without a paired control, so no verdict.">
        Preview only
      </span>
    );
  }
  if (verdict === "VERIFIED") {
    return (
      <span className="badge badge-pass" title={VERDICT_HELP.VERIFIED}>
        <ShieldCheck size={12} aria-hidden /> Verified
      </span>
    );
  }
  if (verdict === "REFUTED") {
    return (
      <span className="badge badge-fail" title={VERDICT_HELP.REFUTED}>
        <ShieldX size={12} aria-hidden /> Refuted
      </span>
    );
  }
  if (verdict === "INCONCLUSIVE") {
    return (
      <span className="badge badge-warn" title={VERDICT_HELP.INCONCLUSIVE}>
        <ShieldQuestion size={12} aria-hidden /> Inconclusive
      </span>
    );
  }
  return <span className="badge badge-neutral">Not verified</span>;
}

const CACHE_HELP: Record<string, string> = {
  cached: "Served from the recording. Its request was unchanged, so no call was made.",
  invalidated: "Downstream of the edit, so its recorded answer can no longer be trusted.",
  live: "Re-executed for real because its inputs changed.",
  edited: "The step you changed.",
};

export function CacheBadge({ status }: { status: string | null | undefined }) {
  if (!status) return null;
  const cls =
    status === "cached"
      ? "badge-neutral"
      : status === "edited"
        ? "badge-accent"
        : status === "live" || status === "invalidated"
          ? "badge-warn"
          : "badge-outline";
  return (
    <span className={`badge ${cls}`} title={CACHE_HELP[status] ?? status}>
      {status}
    </span>
  );
}

export function SuspectBadge({ probability }: { probability?: number | null }) {
  return (
    <span className="badge badge-accent" title="Model's calibrated probability that this is the root cause">
      <CircleDot size={12} aria-hidden /> Suspect
      {probability != null ? ` · ${Math.round(probability * 100)}%` : ""}
    </span>
  );
}

export function FailureBadge() {
  return (
    <span className="badge badge-fail" title="Where the failure became visible (a symptom, not necessarily the cause)">
      <AlertTriangle size={12} aria-hidden /> Visible failure
    </span>
  );
}

export function KindAvatar({ kind }: { kind: StepKind | string }) {
  const k = normaliseKind(kind);
  const icon =
    k === "llm" ? (
      <Sparkles size={13} aria-hidden />
    ) : k === "tool" ? (
      <Wrench size={13} aria-hidden />
    ) : k === "retrieval" ? (
      <Database size={13} aria-hidden />
    ) : (
      <Flag size={13} aria-hidden />
    );
  return (
    <span className={`avatar k-${k}`} title={kindLabel(k)}>
      {icon}
    </span>
  );
}

export function normaliseKind(kind: string): StepKind {
  if (kind === "llm" || kind === "tool" || kind === "retrieval" || kind === "final") return kind;
  if (kind === "state" || kind === "value") return "state";
  return "tool";
}

export function kindLabel(kind: StepKind): string {
  return (
    {
      llm: "LLM call",
      tool: "Tool call",
      retrieval: "Retrieval",
      state: "State update",
      value: "Recorded value",
      final: "Final answer",
    } as const
  )[kind];
}
