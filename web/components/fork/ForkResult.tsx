"use client";

import Link from "next/link";
import { useState } from "react";
import { api, errorText } from "../../lib/api";
import type { ExportTestResponse, ForkSummary } from "../../lib/contract";
import { VerdictBadge } from "../ui/badges";
import { PassRateText } from "./PassRateText";

export interface ForkProgress {
  k: number;
  control: boolean;
  fixDone: number;
  fixPassed: number;
  controlDone: number;
  controlPassed: number;
  current: string;
}

interface ForkResultProps {
  baseRunId: string;
  progress: ForkProgress;
  summary: ForkSummary | null;
  error: string;
  onClose: () => void;
}

/** Live tally while a paired replay runs, then its verdict, savings and export. */
export function ForkResult({ baseRunId, progress, summary, error, onClose }: ForkResultProps) {
  const [exported, setExported] = useState<ExportTestResponse | null>(null);
  const [exportError, setExportError] = useState("");
  const [exporting, setExporting] = useState(false);
  const complete = summary?.status === "complete";
  const failed = summary?.status === "error" || Boolean(error);
  const fixedRun = summary?.edited_runs.find((run) => run.passed) ?? summary?.edited_runs[0];

  async function exportTest() {
    if (!summary) return;
    setExporting(true);
    setExportError("");
    try {
      setExported(await api<ExportTestResponse>(`/forks/${encodeURIComponent(summary.fork_id)}/export-test`, {
        method: "POST",
        body: JSON.stringify({ overwrite: true }),
      }));
    } catch (caught) {
      setExportError(errorText(caught));
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="floating-card fork-result" role="status" aria-live="polite">
      <div className="spread">
        <strong>{complete ? "Fix comparison" : failed ? "Replay stopped" : "Testing the change…"}</strong>
        <div className="row">
          {complete && <VerdictBadge verdict={summary.verdict} preview={summary.preview} />}
          <button type="button" className="btn btn-sm btn-ghost" onClick={onClose} aria-label="Dismiss">✕</button>
        </div>
      </div>

      {!complete && !failed && (
        <div className="fork-tally">
          <div><span className="label">With change</span><strong>{progress.fixPassed}/{progress.fixDone}</strong><span className="faint"> of {progress.k} done passed</span></div>
          {progress.control && <div><span className="label">Unchanged</span><strong>{progress.controlPassed}/{progress.controlDone}</strong><span className="faint"> of {progress.k} done passed</span></div>}
          <p className="faint mono">{progress.current}</p>
        </div>
      )}

      {failed && <p className="error-box">{error || summary?.error?.message || "The replay failed."}{summary?.error?.hint ? ` ${summary.error.hint}` : ""}</p>}

      {complete && summary && (
        <>
          <div className="fork-tally">
            {summary.fix && <div><span className="label">With change</span><PassRateText rate={summary.fix} /></div>}
            {summary.control_result && <div><span className="label">Unchanged</span><PassRateText rate={summary.control_result} /></div>}
          </div>
          {summary.savings && (
            <p className="faint">
              Re-ran {summary.savings.reexecuted} of {summary.savings.total_steps} steps per sample and reused {summary.savings.calls_cached} recorded calls
              {summary.replay_fidelity != null ? ` · replay fidelity ${Math.round(summary.replay_fidelity * 100)}%` : ""}.
            </p>
          )}
          <div className="row fork-actions">
            {fixedRun && (
              <Link className="btn btn-sm" href={`/compare?a=${encodeURIComponent(baseRunId)}&b=${encodeURIComponent(fixedRun.run_id)}`}>See what changed</Link>
            )}
            <button type="button" className="btn btn-sm btn-primary" onClick={exportTest} disabled={!summary.export_available || exporting}
              title={summary.export_available ? "Write an offline pytest check for this verified fix" : "Only a VERIFIED fix can be exported"}>
              {exporting ? "Exporting…" : "Export regression test"}
            </button>
          </div>
          {exportError && <p className="error-box">{exportError}</p>}
          {exported && (
            <div className="export-box">
              <span className="label">Saved offline test</span>
              <code className="mono">{exported.run_command}</code>
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => navigator.clipboard?.writeText(exported.run_command)}>Copy</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
