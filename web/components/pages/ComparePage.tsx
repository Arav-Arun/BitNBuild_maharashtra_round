"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { DiffResponse, DiffRow, RunList } from "../../lib/contract";
import { api } from "../../lib/api";

const STAT_LABELS: Record<string, string> = {
  same: "Unchanged steps",
  cached: "Recorded steps reused",
  changed: "Steps with changes",
  new: "New steps",
  removed: "Steps no longer run",
};

function outcomeLabel(status: string) {
  return status === "passed" ? "Completed" : status === "failed" ? "Failed" : "Still running";
}

function changedLabel(fields: string[]) {
  const labels: Record<string, string> = { input: "Inputs", output: "Result", state: "Agent state" };
  return fields.map((field) => labels[field] || field).join(", ");
}

function stepName(row: DiffRow) {
  const side = row.right || row.left;
  return side ? `Step ${side.seq + 1}: ${side.name}` : row.addr;
}

function outputPreview(value: unknown) {
  if (value === null || value === undefined) return "No result recorded";
  if (typeof value !== "object") return String(value).slice(0, 180);
  if (Array.isArray(value)) return `${value.length} items`;
  return Object.entries(value as Record<string, unknown>)
    .slice(0, 4)
    .map(([key, item]) => `${key}: ${typeof item === "object" && item !== null ? "…" : String(item)}`)
    .join(" · ");
}

export function ComparePage() {
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [runs, setRuns] = useState<RunList | null>(null);
  const [diff, setDiff] = useState<DiffResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setLeftId(params.get("a") || "");
    setRightId(params.get("b") || "");
    api<RunList>("/runs?limit=100").then(setRuns).catch((caught) => setError(caught.message));
  }, []);

  useEffect(() => {
    if (!leftId || !rightId) {
      setDiff(null);
      return;
    }
    setLoading(true);
    setError("");
    api<DiffResponse>(`/diff?a=${encodeURIComponent(leftId)}&b=${encodeURIComponent(rightId)}`)
      .then(setDiff)
      .catch((caught) => setError(caught.message))
      .finally(() => setLoading(false));
  }, [leftId, rightId]);

  const firstChanged = useMemo(
    () => diff?.rows.find((row) => row.addr === diff.first_divergence),
    [diff],
  );
  const selectedRun = (runId: string) => runs?.items.find((run) => run.run_id === runId);

  return (
    <div className="page">
      <div className="page-inner">
        <p className="label">Compare task runs</p>
        <h1 className="h1">See what changed</h1>
        <p className="muted">Choose two runs to compare their results and the steps the agent took to get there.</p>

        <div className="compare-select">
          <label className="compare-choice">
            <span className="label">Original run</span>
            <select className="input" value={leftId} onChange={(event) => setLeftId(event.target.value)}>
              <option value="">Choose a task run</option>
              {runs?.items.map((run) => <option key={run.run_id} value={run.run_id}>{run.task} · {outcomeLabel(run.status)}</option>)}
            </select>
          </label>
          <span className="compare-arrow" aria-hidden="true">→</span>
          <label className="compare-choice">
            <span className="label">Changed or follow-up run</span>
            <select className="input" value={rightId} onChange={(event) => setRightId(event.target.value)}>
              <option value="">Choose a task run</option>
              {runs?.items.map((run) => <option key={run.run_id} value={run.run_id}>{run.task} · {outcomeLabel(run.status)}</option>)}
            </select>
          </label>
        </div>

        {error && <p className="error-box" role="alert">{error}</p>}
        {loading && <p className="muted">Comparing the two recorded workflows…</p>}

        {diff && (
          <>
            <div className="compare-run-context">
              <div><span className="label">Original</span><strong>{selectedRun(leftId)?.task || leftId}</strong></div>
              <span aria-hidden="true">→</span>
              <div><span className="label">Changed run</span><strong>{selectedRun(rightId)?.task || rightId}</strong></div>
            </div>

            <div className="grid-cards compare-metrics">
              <div className="metric">
                <div className="label">First step where results differ</div>
                <div className="value compare-first-step">{firstChanged ? stepName(firstChanged) : "No differences"}</div>
              </div>
              {Object.entries(diff.stats).map(([key, count]) => (
                <div className="metric" key={key}><div className="label">{STAT_LABELS[key] || key}</div><div className="value">{count}</div></div>
              ))}
            </div>

            <div className="spread card card-pad compare-outcome">
              <div>
                <span className="label">Final result</span>
                <div><strong>{outcomeLabel(diff.outcome.left.status)}</strong> → <strong>{outcomeLabel(diff.outcome.right.status)}</strong></div>
                <p className="muted">
                  {diff.outcome.direction === "fail_to_pass"
                    ? "The changed run completed successfully after the original failed."
                    : diff.outcome.direction === "pass_to_fail"
                      ? "The changed run failed even though the original completed."
                      : "Both runs reached the same final result."}
                </p>
              </div>
              <span className={`badge ${diff.outcome.flipped ? "badge-accent" : "badge-neutral"}`}>
                {diff.outcome.flipped ? "Result changed" : "Same result"}
              </span>
              <Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(rightId)}`}>Inspect changed run →</Link>
            </div>

            <div className="compare-table-intro">
              <h2 className="h2">Step-by-step changes</h2>
              <p className="muted">Black Box reuses recorded results for steps unaffected by the change and reruns the steps that depend on it.</p>
            </div>
            <div className="card table-wrap">
              <table className="table">
                <thead><tr><th>Workflow step</th><th>What changed</th><th>Was it rerun?</th><th>Original result</th><th>Changed result</th></tr></thead>
                <tbody>
                  {diff.rows.map((row) => (
                    <tr key={row.addr}>
                      <td><strong>{stepName(row)}</strong><div className="mono faint">{row.addr}</div></td>
                      <td><span className={`badge ${row.status === "changed" ? "badge-warn" : "badge-neutral"}`}>{row.status === "changed" ? "Changed" : row.status === "same" ? "Same" : row.status === "cached" ? "Reused" : row.status === "new" ? "Added" : "Removed"}</span>{row.changed.length > 0 && <div className="faint">{changedLabel(row.changed)}</div>}</td>
                      <td>{row.in_cone ? "Yes · depends on the edit" : row.status === "cached" || row.status === "same" ? "No · recorded result reused" : "No"}</td>
                      <td><details className="diff-output"><summary>{outputPreview(row.left?.output)}</summary><pre className="diff-json">{JSON.stringify(row.left?.output, null, 2) || "—"}</pre></details></td>
                      <td><details className="diff-output"><summary>{outputPreview(row.right?.output)}</summary><pre className="diff-json">{JSON.stringify(row.right?.output, null, 2) || "—"}</pre></details></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
