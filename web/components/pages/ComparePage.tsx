"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import type { DiffResponse, DiffRow, RunDetail, RunList, RunSummary } from "../../lib/contract";
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

function replayLabel(row: DiffRow) {
  if (!row.right) return "Step removed";
  switch (row.right.cache_status) {
    case "cached": return "Recorded result reused";
    case "invalidated": return "Replayed after an input changed";
    case "edited": return "Edited at this step";
    case "live": return "Executed for this run";
  }
}

export function ComparePage() {
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [runQuery, setRunQuery] = useState("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runContext, setRunContext] = useState<Record<string, RunSummary>>({});
  const [diff, setDiff] = useState<DiffResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setLeftId(params.get("a") || "");
    setRightId(params.get("b") || "");
    Promise.all([
      api<RunList>("/runs?limit=100"),
      api<RunList>("/runs?origin=fork&limit=100"),
    ]).then(([recorded, replays]) => {
      const unique = new Map<string, RunSummary>();
      [...recorded.items, ...replays.items].forEach((run) => unique.set(run.run_id, run));
      setRuns([...unique.values()]);
    }).catch((caught) => setError(caught.message));
  }, []);

  useEffect(() => {
    if (!leftId || !rightId) {
      setRunContext({});
      return;
    }
    let active = true;
    Promise.all([leftId, rightId].map((id) => api<RunDetail>(`/runs/${encodeURIComponent(id)}`)))
      .then((details) => {
        if (active) setRunContext(Object.fromEntries(details.map((detail) => [detail.run.run_id, detail.run])));
      })
      .catch(() => {
        if (active) setRunContext({});
      });
    return () => { active = false; };
  }, [leftId, rightId]);

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
  const selectedRun = (runId: string) => runContext[runId] || runs.find((run) => run.run_id === runId);
  const leftRun = leftId ? selectedRun(leftId) : undefined;
  const rightRun = rightId ? selectedRun(rightId) : undefined;
  const visibleRuns = runs.filter((run) =>
    `${run.task} ${run.task_id} ${run.run_id}`.toLowerCase().includes(runQuery.trim().toLowerCase()),
  );
  const sameTaskGroup = Boolean(leftRun && rightRun && leftRun.agent === rightRun.agent && (
    leftRun.task_id === rightRun.task_id ||
    leftRun.parent_run_id === rightRun.run_id ||
    rightRun.parent_run_id === leftRun.run_id ||
    (leftRun.parent_run_id !== null && leftRun.parent_run_id === rightRun.parent_run_id)
  ));

  return (
    <div className="page">
      <div className="page-inner">
        <p className="label">Compare task runs</p>
        <h1 className="h1">See what changed</h1>
        <p className="muted">Compare outcomes and aligned steps. For a repair claim, compare a failed run with a fork made from that same task.</p>

        <input
          className="input compare-search"
          aria-label="Filter runs to compare"
          placeholder="Filter runs by task or ID"
          value={runQuery}
          onChange={(event) => setRunQuery(event.target.value)}
        />
        <div className="compare-select">
          <label className="compare-choice">
            <span className="label">Run A</span>
            <select className="input" value={leftId} onChange={(event) => setLeftId(event.target.value)}>
              <option value="">Choose a task run</option>
              {visibleRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.origin === "fork" ? "Replay · " : "Run · "}{run.task} · {outcomeLabel(run.status)}</option>)}
              {leftId && !visibleRuns.some((run) => run.run_id === leftId) && leftRun && <option value={leftId}>Selected · {leftRun.task}</option>}
            </select>
          </label>
          <span className="compare-arrow" aria-hidden="true">→</span>
          <label className="compare-choice">
            <span className="label">Run B</span>
            <select className="input" value={rightId} onChange={(event) => setRightId(event.target.value)}>
              <option value="">Choose a task run</option>
              {visibleRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.origin === "fork" ? "Replay · " : "Run · "}{run.task} · {outcomeLabel(run.status)}</option>)}
              {rightId && !visibleRuns.some((run) => run.run_id === rightId) && rightRun && <option value={rightId}>Selected · {rightRun.task}</option>}
            </select>
          </label>
        </div>

        {error && <p className="error-box" role="alert">{error}</p>}
        {loading && <p className="muted">Comparing the two recorded workflows…</p>}
        {leftRun && rightRun && leftId !== rightId && !sameTaskGroup && (
          <div className="error-box compare-warning" role="status">
            <strong>These are different tasks.</strong> The comparison can show how their traces differ, but it cannot establish that a change caused a better outcome. Use <em>Test suggested fix</em> from one failed run to create a controlled replay.
          </div>
        )}
        {leftRun && rightRun && leftId === rightId && (
          <div className="error-box compare-warning" role="status">Both sides point to the same run. Choose a different replay to see changes.</div>
        )}

        {diff && (
          <>
            <div className="compare-run-context">
              <div><span className="label">Run A</span><strong>{leftRun?.task || leftId}</strong></div>
              <span aria-hidden="true">→</span>
              <div><span className="label">Run B</span><strong>{rightRun?.task || rightId}</strong></div>
            </div>

            <div className="grid-cards compare-metrics">
              <div className="metric">
                <div className="label">First differing step</div>
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
                    ? "Run B completed successfully after Run A failed."
                    : diff.outcome.direction === "pass_to_fail"
                      ? "Run B failed even though Run A completed."
                      : "Both runs reached the same final result."}
                </p>
              </div>
              <span className={`badge ${diff.outcome.flipped ? "badge-accent" : "badge-neutral"}`}>
                {diff.outcome.flipped ? "Result changed" : "Same result"}
              </span>
              <Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(rightId)}`}>Inspect Run B →</Link>
            </div>

            <div className="compare-table-intro">
              <h2 className="h2">Step-by-step changes</h2>
              <p className="muted">Replay status comes from the recorded step metadata. “Downstream” shows the dependency path from the first differing step; only same-task fork comparisons can attribute an outcome change to an edit.</p>
            </div>
            <div className="card table-wrap">
              <table className="table">
                <thead><tr><th>Workflow step</th><th>What changed</th><th>Replay status</th><th>Downstream path</th><th>Run A result</th><th>Run B result</th></tr></thead>
                <tbody>
                  {diff.rows.map((row) => (
                    <tr key={row.addr}>
                      <td><strong>{stepName(row)}</strong><div className="mono faint">{row.addr}</div></td>
                      <td><span className={`badge ${row.status === "changed" ? "badge-warn" : "badge-neutral"}`}>{row.status === "changed" ? "Changed" : row.status === "same" ? "Same" : row.status === "cached" ? "Reused" : row.status === "new" ? "Added" : "Removed"}</span>{row.changed.length > 0 && <div className="faint">{changedLabel(row.changed)}</div>}</td>
                      <td>{replayLabel(row)}</td>
                      <td>{row.in_cone ? "Inside" : "Outside"}</td>
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
