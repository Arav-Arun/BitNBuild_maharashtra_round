"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { RunList } from "../../lib/contract";

export function RunsPage() {
  const [data, setData] = useState<RunList | null>(null);
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState("");
  const [query, setQuery] = useState("");

  useEffect(() => {
    const params = new URLSearchParams({ limit: "100" });
    if (outcome) params.set("outcome", outcome);
    if (query) params.set("q", query);
    api<RunList>(`/runs?${params}`).then(setData).catch((caught) => setError(caught.message));
  }, [outcome, query]);

  return (
    <div className="page">
      <div className="page-inner">
        <div className="spread runs-heading">
          <div>
            <p className="label">Black Box / Recorded workflows</p>
            <h1 className="h1">Agent task history</h1>
            <p className="muted">Each row is a task the agent ran. Open one to follow its steps, see where a failure appeared, and review the evidence.</p>
          </div>
          <div className="row">
            <Link className="btn" href="/label">Review failures ↗</Link>
            <Link className="btn btn-primary" href="/new">Give the agent a task ↗</Link>
          </div>
        </div>

        {error && <p className="error-box" role="alert">{error}</p>}
        {data && (
          <>
            <div className="grid-cards runs-metrics">
              <Metric label="Tasks shown" value={data.total}/>
              <Metric label="Failed" value={data.facets.outcomes.failed || 0}/>
              <Metric label="Completed" value={data.facets.outcomes.passed || 0}/>
              <Metric label="Agents" value={Object.keys(data.facets.agents).length}/>
            </div>

            <div className="runs-filters">
              <input className="input" aria-label="Search task history" placeholder="Search task text or run ID" value={query} onChange={(event) => setQuery(event.target.value)}/>
              <select className="input" aria-label="Filter by task result" value={outcome} onChange={(event) => setOutcome(event.target.value)}>
                <option value="">Any result</option>
                <option value="failed">Failed</option>
                <option value="passed">Completed</option>
              </select>
              <span className="muted">{data.total.toLocaleString()} tasks · newest first</span>
            </div>

            <div className="card table-wrap">
              <table className="table">
                <thead><tr><th>Result</th><th>What the agent was asked to do</th><th>Agent</th><th>Steps</th><th>Time</th><th>Failure risk</th><th>Most likely step</th><th/></tr></thead>
                <tbody>
                  {data.items.map((run) => (
                    <tr key={run.run_id}>
                      <td><span className={`badge ${run.status === "failed" ? "badge-fail" : "badge-pass"}`}>{run.status === "failed" ? "Failed" : "Completed"}</span></td>
                      <td><strong>{run.task}</strong><div className="faint mono run-id">{run.run_id}</div></td>
                      <td>{run.agent}</td>
                      <td className="num">{run.steps}</td>
                      <td className="num">{Math.round(run.duration_ms)} ms</td>
                      <td className="num">{run.risk === null ? "Not available" : `${Math.round(run.risk * 100)}%`}</td>
                      <td>{run.top_suspect?.name || "No likely step"}</td>
                      <td><Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(run.run_id)}`}>Open workflow →</Link></td>
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

function Metric({ label, value }: { label: string; value: number }) {
  return <div className="metric"><div className="label">{label}</div><div className="value">{value.toLocaleString()}</div></div>;
}
