"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { RunDetail, RunList } from "../../lib/contract";

// These three retained TripCrew runs have passing replay pairs in the local
// demo database. Check availability so other deployments can fall back to the
// full run history rather than linking to missing data.
const DEMO_CASES = [
  {
    id: "7fa0eef807db4263b06e48f5a6f5d8a1",
    fixedId: "06378d7b415e47b0a296c37827984432-fix-0",
    title: "Hyderabad → London",
    detail: "The final INR total was lower than the checker expected.",
  },
  {
    id: "6c1f43f4b87d4765b7e4a82d59ad74f7",
    fixedId: "244b704a7a1449f4a1bd631763bfe913-fix-0",
    title: "Chennai → Dubai",
    detail: "A stale exchange rate caused the budget total to fail.",
  },
  {
    id: "637518630bf641faad5ced4c1cb500fe",
    fixedId: "63f17e63d82941628cdb7ac97df713f1-fix-0",
    title: "Bengaluru → Bangkok",
    detail: "The converted travel budget did not match the catalog total.",
  },
];

export function RunsPage() {
  const [data, setData] = useState<RunList | null>(null);
  const [error, setError] = useState("");
  const [outcome, setOutcome] = useState("");
  const [query, setQuery] = useState("");
  const [availableCases, setAvailableCases] = useState<string[] | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all(
      DEMO_CASES.map(async (sample) => {
        try {
          await Promise.all([
            api<RunDetail>(`/runs/${encodeURIComponent(sample.id)}`),
            api<RunDetail>(`/runs/${encodeURIComponent(sample.fixedId)}`),
          ]);
          return sample.id;
        } catch {
          return null;
        }
      }),
    ).then((ids) => {
      if (active) setAvailableCases(ids.filter((id): id is string => Boolean(id)));
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const params = new URLSearchParams({ limit: "100" });
    if (outcome) params.set("outcome", outcome);
    if (query) params.set("q", query);
    api<RunList>(`/runs?${params}`).then(setData).catch((caught) => setError(caught.message));
  }, [outcome, query]);

  const selectedCases = DEMO_CASES.filter((sample) => availableCases?.includes(sample.id));

  return (
    <div className="page">
      <div className="page-inner">
        <div className="spread runs-heading">
          <div>
            <p className="label">Demo cases</p>
            <h1 className="h1">Inspect a failed trip</h1>
            <p className="muted">Trace the run, review the likely cause, then test a fix.</p>
          </div>
          <div className="row">
            <Link className="btn" href="/compare">Compare runs</Link>
            <Link className="btn btn-primary" href="/new">Try a custom task</Link>
          </div>
        </div>

        <section className="demo-cases" aria-labelledby="demo-cases-title">
          <div className="demo-cases-heading">
            <div>
              <p className="label">Recorded examples</p>
              <h2 className="h2" id="demo-cases-title">One issue across three trips</h2>
            </div>
          </div>

          {availableCases === null && <p className="muted">Checking the recorded demo data…</p>}
          {availableCases !== null && selectedCases.length === 0 && <p className="muted">The curated examples are not available in this dataset. Browse the recorded runs below instead.</p>}
          <div className="demo-case-grid">
            {selectedCases.map((sample, index) => (
              <article className="card demo-case" key={sample.id}>
                <div className="demo-case-top"><span className="badge badge-pass">Saved replay</span></div>
                <h3>{sample.title}</h3>
                <p>{sample.detail}</p>
                <p className="demo-case-clue">Likely step: exchange rate</p>
                <div className="demo-case-actions">
                  <Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(sample.id)}`}>Inspect failed run</Link>
                  <Link className="btn btn-sm btn-primary" href={`/fork/${encodeURIComponent(sample.id)}`}>Test suggested fix</Link>
                </div>
                <Link className="demo-compare-link" href={`/compare?a=${encodeURIComponent(sample.id)}&b=${encodeURIComponent(sample.fixedId)}`}>View saved before-and-after result →</Link>
              </article>
            ))}
          </div>
          <details className="demo-note"><summary>About these examples</summary><p>All three runs point to an exchange-rate mismatch. Each has a saved passing replay so you can inspect the diagnosis and compare the changed run.</p></details>
        </section>

        {error && <p className="error-box" role="alert">Could not load run history: {error}</p>}
        <details className="history-details">
          <summary>Browse all recorded runs{data ? ` · ${data.total.toLocaleString()} available` : ""}</summary>
          {data && (
            <div className="history-content">
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
                <span className="muted">Newest first</span>
              </div>

              <div className="card table-wrap">
                <table className="table">
                  <thead><tr><th>Result</th><th>Task</th><th>Agent</th><th>Steps</th><th>Failure risk</th><th>Likely step</th><th/></tr></thead>
                  <tbody>
                    {data.items.map((run) => (
                      <tr key={run.run_id}>
                        <td><span className={`badge ${run.status === "failed" ? "badge-fail" : "badge-pass"}`}>{run.status === "failed" ? "Failed" : "Completed"}</span></td>
                        <td><strong>{run.task}</strong><div className="faint mono run-id">{run.run_id}</div></td>
                        <td>{run.agent}</td>
                        <td className="num">{run.steps}</td>
                        <td className="num">{run.risk === null ? "—" : `${Math.round(run.risk * 100)}%`}</td>
                        <td>{run.top_suspect?.name || "No likely step"}</td>
                        <td><Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(run.run_id)}`}>Open →</Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </details>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return <div className="metric"><div className="label">{label}</div><div className="value">{value.toLocaleString()}</div></div>;
}
