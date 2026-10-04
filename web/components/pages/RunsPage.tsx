"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { RunDetail, RunList } from "../../lib/contract";

// These are the three retained TripCrew examples with paired, five-sample
// validation in the local training/evaluation database. Check availability so
// the cards never send a user to missing data in another deployment.
const DEMO_CASES = [
  {
    id: "375afbca1cdc42299f60b47338cf252d",
    fixedId: "e719324dea1c4cfb8f5ca691406e8e47-fix-0",
    title: "Mumbai → Bangkok",
    detail: "The budget was undercounted after currency conversion.",
  },
  {
    id: "d4e82d3fc62c47e7835bda5a104cdb5a",
    fixedId: "c668d09128724998b4bbffe88ee98413-fix-0",
    title: "Mumbai → Paris",
    detail: "The converted budget did not match the current catalog.",
  },
  {
    id: "29434fc3cc024e25a4b4fb5c2c686c8f",
    fixedId: "d2e4dcbf59c84ffe8191d1db4c80a0ff-fix-0",
    title: "Bengaluru → Singapore",
    detail: "A currency mismatch caused the final total to fail.",
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
          <details className="demo-note"><summary>About these examples</summary><p>The model flags exchange-rate conversion as a lead, with low confidence. A controlled replay tests the suggested fix against five samples. All three examples use the same fault in different currencies.</p></details>
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
