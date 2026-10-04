"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import { useDebounced } from "../../lib/useDebounced";
import type { FailureGroupList, RunDetail, RunList } from "../../lib/contract";

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
  const [agent, setAgent] = useState("");
  const [split, setSplit] = useState("");
  const [signature, setSignature] = useState("");
  const [groups, setGroups] = useState<FailureGroupList | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const search = useDebounced(query.trim());
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
    api<FailureGroupList>("/failure-groups").then(setGroups).catch(() => setGroups(null));
  }, []);

  useEffect(() => {
    const params = new URLSearchParams({ limit: "100" });
    if (outcome) params.set("outcome", outcome);
    if (search) params.set("q", search);
    if (agent) params.set("agent", agent);
    if (split) params.set("split", split);
    if (signature) params.set("failure_signature", signature);
    api<RunList>(`/runs?${params}`).then((list) => { setData(list); setError(""); }).catch((caught) => setError(caught.message));
  }, [outcome, search, agent, split, signature]);

  function showGroup(next: string) {
    setSignature(next);
    setOutcome("");
    setHistoryOpen(true);
  }

  const activeGroup = groups?.items.find((group) => group.signature === signature);

  const selectedCases = DEMO_CASES.filter((sample) => availableCases?.includes(sample.id));

  return (
    <div className="page">
      <div className="page-inner">
        <div className="spread runs-heading">
          <div>
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
              <h2 className="h2" id="demo-cases-title">One issue across three trips</h2>
              <p className="muted section-note">Each plan failed its budget check several steps after the real mistake.</p>
            </div>
          </div>

          {availableCases === null && <p className="muted">Checking the recorded demo data…</p>}
          {availableCases !== null && selectedCases.length === 0 && <p className="muted">The curated examples are not available in this dataset. Browse the recorded runs below instead.</p>}
          <div className="demo-case-grid">
            {selectedCases.map((sample) => (
              <article className="card demo-case" key={sample.id}>
                <h3>{sample.title}</h3>
                <p>{sample.detail}</p>
                <p className="demo-case-clue">Likely cause <span>the exchange rate</span></p>
                <div className="demo-case-actions">
                  <Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(sample.id)}`}>Inspect</Link>
                  <Link className="btn btn-sm btn-primary" href={`/fork/${encodeURIComponent(sample.id)}`}>Test the fix</Link>
                </div>
                <Link className="demo-compare-link" href={`/compare?a=${encodeURIComponent(sample.id)}&b=${encodeURIComponent(sample.fixedId)}`}>See the saved before and after</Link>
              </article>
            ))}
          </div>
        </section>

        {groups && groups.items.length > 0 && (
          <section className="failure-groups" aria-labelledby="failure-groups-title">
            <div className="demo-cases-heading">
              <div>
                <h2 className="h2" id="failure-groups-title">How runs fail</h2>
                <p className="muted section-note">{groups.total_failed.toLocaleString()} failed runs, grouped by the step the model blames. Select one to see its runs.</p>
              </div>
            </div>
            <div className="group-grid">
              {groups.items.slice(0, 6).map((group) => (
                <button type="button" key={group.signature} className={`group-card${signature === group.signature ? " active" : ""}`} onClick={() => showGroup(group.signature)}>
                  <span className="group-count">{group.count}</span>
                  <span className="group-label">{group.top_suspect_name ?? "Unattributed step"}</span>
                  <span className="faint">{(group.signature.split(":")[2] ?? "").replaceAll("_", " ")}</span>
                </button>
              ))}
            </div>
          </section>
        )}

        {error && <p className="error-box" role="alert">Could not load run history: {error}</p>}
        <details className="history-details" open={historyOpen} onToggle={(event) => setHistoryOpen(event.currentTarget.open)}>
          <summary>All recorded runs{data ? ` (${data.total.toLocaleString()})` : ""}</summary>
          {data && (
            <div className="history-content">
              <p className="muted runs-summary">
                {data.total.toLocaleString()} runs match: {(data.facets.outcomes.failed || 0).toLocaleString()} failed and {(data.facets.outcomes.passed || 0).toLocaleString()} completed.
              </p>

              <div className="runs-filters">
                <input className="input" aria-label="Search task history" placeholder="Search task text or run ID" value={query} onChange={(event) => setQuery(event.target.value)}/>
                <select className="input" aria-label="Filter by task result" value={outcome} onChange={(event) => setOutcome(event.target.value)}>
                  <option value="">Any result</option>
                  <option value="failed">Failed</option>
                  <option value="passed">Completed</option>
                </select>
                <select className="input" aria-label="Filter by agent" value={agent} onChange={(event) => setAgent(event.target.value)}>
                  <option value="">Any agent</option>
                  {Object.keys(data.facets.agents).sort().map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
                <select className="input" aria-label="Filter by evaluation split" value={split} onChange={(event) => setSplit(event.target.value)}>
                  <option value="">Any split</option>
                  {Object.keys(data.facets.splits).sort().map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
                <span className="muted">Newest first</span>
              </div>
              {signature && (
                <p className="row filter-note">
                  <span className="badge badge-accent">{activeGroup?.top_suspect_name ?? signature}</span>
                  <button type="button" className="btn btn-sm btn-ghost" onClick={() => setSignature("")}>Clear</button>
                </p>
              )}

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
                        <td><Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(run.run_id)}`}>Open</Link></td>
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
