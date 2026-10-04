"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "../../lib/api";
import type { RunList } from "../../lib/contract";

export function RunsPage() {
 const [data,setData]=useState<RunList|null>(null), [error,setError]=useState(""), [outcome,setOutcome]=useState(""), [q,setQ]=useState("");
 useEffect(()=>{ const p=new URLSearchParams({limit:"100"}); if(outcome)p.set("outcome",outcome); if(q)p.set("q",q); api<RunList>(`/runs?${p}`).then(setData).catch(e=>setError(e.message)); },[outcome,q]);
 return <div className="page"><div className="page-inner"><div className="spread"><div><p className="label">Flight recorder / Runs</p><h1 className="h1">Agent runs</h1><p className="muted">Recorded executions, ranked by risk and ready to investigate.</p></div><Link className="btn btn-primary" href="/label">Label failures ↗</Link></div>
 {error&&<p className="error-box">{error}</p>}{data&&<><div className="grid-cards" style={{margin:"22px 0"}}><Metric label="Visible runs" value={data.total}/><Metric label="Failed" value={data.facets.outcomes.failed||0}/><Metric label="Passing" value={data.facets.outcomes.passed||0}/><Metric label="Failure groups" value={Object.keys(data.facets.agents).length}/></div>
 <div className="row" style={{marginBottom:12}}><input className="input" placeholder="Search task or run id" value={q} onChange={e=>setQ(e.target.value)}/><select className="input" value={outcome} onChange={e=>setOutcome(e.target.value)}><option value="">All outcomes</option><option value="failed">Failed</option><option value="passed">Passed</option></select><span className="muted">{data.total} runs · newest first</span></div>
 <div className="card table-wrap"><table className="table"><thead><tr><th>Outcome</th><th>Task</th><th>Agent</th><th>Steps</th><th>Duration</th><th>Risk</th><th>Top suspect</th><th/></tr></thead><tbody>{data.items.map(r=><tr key={r.run_id}><td><span className={`badge ${r.status==='failed'?'badge-fail':'badge-pass'}`}>{r.status}</span></td><td><strong>{r.task}</strong><div className="faint mono" style={{fontSize:11}}>{r.run_id}</div></td><td>{r.agent}</td><td className="num">{r.steps}</td><td className="num">{Math.round(r.duration_ms)} ms</td><td className="num">{r.risk===null?'—':`${Math.round(r.risk*100)}%`}</td><td>{r.top_suspect?.name||"—"}</td><td><Link className="btn btn-sm" href={`/investigate/${encodeURIComponent(r.run_id)}`}>Inspect →</Link></td></tr>)}</tbody></table></div></>}</div></div>
}
function Metric({label,value}:{label:string;value:number}) {return <div className="metric"><div className="label">{label}</div><div className="value">{value.toLocaleString()}</div></div>}
