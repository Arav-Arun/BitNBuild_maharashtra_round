"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorText } from "../../lib/api";
import { useDebounced } from "../../lib/useDebounced";
import type { LabelQueue, LabelResponse, RunDetail } from "../../lib/contract";
import { useAppContext } from "../shell/AppContext";

type Certainty = "sure" | "unsure";

export function LabelPage() {
  const { staticBundle } = useAppContext();
  const [who, setWho] = useState("annotator-1");
  const [queue, setQueue] = useState<LabelQueue | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [root, setRoot] = useState("");
  const [certainty, setCertainty] = useState<Certainty>("sure");
  const [notes, setNotes] = useState("");
  const [msg, setMsg] = useState("");
  const [saving, setSaving] = useState(false);
  const annotator = useDebounced(who.trim(), 400);
  const queueRequest = useRef(0);

  const load = useCallback(async () => {
    const request = ++queueRequest.current;
    try {
      if (!annotator) {
        setQueue(null);
        return;
      }
      const next = await api<LabelQueue>(`/labels/queue?annotator=${encodeURIComponent(annotator)}&limit=100`);
      if (request === queueRequest.current) setQueue(next);
    } catch (caught) {
      if (request === queueRequest.current) setMsg(errorText(caught));
    }
  }, [annotator]);

  useEffect(() => {
    setQueue(null);
    load();
    return () => { queueRequest.current++; };
  }, [load]);

  const item = queue?.items[0];

  useEffect(() => {
    let active = true;
    setRun(null);
    setRoot("");
    setCertainty("sure");
    setNotes("");
    if (!item) {
      return;
    }
    // blind=true hides the model's guess and any injected label while the annotator decides.
    api<RunDetail>(`/runs/${encodeURIComponent(item.run_id)}?blind=true`)
      .then((detail) => {
        if (active) setRun(detail);
      })
      .catch((caught) => { if (active) setMsg(errorText(caught)); });
    return () => { active = false; };
  }, [item?.run_id]);

  async function submit() {
    if (!item) return;
    setSaving(true);
    try {
      const result = await api<LabelResponse>("/labels", {
        method: "POST",
        body: JSON.stringify({ run_id: item.run_id, annotator, root_addr: root || null, certainty, notes: notes || null }),
      });
      const kappa = result.agreement.kappa;
      setMsg(`Saved · ${result.progress.labelled}/${result.progress.target} labelled${kappa != null ? ` · agreement κ ${kappa.toFixed(2)} over ${result.agreement.n} shared runs` : ""}`);
      setNotes("");
      await load();
    } catch (caught) {
      setMsg(errorText(caught));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="page">
      <div className="page-inner">
        <h1 className="h1">Mark the step that caused the failure</h1>
        <p className="muted">The model’s guess and injected labels stay hidden while you review, so labels can measure the model honestly.</p>
        {staticBundle && <p className="notice-box">Static showcase mode is read-only. Start the local API to save labels.</p>}
        <div className="row label-toolbar">
          <label className="label" htmlFor="annotator">Annotator</label>
          <input id="annotator" className="input" value={who} disabled={saving} onChange={(event) => setWho(event.target.value)} />
          <span className="badge badge-neutral">{queue?.progress.labelled || 0}/{queue?.progress.target || 0} labelled</span>
        </div>
        {msg && <p className="muted" role="status">{msg}</p>}
        {item && run ? (
          <div className="label-layout">
            <section className="card card-pad stack">
              <div className="spread"><span className="badge badge-fail">failed</span><span className="mono faint">{item.run_id}</span></div>
              <h2 className="h2">{item.task}</h2>
              <p className="muted">Observed result: {item.checker_reason || "Run failed"}</p>
              <label className="label" htmlFor="root-step">Root cause step</label>
              <select id="root-step" className="input full" value={root} onChange={(event) => setRoot(event.target.value)}>
                <option value="">No responsible step</option>
                {run.steps.map((step) => <option key={step.addr} value={step.addr}>{step.seq + 1}. {step.name} · {step.addr}</option>)}
              </select>
              <div className="segmented" role="radiogroup" aria-label="How certain are you?">
                {(["sure", "unsure"] as const).map((option) => (
                  <button key={option} type="button" role="radio" aria-checked={certainty === option} className={certainty === option ? "active" : ""} onClick={() => setCertainty(option)}>
                    {option === "sure" ? "I’m sure" : "Not sure"}
                  </button>
                ))}
              </div>
              <textarea className="input edit-json" placeholder="Reason for your choice (optional)" value={notes} onChange={(event) => setNotes(event.target.value)} />
              <button className="btn btn-primary" onClick={submit} disabled={staticBundle || saving || !annotator || who.trim() !== annotator}>{saving ? "Saving…" : "Save label"}</button>
            </section>
            <section className="thread-pane label-steps">
              <div className="thread-header"><strong>Recorded sequence</strong></div>
              <div className="thread-body">
                {run.steps.map((step) => (
                  <button className={`label-step ${root === step.addr ? "active" : ""}`} onClick={() => setRoot(step.addr)} key={step.addr}>
                    <span className="mono faint">{String(step.seq + 1).padStart(2, "0")}</span>
                    <strong>{step.name}</strong>
                    <span className="mono faint">{step.addr}</span>
                  </button>
                ))}
              </div>
            </section>
          </div>
        ) : (
          <div className="card empty">{queue ? "No unlabelled failed runs are waiting for this annotator." : "Loading the review queue…"}</div>
        )}
      </div>
    </div>
  );
}
