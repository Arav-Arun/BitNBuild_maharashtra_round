"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { api } from "../../lib/api";
import { useAppContext } from "../shell/AppContext";

const SAMPLE =
  "Plan a trip from Mumbai to Singapore departing 2026-12-12, returning 2026-12-15, for 4 adults. Budget INR 160000. Vegetarian: no; refundable: no; no red-eye: no.";

type TaskRunResponse = { run_id: string; status: "passed" | "failed"; task: string; steps: number };

export function NewRunPage() {
  const router = useRouter();
  const { mode, staticBundle } = useAppContext();
  const [prompt, setPrompt] = useState(SAMPLE);
  const [injectStaleFx, setInjectStaleFx] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const result = await api<TaskRunResponse>("/tasks/run", {
        method: "POST",
        body: JSON.stringify({ prompt, inject_stale_fx: injectStaleFx }),
      });
      router.push(`/investigate/${encodeURIComponent(result.run_id)}`);
    } catch (caught) {
      setError((caught as Error).message || "The task could not be run.");
      setBusy(false);
    }
  }

  const enabled = mode === "offline" && !staticBundle;

  return (
    <div className="page">
      <div className="page-inner new-run-page">
        <h1 className="h1">Plan a trip</h1>
        <p className="muted new-run-intro">Enter a request. Inspect each step after the run.</p>

        <form className="new-run-form card card-pad" onSubmit={submit}>
          <label className="label" htmlFor="task-prompt">Your request</label>
          <textarea
            id="task-prompt"
            className="input prompt-input"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={1200}
            required
            minLength={24}
            spellCheck={false}
            placeholder={SAMPLE}
          />
          <div className="spread prompt-meta">
            <span className="faint">{prompt.length}/1200</span>
            <button type="button" className="btn btn-sm" onClick={() => setPrompt(SAMPLE)}>Reset example</button>
          </div>

          <div className="prompt-help">Include a route, dates, travelers, and budget.</div>

          <label className="prompt-checkbox">
            <input type="checkbox" checked={injectStaleFx} onChange={(event) => setInjectStaleFx(event.target.checked)} />
            <span><strong>Use an old exchange rate</strong><small>Add a reproducible budget error.</small></span>
          </label>

          {error && <p className="error-box" role="alert">{error}</p>}
          {!enabled && <p className="muted">Local runner unavailable. Check that the API is running.</p>}
          <div className="row prompt-actions">
            <span className="spacer" />
            <button className="btn btn-primary" type="submit" disabled={!enabled || busy || prompt.trim().length < 24}>
              {busy ? "Recording run…" : "Run and inspect →"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
