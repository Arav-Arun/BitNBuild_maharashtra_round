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
        <p className="label">TripCrew / New run</p>
        <h1 className="h1">Give the agent a task</h1>
        <p className="muted new-run-intro">
          The local demo runner turns a supported trip request into a recorded multi-agent workflow.
          You can inspect every model, tool, and state step when it finishes.
        </p>

        <form className="new-run-form card card-pad" onSubmit={submit}>
          <label className="label" htmlFor="task-prompt">Task prompt</label>
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
            <span className="faint">{prompt.length}/1200 · Edit the example or write your own</span>
            <button type="button" className="btn btn-sm" onClick={() => setPrompt(SAMPLE)}>Use example</button>
          </div>

          <div className="prompt-help">
            Include an origin and destination, two dates in YYYY-MM-DD format, traveler count, and
            INR budget. Offline catalog: Mumbai, Delhi, Bengaluru, Chennai, or Hyderabad to
            Singapore, Bangkok, Dubai, London, Tokyo, or Paris. Preferences such as “vegetarian: yes”
            are optional.
          </div>

          <label className="prompt-checkbox">
            <input type="checkbox" checked={injectStaleFx} onChange={(event) => setInjectStaleFx(event.target.checked)} />
            <span><strong>Inject a stale exchange-rate fixture</strong><small>Create a reproducible failure to investigate. Synthetic data only.</small></span>
          </label>

          {error && <p className="error-box" role="alert">{error}</p>}
          {!enabled && <p className="muted">Prompt runs need the local API in OFFLINE mode. The current recorder is {mode?.toUpperCase() || "connecting"}.</p>}
          <div className="row prompt-actions">
            <span className="faint">Offline deterministic runner · no booking or external calls</span>
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
