"use client";

import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../../lib/api";
import type { Diagnosis, ForkCreated, ForkEdit, ForkRequest, ReplayPrediction, RunDetail } from "../../lib/contract";

type EditKind = ForkEdit["kind"];
type ReplayMode = NonNullable<ForkRequest["mode"]>;

const KIND_LABEL: Record<EditKind, [string, string]> = {
  override_output: ["Replace this step’s result", "Replacement step result · JSON"],
  patch_tool_result: ["Replace a tool’s returned data", "Replacement tool result · JSON"],
  patch_tool_args: ["Change the tool’s inputs", "Updated tool inputs · JSON object"],
  patch_prompt: ["Change the model instruction", "Text appended to the prompt · JSON string or object"],
  swap_model: ["Try another model", "New model ID · quoted JSON text"],
};

const MODE_LABEL: Record<ReplayMode, string> = {
  cone: "Smart: re-run only steps affected by the change",
  prefix: "Re-run everything from this step on",
  full: "Re-run the whole task",
};

function pretty(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2);
}

function same(a: unknown, b: unknown) {
  return JSON.stringify(a) === JSON.stringify(b);
}

interface ForkDrawerProps {
  detail: RunDetail;
  diagnosis: Diagnosis | null;
  selected: string | null;
  onSelect: (addr: string) => void;
  labelStep: (addr: string | null) => string;
  disabledReason: string | null;
  onClose: () => void;
  onStarted: (created: ForkCreated, request: ForkRequest) => void;
}

export function ForkDrawer({ detail, diagnosis, selected, onSelect, labelStep, disabledReason, onClose, onStarted }: ForkDrawerProps) {
  const step = detail.steps.find((item) => item.addr === selected);
  const suggestion = diagnosis?.proposed_fixes.find((fix) => fix.addr === selected);
  const likelyCause = diagnosis?.responsible_addr || diagnosis?.ranking[0]?.addr;
  const [kind, setKind] = useState<EditKind>("override_output");
  const [value, setValue] = useState("{}");
  const [mode, setMode] = useState<ReplayMode>("cone");
  const [control, setControl] = useState(true);
  const [samples, setSamples] = useState(5);
  const [prediction, setPrediction] = useState<ReplayPrediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // Each newly chosen step starts from its suggested correction, or its recorded result.
  useEffect(() => {
    if (suggestion) {
      setKind(suggestion.edit.kind);
      setValue(pretty(suggestion.edit.value));
    } else {
      setKind("override_output");
      setValue(pretty(step?.output));
    }
    // Only a different step resets the form; edits to the value must survive re-renders.
  }, [selected]);

  function changeKind(next: EditKind) {
    setKind(next);
    if (suggestion && suggestion.edit.kind === next) setValue(pretty(suggestion.edit.value));
    else if (next === "swap_model") setValue(pretty(detail.run.model || ""));
    else if (next === "patch_prompt") setValue(pretty(""));
    else if (next === "patch_tool_args") setValue(pretty((step?.input as { args?: unknown } | undefined)?.args ?? {}));
    else setValue(pretty(step?.output));
  }

  const parsed = useMemo(() => {
    try {
      return { ok: true as const, value: JSON.parse(value) as unknown };
    } catch {
      return { ok: false as const, value: undefined };
    }
  }, [value]);

  const request: ForkRequest | null = useMemo(() => {
    if (!step || !parsed.ok) return null;
    // Only an unmodified oracle/twin value may claim to be known-good (it enables REFUTED).
    const knownGood = Boolean(suggestion?.edit.known_good && suggestion.edit.kind === kind && same(suggestion.edit.value, parsed.value));
    return {
      base_run_id: detail.run.run_id,
      edits: [{ addr: step.addr, kind, value: parsed.value, known_good: knownGood }],
      mode,
      samples,
      control,
    };
  }, [step, parsed, suggestion, kind, mode, samples, control, detail.run.run_id]);

  // Debounced preview of what the replay will re-run, and whether this server can run it.
  useEffect(() => {
    if (!request || disabledReason) {
      setPrediction(null);
      return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      api<ReplayPrediction>("/replay/predict", { method: "POST", body: JSON.stringify(request) })
        .then((next) => { if (active) { setPrediction(next); setError(""); } })
        .catch((caught) => { if (active) { setPrediction(null); setError(errorText(caught)); } });
    }, 300);
    return () => { active = false; window.clearTimeout(timer); };
  }, [request, disabledReason]);

  async function start() {
    if (!request) return;
    setBusy(true);
    setError("");
    try {
      onStarted(await api<ForkCreated>("/forks", { method: "POST", body: JSON.stringify(request) }), request);
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  }

  const blocked = prediction?.blocked;
  const sampleOptions = control ? [5, 10] : [1, 3, 5];

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <section className="drawer" role="dialog" aria-label="Test a change" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-header">
          <div><h2 className="h2">{diagnosis?.abstain ? "Start from the leading candidate" : diagnosis ? "Start from the suspected step" : "Change a step and replay"}</h2></div>
          <button className="btn btn-sm" onClick={onClose}>Close</button>
        </div>
        <div className="drawer-body stack">
          <label className="label" htmlFor="fork-step">Step to change</label>
          <select id="fork-step" className="input" value={selected || ""} onChange={(event) => onSelect(event.target.value)}>
            {detail.steps.map((item) => (
              <option value={item.addr} key={item.addr}>
                {labelStep(item.addr)}{item.addr === diagnosis?.visible_failure_addr ? " · failure surfaced here" : ""}{item.addr === likelyCause ? ` · ${diagnosis?.abstain ? "leading candidate" : "most likely cause"}` : ""}
              </option>
            ))}
          </select>
          <div className="edit-context">
            <strong>{step?.name || "Selected step"}</strong>
            <p>{suggestion ? `${suggestion.rationale} (source: ${suggestion.source.replace("_", " ")})` : "No known correction is available for this step. Edit the value yourself to test a different idea."}</p>
          </div>

          <label className="label" htmlFor="fork-operation">What to change</label>
          <select id="fork-operation" className="input" value={kind} onChange={(event) => changeKind(event.target.value as EditKind)}>
            {(Object.keys(KIND_LABEL) as EditKind[]).map((option) => <option key={option} value={option}>{KIND_LABEL[option][0]}</option>)}
          </select>
          <label className="label" htmlFor="fork-value">{KIND_LABEL[kind][1]}</label>
          <textarea id="fork-value" className="input edit-json" value={value} onChange={(event) => setValue(event.target.value)} spellCheck={false}/>
          {!parsed.ok && <p className="error-box" role="alert">Enter valid JSON for this correction.</p>}

          <div className="fork-options">
            <label className="stack-tight">
              <span className="label">Replay</span>
              <select className="input" value={mode} onChange={(event) => setMode(event.target.value as ReplayMode)}>
                {(Object.keys(MODE_LABEL) as ReplayMode[]).map((option) => <option key={option} value={option}>{MODE_LABEL[option]}</option>)}
              </select>
            </label>
            <label className="stack-tight">
              <span className="label">Samples</span>
              <select className="input" value={samples} onChange={(event) => setSamples(Number(event.target.value))}>
                {sampleOptions.map((option) => <option key={option} value={option}>{option}</option>)}
              </select>
            </label>
            <label className="prompt-checkbox fork-control">
              <input type="checkbox" checked={control} onChange={(event) => {
                setControl(event.target.checked);
                if (event.target.checked && samples < 5) setSamples(5);
              }} />
              <span><strong>Paired control</strong><small>Also re-run unchanged with the same seeds. Required for a verdict.</small></span>
            </label>
          </div>

          {disabledReason ? (
            <p className="notice-box">{disabledReason}</p>
          ) : blocked ? (
            <div className="notice-box" role="alert"><strong>This replay cannot run here</strong><p>{blocked.message} {blocked.hint}</p></div>
          ) : prediction ? (
            <div className="edit-context"><strong>{prediction.summary}</strong><p>{control ? `${samples} paired runs: your change versus the original, with identical seeds.` : "Preview only: without a control the result cannot be verified."}</p></div>
          ) : null}
          {error && <p className="error-box" role="alert">{error}</p>}
        </div>
        <div className="drawer-footer">
          <span className="muted">{prediction ? `${prediction.reexecute_steps} of ${prediction.total_steps} steps re-run` : " "}</span><span className="spacer"/>
          <button className="btn btn-primary" onClick={start} disabled={!request || busy || Boolean(blocked) || Boolean(disabledReason)}>
            {busy ? "Starting…" : control ? "Run comparison" : "Run preview"}
          </button>
        </div>
      </section>
    </div>
  );
}
