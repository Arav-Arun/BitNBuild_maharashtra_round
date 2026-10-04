"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { RunGraph } from "../graph/RunGraph";
import { StepThread } from "../thread/StepThread";
import type { EdgeView, StepView, ValueClick } from "../types";
import type {
  Diagnosis,
  ForkCreated,
  ForkEdit,
  ForkRequest,
  ForkSummary,
  RunDetail,
} from "../../lib/contract";
import { API, api } from "../../lib/api";
import { useAppContext } from "../shell/AppContext";

function editValue(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2);
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    localize: "Find the problem",
    attribute: "Trace the evidence",
    propose: "Suggest a correction",
    verify: "Test the correction",
  };
  return labels[stage] || stage;
}

function taskPromptFromInput(input: unknown): string | null {
  if (!input || typeof input !== "object") return null;
  const messages = (input as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return null;
  for (const message of [...messages].reverse()) {
    if (!message || typeof message !== "object") continue;
    const item = message as { role?: unknown; content?: unknown };
    if (item.role !== "user" || typeof item.content !== "string") continue;
    try {
      const envelope = JSON.parse(item.content) as { task?: { original_prompt?: unknown } };
      if (typeof envelope.task?.original_prompt === "string") return envelope.task.original_prompt;
    } catch {
      continue;
    }
  }
  return null;
}

export function InvestigatePage({ runId, startEditing = false }: { runId: string; startEditing?: boolean }) {
  const { staticBundle } = useAppContext();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [editing, setEditing] = useState(startEditing);
  const [kind, setKind] = useState<ForkEdit["kind"]>("override_output");
  const [value, setValue] = useState("{}");
  const [status, setStatus] = useState("");
  const [forkId, setForkId] = useState("");
  const [fork, setFork] = useState<ForkSummary | null>(null);
  const [selectedPointer, setSelectedPointer] = useState<ValueClick | null>(null);

  const load = useCallback(async () => {
    try {
      const run = await api<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
      const diagnosisRunId = run.run.status === "failed" ? runId : run.run.parent_run_id;
      const result = diagnosisRunId
        ? await api<Diagnosis>(`/runs/${encodeURIComponent(diagnosisRunId)}/diagnosis`)
        : null;
      setDetail(run);
      setDiagnosis(result);
      setSelected(
        result?.responsible_addr ||
          result?.visible_failure_addr ||
          result?.ranking[0]?.addr ||
          run.steps[0]?.addr ||
          null,
      );
      setError("");
    } catch (caught) {
      setError((caught as Error).message);
    }
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!forkId) return;
    const source = new EventSource(`${API}/forks/${encodeURIComponent(forkId)}/stream`);
    const update = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (event.type === "step") {
          setStatus(
            `Sample ${data.sample + 1}: ${data.addr} · ${data.phase}${data.cache_status ? ` · ${data.cache_status}` : ""}`,
          );
        }
        if (event.type === "outcome") {
          setStatus(`Sample ${data.sample + 1} ${data.branch}: ${data.passed ? "completed" : "failed"}`);
        }
        if (event.type === "summary") {
          setStatus(`Comparison complete · ${data.verdict || "preview"} · ${data.cached} recorded steps reused`);
          source.close();
          api<ForkSummary>(`/forks/${encodeURIComponent(forkId)}`).then(setFork).catch(() => {});
          load();
        }
      } catch {
        // Ignore malformed stream events; the status endpoint remains the source of truth.
      }
    };
    for (const name of ["step", "outcome", "summary", "error"]) {
      source.addEventListener(name, update as EventListener);
    }
    source.onerror = () => {
      api<ForkSummary>(`/forks/${encodeURIComponent(forkId)}`)
        .then((result) => {
          setFork(result);
          if (result.status === "complete" || result.status === "error") {
            source.close();
            setStatus(`Comparison ${result.status} · ${result.verdict || result.error?.message || ""}`);
            load();
          }
        })
        .catch(() => {});
    };
    return () => source.close();
  }, [forkId, load]);

  const steps: StepView[] = useMemo(
    () =>
      detail?.steps.map((step) => ({
        addr: step.addr,
        seq: step.seq,
        kind: step.kind as StepView["kind"],
        name: step.name,
        role: step.agent_role,
        input: step.input,
        output: step.output,
        stateBefore: step.state_before,
        stateAfter: step.state_after,
        reasoning: step.reasoning.join("\n"),
        latencyMs: step.latency_ms,
        tokens: step.tokens ? step.tokens.input + step.tokens.output : null,
        error: step.error?.message || null,
        cacheStatus: step.cache_status,
        suspicion: diagnosis?.ranking.find((item) => item.addr === step.addr)?.probability,
        isSuspect: diagnosis?.responsible_addr === step.addr,
        isVisibleFailure: diagnosis?.visible_failure_addr === step.addr,
        violations: step.rule_violations.map((violation) => ({
          pointer: violation.citation.json_pointer || "",
          message: violation.message,
        })),
      })) || [],
    [detail, diagnosis],
  );

  const edges: EdgeView[] = useMemo(
    () => detail?.edges.map((edge, index) => ({
      id: `e${index}`,
      source: edge.src_addr,
      target: edge.dst_addr,
      kind: edge.kind,
      label: edge.key,
    })) || [],
    [detail],
  );

  if (error && !detail) {
    return <div className="page"><div className="page-inner"><p className="error-box">{error}</p></div></div>;
  }
  if (!detail) {
    return <div className="page"><div className="page-inner"><p className="muted">Loading recorded task…</p></div></div>;
  }

  const currentDetail = detail;
  const selectedStep = currentDetail.steps.find((step) => step.addr === selected);
  const labelStep = (addr: string | null) => {
    const step = currentDetail.steps.find((item) => item.addr === addr);
    return step ? `Step ${step.seq + 1}: ${step.name}` : "not isolated";
  };
  const originalTask =
    currentDetail.run.agent === "tripcrew"
      ? taskPromptFromInput(currentDetail.steps[0]?.input) || currentDetail.task_text || currentDetail.run.task
      : currentDetail.task_text || currentDetail.run.task;
  const proposedEdit = (addr: string) => diagnosis?.proposed_fixes.find((fix) => fix.addr === addr);

  function applySuggestion(addr: string | null) {
    if (!addr) return;
    const step = currentDetail.steps.find((item) => item.addr === addr);
    if (!step) return;
    const suggestion = proposedEdit(addr);
    setSelected(addr);
    if (suggestion) {
      setKind(suggestion.edit.kind);
      setValue(editValue(suggestion.edit.value));
    } else {
      setKind("override_output");
      setValue(editValue(step.output));
    }
  }

  function openFixDrawer() {
    const address =
      diagnosis?.responsible_addr ||
      diagnosis?.proposed_fixes[0]?.addr ||
      diagnosis?.visible_failure_addr ||
      diagnosis?.ranking[0]?.addr ||
      currentDetail.steps[0]?.addr;
    applySuggestion(address || null);
    setError("");
    setEditing(true);
  }

  function changeOperation(nextKind: ForkEdit["kind"]) {
    setKind(nextKind);
    if (nextKind === "swap_model") {
      setValue(editValue("tripcrew-fixture-v1"));
    } else if (nextKind === "patch_prompt") {
      setValue(editValue(""));
    } else if (nextKind === "patch_tool_args") {
      setValue(editValue(selectedStep?.input));
    } else {
      setValue(editValue(selectedStep?.output));
    }
  }

  async function startFork() {
    if (!selectedStep || !diagnosis) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(value);
    } catch {
      setError("Enter a valid JSON value for this correction.");
      return;
    }
    const suggestion = proposedEdit(selectedStep.addr);
    const edit: ForkEdit = {
      addr: selectedStep.addr,
      kind,
      value: parsed,
      known_good: suggestion?.edit.known_good,
    };
    const request: ForkRequest = {
      base_run_id: runId,
      edits: [edit],
      mode: "cone",
      samples: 5,
      control: true,
    };
    setError("");
    setStatus("Starting five paired tests: your correction versus the original run…");
    try {
      const created = await api<ForkCreated>("/forks", {
        method: "POST",
        body: JSON.stringify(request),
      });
      setForkId(created.fork_id);
      setStatus("Paired test queued…");
    } catch (caught) {
      setError((caught as Error).message);
      setStatus("");
    }
  }

  const likelyCause = diagnosis?.responsible_addr || diagnosis?.ranking[0]?.addr;
  const likelySuggestion = likelyCause ? proposedEdit(likelyCause) : undefined;
  const correctionLabel: Record<ForkEdit["kind"], string> = {
    override_output: "Replacement step result · JSON",
    patch_tool_result: "Replacement tool result · JSON",
    patch_tool_args: "Updated tool inputs · JSON",
    patch_prompt: "Updated model prompt · JSON",
    swap_model: "New model ID · quoted JSON text",
  };

  return (
    <div className="page investigate-page">
      <div className="investigate-top">
        <div className="investigate-heading">
          <p className="label">{detail.run.agent} · Recorded task</p>
          <h1 className="h1">{detail.run.task}</h1>
          <p className="task-statement"><strong>What the agent was asked:</strong> {originalTask}</p>
          <div className="row muted investigate-meta">
            <span className={`badge ${detail.run.status === "failed" ? "badge-fail" : "badge-pass"}`}>
              {detail.run.status === "failed" ? "Failed" : "Completed"}
            </span>
            <span>{detail.steps.length} recorded steps</span>
            <span>{Math.round(detail.run.duration_ms)} ms</span>
            <span>{detail.run.split || "not labelled"}</span>
          </div>
        </div>
        <div className="row investigate-actions">
          <Link className="btn" href={`/report/${encodeURIComponent(runId)}`}>Plain-language report</Link>
          <Link className="btn" href={`/compare?a=${encodeURIComponent(runId)}`}>Compare with another run</Link>
          <button className="btn btn-primary" onClick={openFixDrawer} disabled={staticBundle}>Fork and test a fix</button>
        </div>
      </div>

      <div className="investigate-work">
        <section className="canvas-pane">
          <div className="canvas-toolbar">
            {diagnosis ? (
              <>
                <span className="badge badge-fail">
                  Failure surfaced at · {labelStep(diagnosis.visible_failure_addr)}
                </span>
                <span className={`badge ${diagnosis.abstain ? "badge-neutral" : "badge-accent"}`}>
                  {diagnosis.abstain
                    ? `No single likely cause · ${diagnosis.abstain_message || "review the evidence"}`
                    : `Most likely cause · ${labelStep(diagnosis.responsible_addr)} · ${Math.round((diagnosis.ranking[0]?.probability || 0) * 100)}%`}
                </span>
              </>
            ) : (
              <span className="badge badge-pass">Completed successfully · no failure diagnosis</span>
            )}
            <span className="faint flow-hint">Read left to right; select a box to inspect its input and result.</span>
          </div>
          <RunGraph
            layoutKey={runId}
            steps={steps}
            edges={edges}
            selected={selected}
            onSelect={setSelected}
            highlightPath={diagnosis?.damage_path?.nodes.map((node) => node.addr)}
          />
          {status && (
            <div className="floating-card">
              <div className="spread"><strong>Fix comparison</strong><span className="badge badge-neutral">{fork?.verdict || "running"}</span></div>
              <p className="muted">{status}</p>
              {fork?.edited_runs[0] && (
                <Link className="btn btn-sm" href={`/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(fork.edited_runs[0].run_id)}`}>
                  See what changed →
                </Link>
              )}
            </div>
          )}
        </section>

        <aside className="thread-pane">
          <div className="thread-header">
            <div><strong>What happened and why</strong><div className="faint">Select a step below to inspect its inputs, result, and evidence.</div></div>
            <span className="spacer"/><span className="badge badge-neutral">{detail.steps.length} steps</span>
          </div>
          <div className="thread-body">
            {diagnosis ? (
              <>
                {diagnosis.abstain && <div className="notice-box"><strong>No clear cause yet</strong><p>{diagnosis.abstain_message || "The evidence does not point strongly enough to one step. Compare the leading candidates before changing anything."}</p></div>}
                <div className="rail-title">Investigation progress</div>
                <div className="rail">
                  {(["localize", "attribute", "propose", "verify"] as const).map((stage) => (
                    <div key={stage} className={`rail-step ${diagnosis.stages[stage]}`} title={stageLabel(stage)}>
                      {stageLabel(stage)}
                    </div>
                  ))}
                </div>
                {diagnosis.reasons.slice(0, 3).map((reason, index) => (
                  <div className="reason" key={index}>
                    <strong>{reason.text}</strong>
                    <div className="faint">Evidence · {labelStep(reason.citation.addr)} · {reason.feature.group}</div>
                  </div>
                ))}
                {likelySuggestion && (
                  <div className="suggested-fix">
                    <strong>Suggested starting point</strong>
                    <p>{likelySuggestion.rationale}</p>
                    <button className="btn btn-sm" onClick={openFixDrawer}>Review this correction</button>
                  </div>
                )}
              </>
            ) : (
              <p className="muted">This task completed successfully, so there is no failure to diagnose.</p>
            )}
            {selectedPointer && (
              <div className="card card-pad">
                <span className="label">Selected value · {selectedPointer.addr}</span>
                <div className="mono">{selectedPointer.pointer}</div>
                <pre className="json">{JSON.stringify(selectedPointer.value, null, 2)}</pre>
              </div>
            )}
            <div className="timeline-title">Recorded workflow · select a step to see what went in and came out</div>
            <StepThread steps={steps} selected={selected} onSelect={setSelected} view="pretty" onValueClick={setSelectedPointer}/>
          </div>
        </aside>
      </div>

      {editing && (
        <div className="drawer-backdrop" onClick={() => setEditing(false)}>
          <section className="drawer" onClick={(event) => event.stopPropagation()}>
            <div className="drawer-header">
              <div><p className="label">Test a change</p><h2 className="h2">{diagnosis?.abstain ? "Start from the leading candidate" : "Start from the suspected step"}</h2></div>
              <button className="btn btn-sm" onClick={() => setEditing(false)}>Close</button>
            </div>
            <div className="drawer-body stack">
              <p className="muted">{diagnosis?.abstain ? "No single step is certain, so the leading candidate and its suggested correction are filled in." : "The suspected step and suggested correction are filled in for you."} Change the step or edit the value below if you want to test another explanation.</p>
              <label className="label" htmlFor="fork-step">Step to change</label>
              <select id="fork-step" className="input" value={selected || ""} onChange={(event) => applySuggestion(event.target.value)}>
                {detail.steps.map((step) => (
                    <option value={step.addr} key={step.addr}>
                    {labelStep(step.addr)}{step.addr === diagnosis?.visible_failure_addr ? " · failure surfaced here" : ""}{step.addr === likelyCause ? ` · ${diagnosis?.abstain ? "leading candidate" : "most likely cause"}` : ""}
                  </option>
                ))}
              </select>
              <div className="edit-context">
                <strong>{selectedStep?.name || "Selected step"}</strong>
                {proposedEdit(selected || "") ? (
                  <p>{proposedEdit(selected || "")?.rationale}</p>
                ) : (
                  <p>No known correction is available for this step. Edit the value yourself to test a different idea.</p>
                )}
              </div>
              <label className="label" htmlFor="fork-operation">What to change</label>
              <select id="fork-operation" className="input" value={kind} onChange={(event) => changeOperation(event.target.value as ForkEdit["kind"])}>
                <option value="override_output">Replace this step’s result</option>
                <option value="patch_tool_result">Replace a tool’s returned data</option>
                <option value="patch_tool_args">Change the tool’s inputs</option>
                <option value="patch_prompt">Change the model instruction</option>
                <option value="swap_model">Try another model</option>
              </select>
              <label className="label" htmlFor="fork-value">{correctionLabel[kind]}</label>
              <textarea id="fork-value" className="input edit-json" value={value} onChange={(event) => setValue(event.target.value)} spellCheck={false}/>
              {error && <p className="error-box" role="alert">{error}</p>}
              <div className="edit-context"><strong>How the test works</strong><p>Black Box runs five paired comparisons: the original task and your change. It reuses unaffected recorded steps, then shows whether the change improved the final result.</p></div>
            </div>
            <div className="drawer-footer">
              <span className="muted">5 paired tests · unaffected steps reused</span><span className="spacer"/>
              <button className="btn btn-primary" onClick={startFork} disabled={!selectedStep || staticBundle}>Run comparison</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
