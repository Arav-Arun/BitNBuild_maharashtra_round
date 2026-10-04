"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { RunGraph } from "../graph/RunGraph";
import { StepThread } from "../thread/StepThread";
import { DiagnosisDetails } from "../diagnosis/DiagnosisDetails";
import { ProvenanceCard } from "../diagnosis/ProvenanceCard";
import { ForkDrawer } from "../fork/ForkDrawer";
import { ForkResult, type ForkProgress } from "../fork/ForkResult";
import { ForkTimeline } from "../fork/ForkTimeline";
import type { EdgeView, ReplayVisual, StepView, ValueClick } from "../types";
import type { Diagnosis, ForkCreated, ForkRequest, ForkSummary, ForkTimeline as Timeline, RunDetail, ValueProvenance } from "../../lib/contract";
import { API, api, errorText } from "../../lib/api";
import { useAppContext } from "../shell/AppContext";

type Tab = "input" | "output" | "state" | "reasoning" | "raw";

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

function tabFor(pointer: string): Tab {
  if (pointer.startsWith("/input")) return "input";
  if (pointer.startsWith("/state")) return "state";
  return "output";
}

/** Graph state for one replayed step event (edited branch only, so the graph tells one story). */
function visualFor(phase: string, cacheStatus: string | null): ReplayVisual {
  if (phase === "queued") return "queued";
  if (phase === "running") return "live";
  if (phase === "diverged") return "diverged";
  if (cacheStatus === "cached") return "cached";
  if (cacheStatus === "edited") return "edited";
  return "rerun";
}

function parse(event: Event): Record<string, any> | null {
  if (!(event instanceof MessageEvent) || !event.data) return null;
  try {
    return JSON.parse(event.data);
  } catch {
    return null; // a malformed event is skipped; GET /forks/{id} stays the source of truth
  }
}

const EMPTY_PROGRESS: ForkProgress = { k: 0, control: true, fixDone: 0, fixPassed: 0, controlDone: 0, controlPassed: 0, current: "Queued" };

export function InvestigatePage({ runId, startEditing = false }: { runId: string; startEditing?: boolean }) {
  const { staticBundle, mode } = useAppContext();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [editing, setEditing] = useState(startEditing);
  const [forkId, setForkId] = useState("");
  const [fork, setFork] = useState<ForkSummary | null>(null);
  const [progress, setProgress] = useState<ForkProgress>(EMPTY_PROGRESS);
  const [forkError, setForkError] = useState("");
  const [visuals, setVisuals] = useState<Record<string, ReplayVisual>>({});
  const [provenance, setProvenance] = useState<{ data: ValueProvenance | null; loading: boolean; error: string } | null>(null);
  const [highlight, setHighlight] = useState<{ addr: string; pointer: string } | null>(null);
  const [forcedTab, setForcedTab] = useState<{ addr: string; tab: Tab } | null>(null);

  const load = useCallback(async () => {
    try {
      const run = await api<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
      // A replayed run reuses its base run's diagnosis: step addresses are stable across forks.
      const diagnosisRunId = run.run.status === "failed" ? runId : run.run.parent_run_id;
      const [result, forks] = await Promise.all([
        diagnosisRunId ? api<Diagnosis>(`/runs/${encodeURIComponent(diagnosisRunId)}/diagnosis`).catch(() => null) : null,
        api<Timeline>(`/runs/${encodeURIComponent(runId)}/forks`).catch(() => null),
      ]);
      setDetail(run);
      setDiagnosis(result);
      setTimeline(forks);
      setSelected((current) => current ?? (startEditing
        ? result?.responsible_addr || result?.proposed_fixes[0]?.addr || result?.ranking[0]?.addr
        : result?.responsible_addr || result?.visible_failure_addr || result?.ranking[0]?.addr) ?? run.steps[0]?.addr ?? null);
      setError("");
    } catch (caught) {
      setError(errorText(caught));
    }
  }, [runId, startEditing]);

  useEffect(() => {
    load();
  }, [load]);

  // Follow the paired replay: tally outcomes and animate the edited branch on the graph.
  useEffect(() => {
    if (!forkId) return;
    const source = new EventSource(`${API}/forks/${encodeURIComponent(forkId)}/stream`);
    let finished = false;
    const finish = () => {
      finished = true;
      source.close();
      api<ForkSummary>(`/forks/${encodeURIComponent(forkId)}`).then(setFork).catch((caught) => setForkError(errorText(caught)));
      load();
    };
    const onStep = (event: Event) => {
      const data = parse(event);
      if (!data) return;
      setProgress((current) => ({ ...current, current: `Sample ${data.sample + 1} · ${data.branch === "fix" ? "with change" : "unchanged"} · ${data.addr} ${data.phase}` }));
      if (data.branch === "fix") setVisuals((current) => ({ ...current, [data.addr]: visualFor(data.phase, data.cache_status) }));
    };
    const onOutcome = (event: Event) => {
      const data = parse(event);
      if (!data) return;
      setProgress((current) => data.branch === "fix"
        ? { ...current, fixDone: current.fixDone + 1, fixPassed: current.fixPassed + (data.passed ? 1 : 0) }
        : { ...current, controlDone: current.controlDone + 1, controlPassed: current.controlPassed + (data.passed ? 1 : 0) });
    };
    const onError = (event: Event) => {
      // A server-sent `error` event carries the contract error body; a bare Event is a dropped connection.
      const body = parse(event);
      if (body) {
        setForkError(`${body.message}${body.hint ? ` ${body.hint}` : ""}`);
        finish();
      } else if (!finished) {
        api<ForkSummary>(`/forks/${encodeURIComponent(forkId)}`)
          .then((result) => { if (result.status === "complete" || result.status === "error") { setFork(result); finish(); } })
          .catch(() => {});
      }
    };
    source.addEventListener("step", onStep);
    source.addEventListener("outcome", onOutcome);
    source.addEventListener("summary", finish);
    source.addEventListener("error", onError);
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

  // The damage path lists every step; only the root and its symptoms are on the path.
  const damagePath = useMemo(
    () => diagnosis?.damage_path?.nodes.filter((node) => node.tag !== "unaffected").map((node) => node.addr),
    [diagnosis],
  );

  const stepName = useCallback((addr: string | null) => {
    const step = detail?.steps.find((item) => item.addr === addr);
    return step ? `${step.name} (step ${step.seq + 1})` : addr || "an unknown step";
  }, [detail]);

  const labelStep = useCallback((addr: string | null) => {
    const step = detail?.steps.find((item) => item.addr === addr);
    return step ? `Step ${step.seq + 1}: ${step.name}` : addr || "not isolated";
  }, [detail]);

  const traceValue = useCallback((click: ValueClick) => {
    setProvenance({ data: null, loading: true, error: "" });
    const query = new URLSearchParams({ addr: click.addr, pointer: click.pointer });
    api<ValueProvenance>(`/runs/${encodeURIComponent(runId)}/provenance?${query}`)
      .then((data) => setProvenance({ data, loading: false, error: "" }))
      .catch((caught) => setProvenance({ data: null, loading: false, error: errorText(caught) }));
  }, [runId]);

  const jumpTo = useCallback((addr: string, pointer: string | null) => {
    setSelected(addr);
    setHighlight(pointer ? { addr, pointer } : null);
    setForcedTab(pointer ? { addr, tab: tabFor(pointer) } : null);
  }, []);

  if (error && !detail) {
    return <div className="page"><div className="page-inner"><p className="error-box">{error}</p></div></div>;
  }
  if (!detail) {
    return <div className="page"><div className="page-inner"><p className="muted">Loading recorded task…</p></div></div>;
  }

  const originalTask =
    detail.run.agent === "tripcrew"
      ? taskPromptFromInput(detail.steps[0]?.input) || detail.task_text || detail.run.task
      : detail.task_text || detail.run.task;
  const likelyCause = diagnosis?.responsible_addr || diagnosis?.ranking[0]?.addr;
  const likelySuggestion = diagnosis?.proposed_fixes.find((fix) => fix.addr === likelyCause);
  const staticReason = staticBundle ? "This is the static recorded showcase. Start the local API to replay changes." : null;
  const verifyReason = staticReason || (mode === "recorded"
    ? "The API runs in RECORDED mode, which refuses to re-run model steps. Set MODE=offline in .env and restart make dev-api to test fixes."
    : null);

  function openFixDrawer() {
    setSelected(
      diagnosis?.responsible_addr ||
      diagnosis?.proposed_fixes[0]?.addr ||
      diagnosis?.visible_failure_addr ||
      diagnosis?.ranking[0]?.addr ||
      selected ||
      detail?.steps[0]?.addr ||
      null,
    );
    setEditing(true);
  }

  function started(created: ForkCreated, request: ForkRequest) {
    setEditing(false);
    setFork(null);
    setForkError("");
    setVisuals({});
    setProgress({ ...EMPTY_PROGRESS, k: request.samples ?? 5, control: request.control ?? true });
    setForkId(created.fork_id);
  }

  return (
    <div className="page investigate-page">
      <div className="investigate-top">
        <div className="investigate-heading">
          <h1 className="h1">{detail.run.task}</h1>
          <p className="task-statement"><strong>What the agent was asked:</strong> {originalTask}</p>
          <div className="row muted investigate-meta">
            <span className={`badge ${detail.run.status === "failed" ? "badge-fail" : "badge-pass"}`}>
              {detail.run.status === "failed" ? "Failed" : "Completed"}
            </span>
            {detail.run.checker_reason && detail.run.status === "failed" && <span>{detail.run.checker_reason}</span>}
            <span>{detail.run.origin === "fork" ? "Replayed" : "Recorded"} {detail.run.agent} run, {detail.steps.length} steps, {Math.round(detail.run.duration_ms)} ms</span>
          </div>
        </div>
        <div className="row investigate-actions">
          {detail.run.status === "failed" && <Link className="btn" href={`/report/${encodeURIComponent(runId)}`}>Plain-language report</Link>}
          <Link className="btn" href={`/compare?a=${encodeURIComponent(runId)}`}>Compare with another run</Link>
          <button className="btn btn-primary" onClick={openFixDrawer} disabled={staticBundle} title={staticReason || undefined}>Fork and test a fix</button>
        </div>
      </div>

      <div className="investigate-work">
        <section className="canvas-pane">
          <div className="canvas-toolbar">
            {diagnosis ? (
              <p className="canvas-summary">
                It failed at <strong>{stepName(diagnosis.visible_failure_addr)}</strong>.{" "}
                {diagnosis.abstain ? "Leading candidate" : "Most likely cause"}:{" "}
                <strong className="cause">{stepName(diagnosis.ranking[0]?.addr ?? null)}</strong>{" "}
                <span className="faint">({Math.round((diagnosis.ranking[0]?.probability || 0) * 100)}%)</span>
              </p>
            ) : (
              <p className="canvas-summary">{detail.run.status === "passed" ? "This run completed successfully." : "No diagnosis is available for this run."}</p>
            )}
          </div>
          <RunGraph
            layoutKey={detail.run.parent_run_id || runId}
            steps={steps}
            edges={edges}
            selected={selected}
            onSelect={setSelected}
            visuals={visuals}
            highlightPath={Object.keys(visuals).length ? undefined : damagePath}
          />
          {forkId && (
            <ForkResult
              baseRunId={runId}
              progress={progress}
              summary={fork}
              error={forkError}
              onClose={() => { setForkId(""); setFork(null); setVisuals({}); }}
            />
          )}
        </section>

        <aside className="thread-pane">
          <div className="thread-header">
            <div><strong>What happened and why</strong><div className="faint">Click any value in a step to see where it came from.</div></div>
          </div>
          <div className="thread-body">
            {diagnosis ? (
              <>
                {diagnosis.abstain && <div className="notice-box"><strong>No single clear cause</strong><p>{diagnosis.abstain_message || "The evidence does not point strongly enough to one step."} The leading candidate is highlighted; compare the candidates before changing anything.</p></div>}
                <div className="rail-title">Progress</div>
                <div className="rail">
                  {(["localize", "attribute", "propose", "verify"] as const).map((stage) => (
                    <div key={stage} className={`rail-step ${diagnosis.stages[stage]}`} title={stageLabel(stage)}>
                      {stageLabel(stage)}
                    </div>
                  ))}
                </div>
                {diagnosis.reasons.slice(0, 3).map((reason, index) => (
                  <button type="button" className="reason" key={index} onClick={() => jumpTo(reason.citation.addr, reason.citation.json_pointer)}>
                    <strong>{reason.text}</strong>
                    <div className="faint">{labelStep(reason.citation.addr)}</div>
                  </button>
                ))}
                {likelySuggestion && (
                  <div className="suggested-fix">
                    <strong>Suggested starting point</strong>
                    <p>{likelySuggestion.rationale}</p>
                    <button className="btn btn-sm" onClick={openFixDrawer} disabled={staticBundle}>Review this correction</button>
                  </div>
                )}
                <DiagnosisDetails
                  runId={diagnosis.run_id}
                  diagnosis={diagnosis}
                  labelStep={labelStep}
                  onSelect={(addr) => jumpTo(addr, null)}
                  replayBlocked={verifyReason}
                  onVerified={load}
                />
              </>
            ) : (
              <p className="muted">{detail.run.status === "passed" ? "This task completed successfully, so there is no failure to diagnose. You can still fork it to test a change." : "The diagnoser is unavailable for this run."}</p>
            )}
            <ForkTimeline runId={runId} timeline={timeline} />
            {provenance && (
              <ProvenanceCard
                provenance={provenance.data}
                loading={provenance.loading}
                error={provenance.error}
                labelStep={labelStep}
                onJump={jumpTo}
                onClose={() => { setProvenance(null); setHighlight(null); }}
              />
            )}
            <div className="timeline-title">The run, step by step</div>
            <StepThread steps={steps} selected={selected} onSelect={setSelected} view="pretty" onValueClick={traceValue} highlight={highlight} forcedTab={forcedTab}/>
          </div>
        </aside>
      </div>

      {editing && (
        <ForkDrawer
          detail={detail}
          diagnosis={diagnosis}
          selected={selected}
          onSelect={setSelected}
          labelStep={labelStep}
          disabledReason={staticReason}
          onClose={() => setEditing(false)}
          onStarted={started}
        />
      )}
    </div>
  );
}
