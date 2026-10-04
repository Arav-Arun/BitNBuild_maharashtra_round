"use client";

import { useEffect, useRef, useState } from "react";

import { JsonView } from "../json/JsonView";
import { prettyName } from "../graph/RunGraph";
import { CacheBadge, FailureBadge, KindAvatar, SuspectBadge, normaliseKind } from "../ui/badges";
import type { StepView, ValueClick } from "../types";

type Tab = "input" | "output" | "state" | "reasoning" | "raw";

interface StepThreadProps {
  steps: StepView[];
  selected: string | null;
  onSelect: (addr: string) => void;
  view: "pretty" | "json";
  highlight?: { addr: string; pointer: string } | null;
  onValueClick?: (click: ValueClick) => void;
  forcedTab?: { addr: string; tab: Tab } | null;
}

export function StepThread({
  steps,
  selected,
  onSelect,
  view,
  highlight,
  onValueClick,
  forcedTab,
}: StepThreadProps) {
  const refs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    if (selected) refs.current[selected]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  return (
    <div className="thread">
      {steps.map((step) => (
        <div
          key={step.addr}
          className="thread-item"
          ref={(el) => {
            refs.current[step.addr] = el;
          }}
        >
          <KindAvatar kind={step.kind} />
          <div
            className={`thread-card${selected === step.addr ? " is-selected" : ""}${
              step.isSuspect ? " is-suspect" : ""
            }`}
            onClick={() => onSelect(step.addr)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSelect(step.addr);
            }}
            aria-expanded={selected === step.addr}
          >
            <div className="thread-title">
              <span className="faint num" style={{ fontSize: 11 }}>
                {String(step.seq + 1).padStart(2, "0")}
              </span>
              <strong>{prettyName(step)}</strong>
              {step.isSuspect && <SuspectBadge probability={step.suspicion} />}
              {step.isVisibleFailure && <FailureBadge />}
              {step.error && <span className="badge badge-fail">error</span>}
              {step.cacheStatus && step.cacheStatus !== "live" && <CacheBadge status={step.cacheStatus} />}
              <span className="faint num" style={{ fontSize: 11, marginLeft: "auto" }}>
                {step.latencyMs != null ? `${Math.round(step.latencyMs)} ms` : ""}
              </span>
            </div>
            {selected === step.addr ? (
              <div onClick={(e) => e.stopPropagation()}>
                <StepInspector
                  step={step}
                  view={view}
                  highlight={highlight && highlight.addr === step.addr ? highlight.pointer : null}
                  onValueClick={onValueClick}
                  forcedTab={forcedTab && forcedTab.addr === step.addr ? forcedTab.tab : null}
                />
              </div>
            ) : (
              <StepPreview step={step} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function StepPreview({ step }: { step: StepView }) {
  const kind = normaliseKind(step.kind);
  if (kind === "tool" || kind === "retrieval") {
    const args = scalarEntries(step.input, 4);
    if (args.length > 0) {
      return (
        <table className="kv" style={{ marginTop: 8 }}>
          <tbody>
            {args.map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{formatScalar(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      );
    }
  }
  const text = summarise(step.output);
  return text ? <div className="preview">{text}</div> : null;
}

function StepInspector({
  step,
  view,
  highlight,
  onValueClick,
  forcedTab,
}: {
  step: StepView;
  view: "pretty" | "json";
  highlight: string | null;
  onValueClick?: (click: ValueClick) => void;
  forcedTab: Tab | null;
}) {
  const [tab, setTab] = useState<Tab>("output");
  useEffect(() => {
    if (forcedTab) setTab(forcedTab);
  }, [forcedTab]);

  const tabs: [Tab, string][] = [
    ["input", "Input"],
    ["output", "Output"],
    ["state", "State"],
    ["reasoning", "Reasoning"],
    ["raw", "Raw"],
  ];

  const click = (section: string) => (pointer: string, value: unknown) =>
    onValueClick?.({ addr: step.addr, pointer: `/${section}${pointer}`, value });

  const sectionHighlight = (section: string) =>
    highlight && highlight.startsWith(`/${section}`) ? highlight.slice(section.length + 1) : null;

  return (
    <div>
      {step.violations && step.violations.length > 0 && (
        <div className="stack" style={{ gap: 4, marginTop: 8 }}>
          {step.violations.map((v) => (
            <div key={v.pointer + v.message} className="error-box" style={{ padding: "6px 10px" }}>
              <span className="mono">{v.pointer || "/"}</span> · {v.message}
            </div>
          ))}
        </div>
      )}
      <div className="tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>
      {tab === "input" && (
        <Section
          value={step.input}
          view={view}
          highlight={sectionHighlight("input")}
          onValueClick={click("input")}
        />
      )}
      {tab === "output" && (
        <Section
          value={step.output}
          view={view}
          highlight={sectionHighlight("output")}
          onValueClick={click("output")}
        />
      )}
      {tab === "state" && <StateDiff before={step.stateBefore} after={step.stateAfter} />}
      {tab === "reasoning" &&
        (step.reasoning ? (
          <div className="json">{step.reasoning}</div>
        ) : (
          <div className="preview faint">No reasoning text was recorded for this step.</div>
        ))}
      {tab === "raw" && <JsonView value={rawStep(step)} />}
      <div className="row faint" style={{ fontSize: 11.5, marginTop: 8, flexWrap: "wrap", gap: 14 }}>
        <span className="mono">{step.addr}</span>
        {step.role && <span>role {step.role}</span>}
        {step.tokens != null && <span>{step.tokens} tokens</span>}
        {onValueClick && <span>Click any value to trace where it came from</span>}
      </div>
    </div>
  );
}

function Section({
  value,
  view,
  highlight,
  onValueClick,
}: {
  value: unknown;
  view: "pretty" | "json";
  highlight: string | null;
  onValueClick: (pointer: string, value: unknown) => void;
}) {
  if (value === undefined || value === null) {
    return <div className="preview faint">Nothing recorded.</div>;
  }
  if (view === "pretty" && typeof value === "string") {
    return <div className="json">{value}</div>;
  }
  return <JsonView value={value} highlight={highlight} onValueClick={onValueClick} />;
}

function StateDiff({
  before,
  after,
}: {
  before?: Record<string, unknown>;
  after?: Record<string, unknown>;
}) {
  const keys = [...new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})])].sort();
  const changed = keys.filter(
    (k) => JSON.stringify((before ?? {})[k]) !== JSON.stringify((after ?? {})[k]),
  );
  if (keys.length === 0) return <div className="preview faint">This step did not touch shared state.</div>;
  return (
    <div className="stack" style={{ gap: 6, marginTop: 8 }}>
      <div className="faint" style={{ fontSize: 12 }}>
        {changed.length} of {keys.length} keys changed by this step
      </div>
      <table className="kv" style={{ width: "100%" }}>
        <tbody>
          {keys.map((k) => {
            const isChanged = changed.includes(k);
            return (
              <tr key={k}>
                <td style={{ color: isChanged ? "var(--accent)" : undefined }}>{k}</td>
                <td>
                  {isChanged && before && k in before && (
                    <div style={{ color: "var(--fail)", textDecoration: "line-through", opacity: 0.8 }}>
                      {short(before[k])}
                    </div>
                  )}
                  <div style={{ color: isChanged ? "var(--pass)" : "var(--text-2)" }}>
                    {after && k in after ? short(after[k]) : <span className="faint">removed</span>}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function rawStep(step: StepView): Record<string, unknown> {
  return {
    addr: step.addr,
    seq: step.seq,
    kind: step.kind,
    name: step.name,
    input: step.input,
    output: step.output,
    state_before: step.stateBefore,
    state_after: step.stateAfter,
    latency_ms: step.latencyMs,
    tokens: step.tokens,
    error: step.error,
    cache_status: step.cacheStatus,
  };
}

function scalarEntries(value: unknown, limit: number): [string, unknown][] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  const source =
    "args" in (value as Record<string, unknown>) &&
    typeof (value as Record<string, unknown>).args === "object"
      ? ((value as Record<string, unknown>).args as Record<string, unknown>)
      : (value as Record<string, unknown>);
  return Object.entries(source ?? {})
    .filter(([, v]) => v === null || typeof v !== "object")
    .slice(0, limit);
}

export function summarise(value: unknown, max = 220): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return truncate(value, max);
  if (typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    for (const key of ["content", "text", "answer", "summary", "message"]) {
      if (typeof obj[key] === "string") return truncate(obj[key] as string, max);
    }
  }
  return truncate(JSON.stringify(value), max);
}

function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

function short(value: unknown): string {
  return truncate(typeof value === "string" ? value : JSON.stringify(value), 160);
}

function formatScalar(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}
