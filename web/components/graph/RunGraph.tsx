"use client";

import "@xyflow/react/dist/style.css";

import {
  Background,
  BackgroundVariant,
  Controls,
  type Edge,
  Handle,
  MarkerType,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import ELK from "elkjs/lib/elk.bundled.js";
import { memo, useEffect, useMemo, useState } from "react";

import { normaliseKind } from "../ui/badges";
import type { EdgeView, ReplayVisual, StepView } from "../types";

const NODE_W = 196;
const NODE_H = 62;

const elk = new ELK();
// Coordinates are computed once per base run and reused by every fork of it, so replay
// animations never move nodes around.
const layoutCache = new Map<string, Record<string, { x: number; y: number }>>();

async function computeLayout(
  key: string,
  steps: StepView[],
  edges: EdgeView[],
): Promise<Record<string, { x: number; y: number }>> {
  const cached = layoutCache.get(key);
  if (cached && steps.every((s) => cached[s.addr])) return cached;
  const ids = new Set(steps.map((s) => s.addr));
  const graph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.layered.spacing.nodeNodeBetweenLayers": "56",
      "elk.spacing.nodeNode": "36",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.crossingMinimization.semiInteractive": "true",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    },
    children: [...steps]
      .sort((a, b) => a.seq - b.seq)
      .map((s) => ({ id: s.addr, width: NODE_W, height: NODE_H })),
    edges: uniqueEdges(edges)
      .filter((e) => ids.has(e.source) && ids.has(e.target) && e.source !== e.target)
      .map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] })),
  };
  const result = await elk.layout(graph);
  const positions: Record<string, { x: number; y: number }> = {};
  for (const child of result.children ?? []) {
    positions[child.id] = { x: child.x ?? 0, y: child.y ?? 0 };
  }
  layoutCache.set(key, positions);
  return positions;
}

function uniqueEdges(edges: EdgeView[]): EdgeView[] {
  const seen = new Map<string, EdgeView & { count: number }>();
  for (const e of edges) {
    const k = `${e.source}->${e.target}`;
    const existing = seen.get(k);
    if (existing) existing.count += 1;
    else seen.set(k, { ...e, id: k, count: 1 });
  }
  return [...seen.values()];
}

interface StepNodeData extends Record<string, unknown> {
  step: StepView;
  visual: ReplayVisual;
  selected: boolean;
  dim: boolean;
  onPath: boolean;
}

const StepNode = memo(function StepNode({ data }: NodeProps<Node<StepNodeData>>) {
  const { step, visual, selected, dim, onPath } = data;
  const kind = normaliseKind(step.kind);
  const classes = [
    "node",
    `k-${kind}`,
    step.isSuspect ? "is-suspect" : "",
    step.isVisibleFailure ? "is-failure" : "",
    selected ? "is-selected" : "",
    dim && !onPath && !selected ? "is-dim" : "",
    visual !== "idle" ? `st-${visual}` : "",
  ]
    .filter(Boolean)
    .join(" ");
  const heat = Math.max(0, Math.min(1, step.suspicion ?? 0));
  return (
    <div className={classes} title={step.addr}>
      <Handle type="target" position={Position.Top} />
      <div className="spread" style={{ gap: 8 }}>
        <div className="node-title">{prettyName(step)}</div>
        {visual !== "idle" && visual !== "queued" && <VisualTag visual={visual} />}
      </div>
      <div className="node-sub">{step.addr}</div>
      <div className="node-heat" aria-label={`suspicion ${Math.round(heat * 100)}%`}>
        <span style={{ width: `${heat * 100}%`, opacity: heat > 0 ? 1 : 0 }} />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

function VisualTag({ visual }: { visual: ReplayVisual }) {
  const map: Record<string, [string, string]> = {
    cached: ["cached", "var(--text-3)"],
    invalidated: ["invalidated", "var(--warn)"],
    live: ["running", "var(--warn)"],
    edited: ["edited", "var(--accent)"],
    pass: ["done", "var(--pass)"],
    fail: ["done", "var(--fail)"],
    diverged: ["diverged", "var(--fail)"],
  };
  const [label, color] = map[visual] ?? [visual, "var(--text-3)"];
  return (
    <span className="mono" style={{ color, fontSize: 10.5 }}>
      {label}
    </span>
  );
}

export function prettyName(step: StepView): string {
  const [role] = step.addr.split("/");
  const name = step.name || step.addr;
  if (role && !name.toLowerCase().startsWith(role.toLowerCase())) {
    return `${role} · ${name}`;
  }
  return name;
}

const nodeTypes = { step: StepNode };

export interface RunGraphProps {
  layoutKey: string;
  steps: StepView[];
  edges: EdgeView[];
  selected?: string | null;
  onSelect?: (addr: string) => void;
  visuals?: Record<string, ReplayVisual>;
  highlightPath?: string[];
}

export function RunGraph(props: RunGraphProps) {
  return (
    <ReactFlowProvider>
      <GraphInner {...props} />
    </ReactFlowProvider>
  );
}

function GraphInner({
  layoutKey,
  steps,
  edges,
  selected,
  onSelect,
  visuals,
  highlightPath,
}: RunGraphProps) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }> | null>(
    () => layoutCache.get(layoutKey) ?? null,
  );
  const { fitView } = useReactFlow();

  useEffect(() => {
    let cancelled = false;
    computeLayout(layoutKey, steps, edges).then((p) => {
      if (!cancelled) setPositions(p);
    });
    return () => {
      cancelled = true;
    };
  }, [layoutKey, steps, edges]);

  useEffect(() => {
    if (positions) requestAnimationFrame(() => fitView({ padding: 0.18, duration: 0 }));
  }, [positions, fitView]);

  const pathSet = useMemo(() => new Set(highlightPath ?? []), [highlightPath]);
  const dim = pathSet.size > 0;

  const nodes: Node<StepNodeData>[] = useMemo(() => {
    if (!positions) return [];
    return steps.map((step) => ({
      id: step.addr,
      type: "step",
      position: positions[step.addr] ?? { x: 0, y: 0 },
      data: {
        step,
        visual: visuals?.[step.addr] ?? "idle",
        selected: selected === step.addr,
        dim,
        onPath: pathSet.has(step.addr),
      },
      draggable: false,
    }));
  }, [positions, steps, visuals, selected, dim, pathSet]);

  const rfEdges: Edge[] = useMemo(() => {
    return uniqueEdges(edges).map((e) => {
      const onPath = pathSet.has(e.source) && pathSet.has(e.target);
      const color = onPath ? "var(--accent)" : e.kind === "message" ? "#3d434d" : "#525967";
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        type: "default",
        animated: onPath,
        style: {
          stroke: color,
          strokeWidth: onPath ? 2 : 1.4,
          strokeDasharray: e.kind === "inferred" ? "5 4" : undefined,
          opacity: dim && !onPath ? 0.35 : 1,
        },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      };
    });
  }, [edges, pathSet, dim]);

  if (!positions) {
    return (
      <div className="empty" style={{ height: "100%" }}>
        <div className="skeleton" style={{ height: 14, width: 180 }} />
        <span className="faint">Laying out the execution graph…</span>
      </div>
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={rfEdges}
      nodeTypes={nodeTypes}
      onNodeClick={(_, node) => onSelect?.(node.id)}
      nodesConnectable={false}
      nodesDraggable={false}
      elementsSelectable
      minZoom={0.2}
      maxZoom={1.6}
      proOptions={{ hideAttribution: true }}
      colorMode="dark"
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#262a31" />
      <Controls showInteractive={false} position="bottom-right" />
    </ReactFlow>
  );
}
