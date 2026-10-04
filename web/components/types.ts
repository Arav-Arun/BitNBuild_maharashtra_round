// View models for presentational components. Pages map API contract types into these,
// so components stay stable if the wire format gains fields.

export type StepKind = "llm" | "tool" | "retrieval" | "state" | "value" | "final";

export type ReplayVisual =
  | "idle"
  | "queued"
  | "cached"
  | "invalidated"
  | "live"
  | "edited"
  | "rerun"
  | "pass"
  | "fail"
  | "diverged";

export interface Violation {
  pointer: string;
  message: string;
}

export interface StepView {
  addr: string;
  seq: number;
  kind: StepKind;
  name: string;
  role?: string | null;
  input: unknown;
  output: unknown;
  stateBefore?: Record<string, unknown>;
  stateAfter?: Record<string, unknown>;
  reasoning?: string | null;
  latencyMs?: number | null;
  tokens?: number | null;
  error?: string | null;
  cacheStatus?: string | null;
  suspicion?: number | null;
  isSuspect?: boolean;
  isVisibleFailure?: boolean;
  violations?: Violation[];
}

export interface EdgeView {
  id: string;
  source: string;
  target: string;
  kind: string;
  label?: string | null;
}

export interface ValueClick {
  addr: string;
  pointer: string;
  value: unknown;
}
