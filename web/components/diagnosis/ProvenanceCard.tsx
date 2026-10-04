"use client";

import type { ProducerCandidate, ValueProvenance } from "../../lib/contract";

interface ProvenanceCardProps {
  provenance: ValueProvenance | null;
  loading: boolean;
  error: string;
  labelStep: (addr: string | null) => string;
  /** Select a step and highlight one of its fields. */
  onJump: (addr: string, pointer: string | null) => void;
  onClose: () => void;
}

function preview(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text && text.length > 80 ? `${text.slice(0, 79)}…` : text ?? "null";
}

function Origin({ candidate, labelStep, onJump }: {
  candidate: ProducerCandidate;
  labelStep: (addr: string | null) => string;
  onJump: ProvenanceCardProps["onJump"];
}) {
  return (
    <button type="button" className="prov-link" onClick={() => onJump(candidate.addr, candidate.json_pointer)}>
      <span>{labelStep(candidate.addr)}</span>
      <span className="mono faint">{candidate.json_pointer || "/"}{candidate.key ? ` · state "${candidate.key}"` : ""}</span>
    </button>
  );
}

/** Where a clicked value came from (producer) and where it went (consumers). */
export function ProvenanceCard({ provenance, loading, error, labelStep, onJump, onClose }: ProvenanceCardProps) {
  return (
    <section className="prov-card" aria-live="polite">
      <div className="spread">
        <span className="label">Value origin</span>
        <button type="button" className="btn btn-sm btn-ghost" onClick={onClose} aria-label="Close value origin">Close</button>
      </div>
      {loading && <p className="muted">Tracing this value…</p>}
      {error && <p className="error-box">{error}</p>}
      {provenance && (
        <>
          <p className="prov-target">
            <code>{preview(provenance.target.value)}</code>
            <span className="faint"> at {labelStep(provenance.target.addr)} · <span className="mono">{provenance.target.json_pointer}</span></span>
          </p>
          {provenance.producer ? (
            <div className="prov-section">
              <span className="faint">Came from</span>
              <Origin candidate={provenance.producer} labelStep={labelStep} onJump={onJump} />
            </div>
          ) : provenance.ambiguous ? (
            <div className="prov-section">
              <span className="faint">Several earlier steps produced this exact value. Pick the one to inspect:</span>
              {provenance.candidates.map((candidate) => (
                <Origin key={`${candidate.addr}${candidate.json_pointer}`} candidate={candidate} labelStep={labelStep} onJump={onJump} />
              ))}
            </div>
          ) : (
            <p className="faint prov-section">No earlier step produced this value: it originates in this step.</p>
          )}
          {provenance.consumers.length > 0 && (
            <div className="prov-section">
              <span className="faint">
                Flows into {provenance.consumers.length} later {provenance.consumers.length === 1 ? "use" : "uses"}
                {provenance.reaches_final_answer ? ", including the final answer" : ""}
              </span>
              <div className="prov-chips">
                {[...new Map(provenance.consumers.map((hop) => [hop.dst_addr, hop])).values()].slice(0, 8).map((hop) => (
                  <button type="button" key={hop.dst_addr} className="badge badge-outline" onClick={() => onJump(hop.dst_addr, null)}>
                    {labelStep(hop.dst_addr)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
