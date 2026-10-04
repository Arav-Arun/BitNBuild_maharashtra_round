"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, errorText } from "../../lib/api";
import type { Diagnosis, VerifyJob } from "../../lib/contract";
import { VerdictBadge } from "../ui/badges";
import { PassRateText } from "../fork/PassRateText";

interface DiagnosisDetailsProps {
  runId: string;
  diagnosis: Diagnosis;
  labelStep: (addr: string | null) => string;
  onSelect: (addr: string) => void;
  /** Why replays cannot run here (static bundle, recorded mode), or null. */
  replayBlocked: string | null;
  /** Called when a verification job finishes, so the page can refresh its diagnosis. */
  onVerified: () => void;
}

/** Evidence the diagnoser computes beyond the top reasons: candidates, precedents, twin, proof. */
export function DiagnosisDetails({ runId, diagnosis, labelStep, onSelect, replayBlocked, onVerified }: DiagnosisDetailsProps) {
  const [job, setJob] = useState<VerifyJob | null>(null);
  const [error, setError] = useState("");
  const running = job !== null && (job.status === "queued" || job.status === "running");

  useEffect(() => {
    if (!job || !running) return;
    const timer = window.setTimeout(() => {
      api<VerifyJob>(`/jobs/${encodeURIComponent(job.job_id)}`)
        .then((next) => {
          setJob(next);
          if (next.status === "complete" || next.status === "failed") onVerified();
        })
        .catch((caught) => setError(errorText(caught)));
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [job, running, onVerified]);

  async function verify() {
    setError("");
    try {
      setJob(await api<VerifyJob>(`/runs/${encodeURIComponent(runId)}/verify`, {
        method: "POST",
        body: JSON.stringify({ samples: 5 }),
      }));
    } catch (caught) {
      setError(errorText(caught));
    }
  }

  const candidates = diagnosis.conformal_set.filter((addr) => addr !== diagnosis.responsible_addr);
  const verification = diagnosis.verification;

  return (
    <div className="diagnosis-details">
      {diagnosis.abstain && candidates.length > 0 && (
        <section className="detail-block">
          <div className="rail-title">Could be any of</div>
          <div className="prov-chips">
            {candidates.slice(0, 6).map((addr) => (
              <button type="button" key={addr} className="badge badge-outline" onClick={() => onSelect(addr)}>
                {labelStep(addr)} <span className="faint">{Math.round((diagnosis.ranking.find((s) => s.addr === addr)?.probability || 0) * 100)}%</span>
              </button>
            ))}
            {candidates.length > 6 && <span className="faint">+{candidates.length - 6} more</span>}
          </div>
        </section>
      )}

      {verification && verification.status !== "not_started" && (
        <section className="detail-block">
          <div className="rail-title">Proof</div>
          <div className="row"><VerdictBadge verdict={verification.verdict} preview={verification.preview} />
            {verification.edit_addr && <button type="button" className="btn-link" onClick={() => onSelect(verification.edit_addr!)}>{labelStep(verification.edit_addr)}</button>}
          </div>
          {verification.fix && <p className="faint">With the change <PassRateText rate={verification.fix} />{verification.control && <> · unchanged <PassRateText rate={verification.control} /></>}</p>}
          {verification.explanation && <p className="muted">{verification.explanation}</p>}
        </section>
      )}

      <section className="detail-block">
        <div className="rail-title">Test the top candidates automatically</div>
        <p className="faint">Replays the suggested correction for up to three candidates, each paired with an unchanged control (5 samples).</p>
        <button type="button" className="btn btn-sm" onClick={verify} disabled={running || Boolean(replayBlocked) || diagnosis.proposed_fixes.length === 0}
          title={replayBlocked || (diagnosis.proposed_fixes.length === 0 ? "No correction is proposed for this run." : undefined)}>
          {running ? "Testing candidates…" : "Verify candidates"}
        </button>
        {replayBlocked && <p className="faint">{replayBlocked}</p>}
        {error && <p className="error-box">{error}</p>}
        {job && (
          <ul className="verify-list">
            {job.candidates.map((candidate) => (
              <li key={candidate.addr}>
                <button type="button" className="btn-link" onClick={() => onSelect(candidate.addr)}>{labelStep(candidate.addr)}</button>
                {candidate.verdict ? <VerdictBadge verdict={candidate.verdict} /> : <span className="badge badge-neutral">{candidate.status}</span>}
                {candidate.note && <p className="faint">{candidate.note}</p>}
              </li>
            ))}
          </ul>
        )}
        {job?.status === "complete" && <p className="muted">{job.best_addr ? `Best verified fix: ${labelStep(job.best_addr)}.` : "No candidate was verified; compare the evidence or write a correction by hand."}</p>}
      </section>

      {diagnosis.precedents && (
        <section className="detail-block">
          <div className="rail-title">Seen before</div>
          <p className="muted">{diagnosis.precedents.summary}</p>
        </section>
      )}

      {diagnosis.nearest_twin && (
        <section className="detail-block">
          <div className="rail-title">Nearest passing run</div>
          <p className="faint">
            {diagnosis.nearest_twin.run.task}, {Math.round(diagnosis.nearest_twin.similarity * 100)}% similar{diagnosis.nearest_twin.same_task ? ", same task" : ""}
          </p>
          <Link className="btn btn-sm" href={`/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(diagnosis.nearest_twin.run.run_id)}`}>
            Compare with it
          </Link>
        </section>
      )}

      {diagnosis.narrative && (
        <section className="detail-block">
          <div className="rail-title">Summary</div>
          <p className="muted">{diagnosis.narrative.summary}</p>
        </section>
      )}
    </div>
  );
}
