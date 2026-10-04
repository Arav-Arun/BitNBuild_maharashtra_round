"use client";

import Link from "next/link";
import type { ForkTimeline as Timeline } from "../../lib/contract";
import { VerdictBadge } from "../ui/badges";

/** Earlier fix experiments on this run: each edited fork with its verdict and pass rate. */
export function ForkTimeline({ runId, timeline }: { runId: string; timeline: Timeline | null }) {
  const forks = timeline?.entries.filter((entry) => entry.kind === "fork") ?? [];
  if (forks.length === 0) return null;
  return (
    <section className="detail-block">
      <div className="rail-title">Fixes tried on this run · {forks.length}</div>
      <ul className="timeline-list">
        {[...forks].reverse().slice(0, 6).map((entry) => (
          <li key={entry.fork_id}>
            <div className="spread">
              <span className="mono faint timeline-edit" title={entry.edit_summary || undefined}>{entry.edit_summary || entry.label}</span>
              <VerdictBadge verdict={entry.verdict} preview={entry.preview} />
            </div>
            <div className="row faint">
              {entry.pass_rate && <span>{entry.pass_rate.passed}/{entry.pass_rate.total} passed</span>}
              {entry.run_id && <Link className="btn-link" href={`/compare?a=${encodeURIComponent(runId)}&b=${encodeURIComponent(entry.run_id)}`}>Compare</Link>}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
