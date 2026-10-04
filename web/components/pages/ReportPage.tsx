"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { API, ApiRequestError, STATIC_BUILD, api, errorText } from "../../lib/api";
import type { CrashReport, ReportClaim } from "../../lib/contract";

function Claim({ claim }: { claim: ReportClaim }) {
  return (
    <p>
      {claim.text}
      {claim.cites.map((cite) => <code key={`${cite.addr}${cite.json_pointer ?? ""}`} className="cite">{cite.addr}{cite.json_pointer ?? ""}</code>)}
    </p>
  );
}

export function ReportPage({ runId }: { runId: string }) {
  const [data, setData] = useState<CrashReport | null>(null);
  const [err, setErr] = useState("");
  const [notApplicable, setNotApplicable] = useState(false);

  useEffect(() => {
    api<CrashReport>(`/runs/${encodeURIComponent(runId)}/report`)
      .then(setData)
      .catch((caught) => {
        setNotApplicable(caught instanceof ApiRequestError && caught.code === "not_applicable");
        setErr(errorText(caught));
      });
  }, [runId]);

  return (
    <div className="page">
      <article className="page-inner report">
        <div className="spread">
          <div>
            <h1 className="h1">{data?.title.replace(/^Failure report · /, "") || "Failure report"}</h1>
            <p className="faint report-run">Failure report for run <Link className="btn-link" href={`/investigate/${encodeURIComponent(runId)}`}>{runId}</Link></p>
          </div>
          {data && !STATIC_BUILD && <a className="btn" href={`${API}/runs/${encodeURIComponent(runId)}/report.md`}>Download Markdown</a>}
        </div>
        {err && <p className={notApplicable ? "notice-box" : "error-box"}>{notApplicable ? "This run completed successfully, so there is no failure to report." : err}</p>}
        {data && (
          <>
            <p className="report-synopsis">{data.synopsis}</p>
            {data.probable_cause && (
              <section className="card card-pad">
                <p className="label">Probable cause</p>
                <h2 className="h2">{data.probable_cause.text}</h2>
              </section>
            )}
            {data.recommended_fix && (
              <section className="card card-pad">
                <p className="label">Recommended fix</p>
                <Claim claim={data.recommended_fix} />
              </section>
            )}
            {data.verification && (
              <section className="card card-pad">
                <p className="label">Verification</p>
                <Claim claim={data.verification} />
              </section>
            )}
            {data.contributing_factors.length > 0 && (
              <section className="card card-pad">
                <p className="label">Contributing factors</p>
                {data.contributing_factors.map((claim, index) => <Claim key={index} claim={claim} />)}
              </section>
            )}
            <h2 className="h2" style={{ marginTop: 24 }}>Sequence of events</h2>
            <ol className="report-events">
              {data.sequence_of_events.map((event) => (
                <li key={event.addr} className={event.tag || ""}>
                  <span className="mono">{String(event.seq + 1).padStart(2, "0")}</span>
                  <code>{event.addr}</code>
                  <span>{event.text}</span>
                </li>
              ))}
            </ol>
            {data.findings.length > 0 && (
              <section className="card card-pad">
                <p className="label">Evidence</p>
                {data.findings.map((claim, index) => <Claim key={index} claim={claim} />)}
              </section>
            )}
            <p className="faint">Generated {new Date(data.generated_at).toLocaleString()}</p>
          </>
        )}
      </article>
    </div>
  );
}
