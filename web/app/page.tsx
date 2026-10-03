import runs from "../mocks/runs.json";

export default function Home() {
  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">BLACK BOX / RUNS</p>
          <h1>Agent failures, with a place to start.</h1>
        </div>
        <span className="mode">RECORDED</span>
      </header>
      <section aria-label="Recorded agent runs" className="runs">
        {runs.map((run) => (
          <article className="run" key={run.run_id}>
            <span className={`status ${run.status}`}>{run.status}</span>
            <div>
              <strong>{run.task}</strong>
              <p>
                {run.agent} · {run.steps} steps · {run.duration_ms} ms
              </p>
            </div>
            <div className="suspect">
              <span>Top suspect</span>
              <strong>{run.top_suspect ?? "No confident culprit"}</strong>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
