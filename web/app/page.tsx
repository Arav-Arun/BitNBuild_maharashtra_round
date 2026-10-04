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
        {runs.items.map((run) => (
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
              <strong>
                {suspectName(run.top_suspect as unknown)}
              </strong>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}

function suspectName(suspect: unknown): string {
  if (typeof suspect === "string") return suspect;
  if (suspect && typeof suspect === "object" && "name" in suspect) return String(suspect.name);
  return "No confident culprit";
}
