"use client";

import { useEffect, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { UNREACHABLE_HINT, api, errorText } from "../../lib/api";
import { useAppContext } from "../shell/AppContext";

// Travel examples use a synthetic catalog; research uses downloaded public documents.
const EXAMPLES = [
  { route: "Delhi to Tokyo", prompt: "Plan a trip from Delhi to Tokyo departing 2026-12-12, returning 2026-12-17, for 2 adults. Budget ₹1,00,000." },
  { route: "Chennai to Bangkok", prompt: "Family holiday from Chennai to Bangkok, 2026-12-20 to 2026-12-24, 3 adults, budget ₹1,40,000. Vegetarian: yes." },
  { route: "Hyderabad to Dubai", prompt: "Weekend from Hyderabad to Dubai departing 2026-12-05, returning 2026-12-08, for 2 adults. Budget Rs. 1,00,000. No red-eye: yes." },
  { route: "Bangalore to London", prompt: "Solo trip from Bangalore to London departing 2026-12-01, returning 2026-12-07, for 1 adult. Budget ₹80,000. Refundable: yes." },
];
const SAMPLE = EXAMPLES[0].prompt;
const RESEARCH_EXAMPLE_LABELS: Record<string, string> = {
  "5a8b57f25542995d1e6f1371": "Compare nationalities",
  "5a8c7595554299585d9e36b6": "Find a government role",
  "5a85ea095542994775f606a8": "Identify a book series",
};

type TaskRunResponse = { run_id: string; status: "passed" | "failed"; task: string; steps: number };

export function NewRunPage() {
  const router = useRouter();
  const { mode, staticBundle, reachable } = useAppContext();
  const [prompt, setPrompt] = useState(SAMPLE);
  const [workflow, setWorkflow] = useState<"research" | "tripcrew">("research");
  const [questions, setQuestions] = useState<{id: string; question: string; split: string}[]>([]);
  const [injectEmpty, setInjectEmpty] = useState(false);
  const [loadingQuestions, setLoadingQuestions] = useState(true);
  const [injectStaleFx, setInjectStaleFx] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api<{items: {id: string; question: string; split: string}[]}>("/research/questions")
      .then((result) => { if (active) {
        setQuestions(result.items);
        setPrompt(result.items.find((q) => q.split === "validation")?.question || result.items[0]?.question || "");
      } })
      .catch(() => { if (active) setPrompt(""); })
      .finally(() => { if (active) setLoadingQuestions(false); });
    return () => { active = false; };
  }, []);

  function chooseWorkflow(value: "research" | "tripcrew") {
    setWorkflow(value);
    setPrompt(value === "tripcrew" ? SAMPLE : questions.find((q) => q.split === "validation")?.question || questions[0]?.question || "");
    setError("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const result = await api<TaskRunResponse>("/tasks/run", {
        method: "POST",
        body: JSON.stringify({ prompt, workflow, inject_stale_fx: workflow === "tripcrew" && injectStaleFx, inject_empty_retrieval: workflow === "research" && injectEmpty }),
      });
      router.push(`/investigate/${encodeURIComponent(result.run_id)}`);
    } catch (caught) {
      setError(errorText(caught) || "The task could not be run.");
      setBusy(false);
    }
  }

  const enabled = reachable && (workflow === "research" ? mode === "live" && questions.length > 0 : mode === "offline" || mode === "live") && !staticBundle;
  const unavailable = !reachable
    ? `The API is not reachable. ${UNREACHABLE_HINT}`
    : staticBundle
      ? "This is the static recorded showcase, so new runs cannot be recorded. Start the local API to run tasks."
      : mode === null
        ? "Connecting to the API…"
        : mode === "recorded"
          ? "Recorded mode is read-only. Set MODE=live to run tasks with the configured model."
          : "";

  return (
    <div className="page">
      <div className="page-inner new-run-page">
        <h1 className="h1">New task</h1>
        <div className="row" role="group" aria-label="Workflow">
          <button className={`btn${workflow === "research" ? " btn-primary" : ""}`} type="button" onClick={() => chooseWorkflow("research")}>Research</button>
          <button className={`btn${workflow === "tripcrew" ? " btn-primary" : ""}`} type="button" onClick={() => chooseWorkflow("tripcrew")}>Travel demo</button>
        </div>
        <p className="faint">{workflow === "research" ? "Groq · HotpotQA / Wikipedia" : mode === "live" ? "Live LLM · synthetic travel catalog" : "Test runner · synthetic travel catalog"}</p>

        <form className="new-run-form card card-pad" onSubmit={submit}>
          <label className="label" htmlFor="task-prompt">Your request</label>
          <textarea
            id="task-prompt"
            className={`input prompt-input${workflow === "research" ? " research-prompt" : ""}`}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={1200}
            required
            minLength={24}
            spellCheck={false}
            placeholder={workflow === "research" ? "Ask about the downloaded documents…" : SAMPLE}
          />
          <div className="spread prompt-meta">
            <span className="faint">{prompt.length}/1200</span>
            <button type="button" className="btn btn-sm" onClick={() => chooseWorkflow(workflow)}>Reset example</button>
          </div>

          {workflow === "research" ? <>
            <div className="prompt-examples" role="group" aria-label="Research examples">
              {questions.filter((q) => q.split === "validation").slice(0, 3).map((q) => (
                <button type="button" className={`btn btn-sm${prompt === q.question ? " btn-chosen" : ""}`} key={q.id} aria-label={q.question} title={q.question} onClick={() => setPrompt(q.question)}>{RESEARCH_EXAMPLE_LABELS[q.id] || q.question}</button>
              ))}
            </div>
            <p className="faint">Listed questions have reference answers. Custom questions are checked for source citations only.</p>
            {loadingQuestions ? <p className="muted" role="status">Loading questions…</p> : questions.length === 0 && <p className="error-box" role="status">Download documents: python -m agents.research download --count 24</p>}
            {mode !== null && mode !== "live" && <p className="muted">Research requires MODE=live and GROQ_API_KEY.</p>}
            <label className="prompt-checkbox">
              <input type="checkbox" checked={injectEmpty} onChange={(event) => setInjectEmpty(event.target.checked)} />
              <span><strong>Test a retrieval failure</strong><small>Return no documents, then restore them with Fork and fix.</small></span>
            </label>
          </> : <>
          <div className="prompt-examples" role="group" aria-label="Example trips with the exchange-rate bug">
            <span className="faint">Examples with the exchange-rate bug</span>
            {EXAMPLES.map((example) => (
              <button type="button" key={example.route} className={`btn btn-sm${prompt === example.prompt && injectStaleFx ? " btn-chosen" : ""}`}
                onClick={() => { setPrompt(example.prompt); setInjectStaleFx(true); }}>
                {example.route}
              </button>
            ))}
          </div>

          <div className="prompt-help">Fly from Mumbai, Delhi, Bengaluru, Chennai or Hyderabad to Singapore, Bangkok, Dubai, London, Tokyo or Paris. Give dates as YYYY-MM-DD, 1 to 6 travellers, and a budget in rupees.</div>

          <label className="prompt-checkbox">
            <input type="checkbox" checked={injectStaleFx} onChange={(event) => setInjectStaleFx(event.target.checked)} />
            <span><strong>Use an old exchange rate</strong><small>Add a reproducible budget error.</small></span>
          </label>

          </>}

          {error && <p className="error-box" role="alert">{error}</p>}
          {unavailable && <p className="muted" role="status">{unavailable}</p>}
          <div className="row prompt-actions">
            <span className="spacer" />
            <button className="btn btn-primary" type="submit" disabled={!enabled || busy || prompt.trim().length < 24}>
              {busy ? "Recording run…" : "Run and inspect"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
