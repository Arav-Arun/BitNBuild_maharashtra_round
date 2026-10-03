# BLACK BOX: the flight recorder for AI agents (Bit N Build 2026, PS #2)

## Context
Problem statement #2 asks for a system that:
- records agent executions,
- **trains a model** to find the step that caused a failure,
- explains the diagnosis with evidence,
- replays the run from checkpoints, tries alternative executions, and compares traces,
- shows that the diagnosis holds up on known and **unseen** failures,
- tests fixes **without re-running unaffected steps**.

This repo previously held a scaffold for a simpler version of the idea: one toy task, 5 faults and a CLI. That scaffold has been removed. Two pieces were kept as seeds: the content-addressed checkpoint store (`blackbox/recorder`) and the localization metrics (`blackbox/eval`).

**Constraints:** team of 4, 48h+ build, Groq for LLM calls, Python and Next.js, and a "rigor + wow demo" pitch.

**Facts checked on 2026-10-03 that shape this plan:**
- Groq's free tier now offers only `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` (preview). Each is capped at 30 RPM and 200K tokens/day. Llama models were removed from the free and Developer tiers on 2026-08-16.
- Groq returns **no logprobs**, and `seed` is best effort only.
- LangGraph's time travel **re-runs every node after the checkpoint**.
- The finale is a **24h offline sprint (Oct 31–Nov 1)**. Confirm the rules on pre-built code and data. A 24h cut-line is given in the timeline section.

**What judges should remember:** tracing tools show *what* happened. Black Box shows *which step* caused the failure and *why*, and **proves the fix** by replaying only the affected part of the run.

---

## 1. Pitch and novelty (3 claims, each backed by a measured number)
1. **A learned step localizer** (LightGBM LambdaRank + GATv2 ensemble). We evaluate it on **unseen fault types, an unseen agent architecture and natural failures**, and run it zero-shot on the human-labelled **Who&When (ICML'25)** benchmark. On Who&When, the GPT-4o judge reaches only 14.2% step-level accuracy.
2. **Dependency-cone incremental replay.** It uses content-addressed memoization with build-system "early cutoff", as in Bazel and Salsa. After an edit, only the downstream steps that consume the edited value re-run. We measure it against LangGraph-style prefix replay: *"re-executed 4/18 steps, −79% tokens"*.
3. **Counterfactual verification.** The model proposes the top-3 suspect steps. For each, an auto-fix plus replay plus a *null-replay control* confirms whether that step causes the failure. **The same replay engine generates our labelled training data.**

Prior art to name in Q&A:
- LangSmith/Studio: forks re-run everything downstream.
- AgentDebugX (EMNLP'26 demo): heuristics and LLM judges, no trained model.
- AgenTracer, GraphTracer, GCJR: research pipelines with no interactive system.

## 2. Architecture
```
Target agents ──bb SDK──► Recorder ──► SQLite(WAL) + CAS blobs (zstd, Merkle state)
 (TripCrew/HopRAG/ShopDesk)   │ spans (OTel GenAI + OpenInference + blackbox.*), dep-edges, cassette
                              ▼
        Fault Forge (fork golden runs → inject → resume → label)   ──► Parquet ─► DuckDB features
                              ▼                                                    ▼
   Replay Engine (full | prefix | cone, K samples + control)  ◄──  Diagnoser (LGBM ranker + GATv2 + conformal)
                              ▼                                                    ▼
                FastAPI (REST + SSE)  ──────────►  Next.js "Investigator" UI (graph, heatmap, report, fork, diff, eval)
```

## 3. Tech stack (versions checked)
| Layer | Choice |
|---|---|
| SDK / span schema | Custom `blackbox` Python SDK, ~500 LOC. Spans use OTel GenAI semconv (`gen_ai.*`) + `openinference.span.kind` + a `blackbox.*` extension. Dependencies are span links. Optional OTLP export goes to **Arize Phoenix 20.19** (dev viewer only). Uses `opentelemetry-sdk` 1.45 |
| LLM | Groq through the **OpenAI-compatible client** (`base_url=https://api.groq.com/openai/v1`), so the backend can be swapped for Ollama with no code change |
| Agent model | `openai/gpt-oss-20b`: temp 0, fixed seed, `reasoning_effort="low"`, `reasoning_format="parsed"`. Reasoning text is stored as evidence and kept out of cache keys |
| Judge / fixer / narrator | `openai/gpt-oss-120b` with strict `json_schema` (strict mode can't be combined with tools or streaming) |
| Offline fallback | gpt-oss-20b on Ollama (same weights) |
| Store | SQLite in WAL mode with a single writer queue, plus a content-addressed blob store; DuckDB 1.5.6 over Parquet for features and eval |
| ML | LightGBM 4.7 (`LGBMRanker` lambdarank); PyG 2.8 (GATv2Conv with `edge_dim`); sentence-transformers 6.1 (MiniLM-L6 or Qwen3-Embedding-0.6B @128d); Qwen3-0.6B for surprisal (needs a Kaggle/Colab T4); scikit-learn (IsolationForest, kNN); `shap` via `pred_contrib` |
| API | FastAPI 0.142, built-in `fastapi.sse.EventSourceResponse`. Edits go over POST; the browser connects directly to FastAPI with CORS, because proxies can buffer SSE |
| UI | Next.js (App Router) + TypeScript + Tailwind + shadcn/ui; **@xyflow/react** (graph); **Monaco** diff editor; Recharts/visx (eval charts); Framer Motion (replay animation) |
| Ops | `uv` for Python, `pnpm` for JS, docker-compose for the offline demo stack |

**Budget decision on day 0:** move to Groq's **Developer tier** (1K RPM, 250K TPM) and set a spend cap. About 3K–5K runs cost **$10–35**. The free tier allows only about 30–50 runs/day per org. If the card is not possible, use the free tier plan: TripCrew + HopRAG only, plus an Ollama/Kaggle worker.

---

## 4. Feature by feature

### F1. Execution data: capture SDK
- **API:**
  - `@bb.agent`, `bb.step(addr, kind, fn, reads, writes)`
  - `bb.chat(...)` and `bb.tool(fn, **args)`: memoized effects
  - `bb.State`: a proxy that tracks reads and writes
  - `bb.now/uuid/random`: recorded, so they replay identically
- **Capability levels:**
  - **L0:** OTLP ingest. Diagnose and compare only.
  - **L1:** SDK effects. Adds prefix replay and output edits.
  - **L2:** tracked state. Adds cone replay and early cutoff.
- **Logical step addresses** (`worker[2]/chat#1`, not sequence numbers) line up an original run with its forks.
- **Normalize before hashing:** tool-call ids become `tc_{addr}_{i}`; timestamps and float formatting are canonicalized. Otherwise every cache key downstream misses.
- **Tables:**
  - `runs`: run_id, parent_run_id, fork_id, agent, agent_version, task_id, model, seed, split, outcome_success, score
  - `steps`: step_id, run_id, addr, seq, kind, name, request_key, input_hash, output_hash, state_before, state_after, cache_status, tokens_in, tokens_out, tokens_cached, latency, finish_reason, error_type, attrs_json
  - `edges`: src, dst, key, version, kind = explicit | message | inferred
  - `cassette`: key, kind, response_blob, model
  - `forks`: fork_id, base_run, edits_json, mode, samples, reexec_n, hit_n, tokens_saved, flip_rate, control_flip_rate
  - `labels`: run_id, fault_addr, fault_type, injected, verified_by_replay, manifestation_addr, recovered. **Kept separate from features.**

### Target agents (three architectures, for the cross-architecture split)
- **A. TripCrew** (demo agent, built first). It is a DAG:
  - `Planner → {FlightScout ∥ HotelScout ∥ WeatherScout ∥ FXDesk} → BudgetCalc(py) → ItineraryWriter → Verifier`
  - Mocked APIs (`search_flights`, `search_hotels`, `get_weather`, `fx_rate`, `visa_rules`) served from a seeded DB.
  - About 300 templated scenarios: Indian origin cities × "around the world" destinations, budgets in ₹, and constraints.
  - A deterministic solver/checker decides success; the INR total must match to within ₹1. Tune difficulty so the run succeeds about 70% of the time.
  - Workers have scoped contexts, so the dependency cone stays small.
- **B. HopRAG:** MuSiQue-Ans (CC BY 4.0), 2–4 hop questions.
  - Tools: local BM25 (`rank_bm25`) for search and read, plus answer.
  - Scored by EM/F1.
  - Gold sub-answers serve as oracle fixes.
- **C. ShopDesk:** single-agent ReAct on **tau2-bench retail** (MIT); airline is held out.
  - Imports tau2's env and tools and runs our own instrumented loop.
  - Uses solo mode or a cached user simulator to control tokens.
  - Checked by DB-hash reward.
  - Fallback if the token budget is tight: use it as a natural-failure test set only.

### Fault Forge: training data generated by replay
**Pipeline:**
1. Start from a golden (successful) run R.
2. Restore checkpoint C_{k-1}. This costs nothing: the cassette serves the prefix.
3. Apply fault operator Π_f to step k.
4. Resume live from there and check the outcome.
5. Run a **control fork** (no perturbation) on 10–20% of forks.

**Labels:**
- fail and control succeeds → **POSITIVE** (root = k)
- success → **RECOVERED**: a hard negative where the step was suspicious but did not cause the failure
- control fails → **FLAKY**: discard it and report the rate

**Invariants:**
- **Single-diff:** assert that the forked state differs only in the declared fields.
- Store a **manifestation step**: the first step whose action diverges after the fault.

**14 operators**, mapped to the MAST, TRAIL and AgentErrorTaxonomy taxonomies:

| Family | Seen in training | Held out as **UNSEEN** |
|---|---|---|
| Tool | T1 wrong value, T3 empty/404, T4 timeout/500 | **T2 stale data**, **T5 unit/schema drift** |
| Retrieval | R1 irrelevant docs | **R2 poisoned bridge fact** |
| Decision | D1 wrong args, D2 wrong tool, D4 premature stop | **D3 hallucinated intermediate value** |
| Coordination / state | C1 instruction misread, C4 loop | **C2 constraint dropped in handoff**, **C3 state corruption** |

**Anti-artifact rules:**
- **Ghost-hint injection** for decision and coordination faults. A biasing hint goes into step k's *input*, the LLM writes the wrong action in its own style, and the trace is recorded *without* the hint.
- Tool faults stay schema-valid and plausible: replacement values are sampled from the same field's distribution.
- Randomize the injection position to match natural failures.
- **Adversarial artifact audit:** a surface-feature classifier that separates injected steps from normal ones should score AUROC ≤ 0.65. Report this number.

**Natural failures (test only):**
- Labelled by an *earliest-first counterfactual sweep* that does not use our model: gpt-oss-120b, given gold artifacts, makes a minimal fix at each candidate step, replays, and the earliest step that flips the outcome is the label.
- Two teammates label 40–60 runs in the UI, and we report **Cohen's κ**.

**Targets:** about 1,000 base runs, about 1,700 fault forks, and about 400 natural failures, giving **about 1,500 labelled failed traces and about 22K step rows**. Freeze the dataset at a fixed hour and record its hash.

### F2. Failure diagnosis: BlackBox-Diagnoser v1
- **Features:** about 90 per step, in ablatable groups.
  - **A. Structure:** position, kind, tool, DAG depth and degree, descendants, on the path to the final answer, repeat count
  - **B. Telemetry:** latency, tokens, retries, `finish_reason=length`, errors. Z-scored against successful runs for the same tool.
  - **C. Validity:** argument and output schema checks, NaN or out-of-range values, unit tokens
  - **D. Grounding:** unsupported-entity/number rate, conflicts with earlier values, embedding drift against task and parents
  - **E. Surprisal:** Qwen3-0.6B prefill NLL (mean, max, p90, entity NLL, ΔNLL). This replaces the logprobs Groq does not return.
  - **F. Novelty:** IsolationForest, kNN and Mahalanobis distance, fitted **only on successful training runs**
  - **G. Context window:** neighbours at i±1, count of errors in descendants, is_earliest_anomaly
  - **H. Lineage:** number of consumers, reaches the final answer, later corrected
  - **I. Embedding:** PCA-16 of a MiniLM embedding
  - Also: **within-run percentile copies** of the key features (helps transfer across domains), and **feature-group dropout** with p = 0.3 on the telemetry and validity groups.
- **M1 (primary):** `LGBMRanker(objective="lambdarank")`, one group per failed run.
  - Graded labels: root = 3, same-lineage ±1 = 1, everything else 0.
  - Per-run softmax with a temperature T fitted on validation.
- **M2:** run-level failure detector, `LGBMClassifier` over run aggregates. Report AUROC, including on a length-matched subset.
- **M3:** GATv2 graph network.
  - Edges: temporal, data-dependency, same-agent and call-parent.
  - Loss: listwise cross-entropy with a Gaussian σ = 1 target, + 0.3 × run-level BCE, + 0.1 × a "successful runs are not suspicious" term.
  - Ensemble it with M1 only if it improves validation.
- **Conformal step sets** (APS, α = 0.1): *"90% chance the root cause is in {S6, S7, S9}."*
- **Counterfactual verifier:**
  1. Take the top 3 suspects.
  2. For each, retrieve the fault type from the kNN precedent library, then apply a template fix (fall back to a gpt-oss-120b fix).
  3. Run a **cone replay** of that step, 2 seeds.
  4. Run **1 null replay** from the earliest suspect.
  5. The step is a confirmed cause only if the patched run passes 2/2 and the null replay fails.
  - Every confirmed flip is added back as a new label: *"learns from every investigation"*.

### F3. Failure explanation: Crash Investigation Report
The report follows NTSB-style headings: Synopsis, Sequence of Events, Probable Cause, Contributing Factors, Findings, Safety Recommendation, Verification. Evidence, from strongest to weakest:
1. **Counterfactual effect:** patching S7 flips FAIL to PASS on 2/2 seeds while the null replay stays FAIL.
2. **Lineage blast radius:** root → symptoms → final answer.
3. **TreeSHAP top-3 reasons**, mapped to readable text.
4. **kNN precedents:** *"resembles 9 of 12 past stale-tool-data failures; `fresh=true` flipped 7 of them."*
5. **gpt-oss-120b narrative:** the input is only a JSON evidence bundle, and every claim must cite a step id. A validator rejects any number not in the bundle, regenerates once, then falls back to a template.
6. **Conformal set and calibrated confidence.** Abstain when the margin is low.

### F4 and F5. Checkpointed replay and alternative execution
- **Replay re-runs the agent program from the start**, with every effect passed through the memo layer (event sourcing in the Temporal/Restate style). Python state is never deserialized; snapshots are used for inspection and to detect divergence.
- **Cache keys:**
  - `llm_key = H(model, canon(messages), tools, params)`
  - `tool_key = H(name, version, canon(args))`
- **Modes:**
  - `full`: everything re-runs.
  - `prefix`: LangGraph semantics; this is the baseline.
  - **`cone`:** steps whose request key is unchanged hit the cache. Cone replay and early cutoff fall out of the cache keys automatically.
  - A control-flow change creates new addresses, which miss the cache and run live.
  - `ReplayDivergence` is raised if a key changes with no upstream edit.
- **Edit types:**
  - override a step's output
  - patch a prompt or system instruction
  - swap the model (20b → 120b)
  - patch a tool result or arguments
  - an LLM-proposed fix
- **Nondeterminism:** run K = 3–5 samples of the edited fork plus a control fork. Show P(success | edit) against P(success | control) with Wilson confidence intervals.
- **Metrics shown:** re-executed/cached steps, LLM calls saved, tokens saved, ₹/$ and seconds, compared with both prefix and full replay.
- **SSE events:**
  - `step`: {addr, status: hit | running | live_done | override | diverged}
  - `outcome`
  - `summary`

### F6. Model evaluation
- **Splits** are always grouped by `task_id` and source golden run, so forks never cross splits.

| Split | What it tests |
|---|---|
| S0 | Known faults on held-out tasks |
| **S1** | **Unseen fault types** (the 6 held-out operators) |
| S2 | Unseen domain |
| **S3** | **Leave-one-agent-out** |
| S4 | **Natural failures**, labelled by κ-checked counterfactual replay |
| **S5** | **Who&When AG/HC zero-shot** (`Kevin355/Who_and_When`, text-only feature variant, evaluated once after freezing) |

- **Metrics:**
  - Step localization: Top-1, Top-3, ±1, MRR, nDCG@5
  - Run-level: AUROC/AUPRC (plus length-matched)
  - **Hard-negative false-positive rate**, i.e. how often recovered steps get blamed
  - **Actionable@1**, flip@k, null-flip rate
  - Replay savings
  - Calibration: ECE and conformal coverage
  - Latency and cost per trace
- **Baselines:**
  - random, last step, position-only LightGBM, first-error heuristic, IsolationForest maximum
  - **gpt-oss-120b judge in all-at-once / step-by-step / binary-search modes** (the Who&When protocols)
  - the A2P prompt
  - published state of the art, labelled "reported"
- **Rigor checklist slide:**
  - artifact AUROC
  - shuffled-label sanity check
  - 3 seeds
  - **bootstrap 95% CIs over runs** and a paired McNemar test against the best baseline
  - ablation per feature group
- **Realistic targets:**
  - S0 Top-1 0.75–0.90
  - **S1 Top-1 0.40–0.60** (above 0.85 suggests leakage)
  - Who&When AG 0.18–0.30, which beats GPT-4o's 0.125
  - replay savings 40–65%

### F7. Trace comparison
- **Alignment:** match the two traces by logical address. Where control flow differs, use an LCS/Needleman-Wunsch alignment over step signatures (kind, tool, args hash).
- **Per-step class:** identical, cached, changed, inserted or removed.
- **The view shows:**
  - the **first divergence** point
  - per-key Merkle state diffs
  - Monaco side-by-side diffs of inputs and outputs
  - the outcome delta
  - **"Compare with nearest successful run of same task"** for any failed run
- **Fleet view (stretch):** apply one fix across all traces that share a failure signature: *"fixes 17/20, breaks 0/50 passing runs"*.

---

## 5. Frontend: "Investigator" (Next.js)
**Style:** cockpit HUD with international-orange accents, monospace telemetry, a FDR pane (tools and state) and a CVR pane (LLM messages and reasoning), and a "MAYDAY" toast when a run fails. Do not reference real crashes.

**Screens:**
1. **Hangar:** list of runs with filters (agent, outcome, split) and a run-level risk score.
2. **Flight view:** a React Flow DAG with a suspicion heatmap, a timeline scrubber, and a step inspector (inputs, outputs, state, reasoning, SHAP reasons).
3. **Crash Investigation Report** panel.
4. **Fork & Fix:** a Monaco editor for the edit, a mode toggle (cone/prefix/full), and an animated live replay over SSE. Cached nodes grey out and re-running nodes pulse, with a counters bar.
5. **Trace Diff.**
6. **Eval dashboard:** leaderboard with CIs, generalization matrix, ablations, reliability diagram, and replay-savings chart. It reads precomputed JSON.
7. **Label mode**, used for the human κ check.

**API:**
- `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/diagnosis`, `GET /runs/{id}/report`
- `POST /forks` → `GET /forks/{id}/stream` (SSE)
- `GET /diff?a=&b=`, `GET /eval`, `POST /v1/traces` (OTLP ingest)

## 6. Repo layout (monorepo)
```
blackbox/              Python package (uv + pyproject.toml)
  recorder/            CAS checkpoint store (kept) → + Merkle state, SQLite writer
  sdk/                 state.py memo.py replay.py otel.py adapters/langgraph.py
  forge/               operators.py inject.py label.py natural_label.py
  ml/                  features.py surprisal.py novelty.py train_lgbm.py gnn.py conformal.py verifier.py explain.py
  eval/                localization metrics (kept) + splits, baselines, judges, bootstrap, whowhen_adapter.py
agents/                tripcrew/ hoprag/ shopdesk/   (+ mock_apis/, checkers/)
server/                main.py (FastAPI) routes/ sse.py diff.py
web/                   Next.js app (app/, components/graph, components/diff, components/eval)
data/                  blackbox.db, blobs/, parquet/, eval/*.json  (gitignored, hashed)
docker-compose.yml     Makefile (make demo-offline)
```

## 7. Timeline (48h, 4 roles)
**Roles:**
- **P1, Infra:** SDK, CAS, replay engine, FastAPI, SSE
- **P2, Agents and data:** three agents, mock APIs, checkers, Fault Forge, run generation
- **P3, ML:** features, models, verifier, evaluation, Who&When adapter
- **P4, Frontend and pitch:** UI, report, branding, deck, demo script

| Hours | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| 0–4 | Freeze the schema and SDK API contract (shared doc); CAS + memo | TripCrew skeleton + mock DB + checker | Who&When adapter; baselines harness | Next.js scaffold, design system, mock JSON |
| 4–10 | Recorder → SQLite; prefix replay | TripCrew done; HopRAG; **start golden runs** | Feature pipeline on early traces | Hangar + Flight view on mock data |
| 10–16 | **Cone replay + early cutoff**; divergence check | Fault Forge operators + control forks; **bulk generation running** | LightGBM v0 + first-error/LLM-judge baselines | React Flow heatmap wired to real API |
| **16: integration checkpoint 1** | end-to-end: record → diagnose → view one run | | | |
| 16–24 | Fork API + SSE; K-sample + control | ShopDesk (tau2); natural-failure labelling sweep | Surprisal/novelty features; splits S0–S3; artifact audit | Fork & Fix animation + counters |
| 24–32 | Diff engine; OTLP ingest | κ labelling (P2 + P4); dataset **freeze @ h30** | GATv2 + conformal + verifier loop + SHAP + precedents | Trace Diff + Report panel |
| **32: integration checkpoint 2** | full demo loop works on real data | | | |
| 32–40 | Offline mode (Ollama, cassette); docker | Pick and verify demo scenarios (10/10 flips) | Final eval with CIs; Who&When run once; eval JSON | Eval dashboard; deck |
| **40: FEATURE FREEZE** | bug fixes only | | | |
| 40–48 | Hardening, fallback video | Backup scenarios | Q&A numbers sheet | 3 timed rehearsals |

**MVP cut-line, which must work:**
- TripCrew + HopRAG
- SDK with cone replay
- Fault Forge
- LightGBM with S0 and S1
- first-error and LLM-judge baselines
- Flight view, Fork & Fix, Diff, Report, and the eval table

**Stretch goals:**
- ShopDesk
- GATv2
- Who&When
- conformal sets
- the LangGraph adapter
- fleet fix validation

**For a 24h finale:** drop ShopDesk and the GNN, generate data by hour 8, and freeze at hour 18.

## 8. Demo script (4:00)
1. **0:00–0:20, hook.** "Every aircraft carries a black box, and it's actually orange. Your AI agents carry nothing."
2. **0:20–0:50, the failure.** TripCrew runs live: *Mumbai→Singapore, 2 adults, 12–16 Dec, under ₹1,20,000, vegetarian, refundable*. The plan comes back at ₹1,31,400 and is marked FAILED.
3. **0:50–1:30, the investigation.**
   - The heatmap's top suspect is **step 7 `fx_rate(SGD→INR)`, p = 0.82**, with an `as_of` date 7 months old.
   - Lineage arrows run 7 → BudgetCalc → final plan.
   - Step 9, a recovered 500 error, is shown as *suspicious but not causal*.
   - The Crash Investigation Report opens.
4. **1:30–2:20, the "wow" moment.**
   - Click **Fork & Fix** and apply the suggested `fresh=true` fix.
   - 14 nodes grey out ("served from recorder") and **4 re-execute**.
   - Counters: *LLM calls 4/11 · tokens −79% · 2.8 s vs 13 s*.
   - The outcome turns **green** at ₹1,14,900, and the Trace Diff shows the divergence.
5. **2:20–3:10, the rigor.**
   - Leaderboard with the columns Known · **Unseen faults** · **Unseen agent** · **Natural** · Who&When, against the gpt-oss-120b judge, with CIs.
   - ms vs seconds per trace, and ₹0 vs ₹ per trace.
   - The artifact audit and κ, one line each.
6. **3:10–3:40, generalization.** A ShopDesk *natural* failure (wrong `order_id`) is caught by the same model.
7. **3:40–4:00, the close.** "Three lines to instrument any agent, OTel-compatible. Black Box finds the step, explains why, and proves the fix without re-flying the flight."

**Deck (9 slides + 1 backup):** Title → Problem (14.2% Who&When step accuracy) → Insight (record, learn, replay; the replay engine is the data factory) → Architecture → Live demo → Data (3 agents × 14 faults) → Results → Replay efficiency → Impact and roadmap (crash tests for agents in CI) → Backup: threats to validity.

**Q&A prep:**
- "vs LangSmith?" They show traces; we localize the step and replay only the affected part.
- "vs AgentDebugX?" No trained model there, and no generalization evaluation.
- "Isn't synthetic data cheating?" Unseen fault types, natural failures that are test-only, κ-checked labels, and the artifact audit.
- "Nondeterminism?" Cassette, K samples, control forks, and a measured flaky rate.
- "Why not just use an LLM?" The leaderboard, plus about 1000× lower cost and latency.

## 9. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Groq limits or outage | Developer tier on day 0; token bucket driven by the `x-ratelimit-*` headers; prompt-prefix caching (cached tokens don't count toward limits); pause bulk generation during the pitch (limits are per organization) |
| Groq model churn | Model ids in config; OpenAI-compatible client; record `response.model` |
| Replay is not deterministic | Cassette for the prefix; K samples + control; demo scenarios that flip 10/10 |
| ReAct agents show no savings | TripCrew's DAG with scoped contexts is the demo agent. Be candid that linear agents save on the prefix and memoized tools only |
| Model learns injection artifacts | Ghost-hint injection; plausible values; audit AUROC ≤ 0.65; natural-failure column |
| Wi-Fi or demo failure | `make demo-offline` (Ollama + cassette); a LIVE/RECORDED badge; a 90 s backup video; phone hotspot |
| Data generation runs late | Freeze at hour 30 (hour 18 for a 24h finale); LightGBM trains in seconds; eval JSON is precomputed |

## 10. Verification (end-to-end)
1. **SDK unit tests (pytest):**
   - Content addressing is stable across runs, after normalizing ids.
   - A replay with no edits gives **100% cache hits** and no `ReplayDivergence`.
   - Editing FXDesk re-executes only {FX, BudgetCalc, Writer, Verifier}.
2. **Forge tests:** the single-diff invariant holds, and the control-fork flaky rate stays under 10%.
3. **ML:**
   - `make eval` regenerates `eval/*.json` and runs the shuffled-label check (should score about random) and the artifact audit.
   - Ablation table.
   - The Who&When result is recorded once, together with the model hash.
4. **API:** `curl` the SSE stream for a fork and confirm the event order: step* → outcome → summary.
5. **UI:** drive the full demo script in a browser (record → heatmap → Fork & Fix → diff → eval). Check it at 1366×768 and with Wi-Fi off (offline mode).
6. **Rehearsal:** three timed runs of the 4:00 script on the projector setup.
