# Black Box: build plan

Black Box is a flight recorder for AI agents. It records every step an agent takes. When a
run fails, it points to the step that most likely caused the failure, shows why with evidence
from the recording, and lets you fix that step and re-run from exactly that point. Only the
steps the fix actually affects are re-run.

The differentiator is the combination: a trained step localizer, dependency-aware partial
replay, and fix verification against an unchanged control. No result is called proven just
because one stochastic re-run happened to pass.

This document is the build order. Each task ends with a **Test** section. Don't start the next
task until the current task's tests pass.

---

## Part A: What we are building

### A1. One-paragraph story (the demo)

A trip-planning agent books Mumbai → Singapore for 2 adults under ₹1,20,000 and returns a plan
costing ₹1,31,400, so it fails. Black Box opens the run:
- Its top suspect is step 7, `fx_rate(SGD→INR)`, at 82%.
- The evidence: the exchange rate was 7 months old, and that number flowed into the budget and
  then the final plan.
- Step 9 also had an error, but the agent recovered from it, so it is not blamed.

The user clicks **Fork and fix** and re-fetches a fresh rate. Black Box re-runs only the 4 steps
that depend on that rate. The other 14 come from the recording. The plan now costs ₹1,14,900 and
passes. A side-by-side comparison shows where the two runs split apart. A results page shows how
accurate the model is on failure types it never saw during training.

### A2. The problem statement's 7 features and where each one is built

| Feature | What it means for us | Built in |
|---|---|---|
| Execution data | Record every step's exact inputs, outputs, and state before and after | Tasks 2, 4, 5 |
| Failure diagnosis | A trained model ranks the steps of a failed run by blame | Tasks 6, 7, 8 |
| Failure explanation | Evidence: reasons, the path of damage, similar past cases, replay proof | Task 9 |
| Checkpointed replay | Resume from the exact state before any step | Task 3 |
| Alternative execution | Edit a step and re-run only what that edit affects | Tasks 3, 9, 12 |
| Model evaluation | Accuracy on known failures, unseen failures, and natural failures, with baselines | Task 8 |
| Trace comparison | Align two runs, find where they diverge, and diff states and outcomes | Tasks 9, 12 |

### A3. Words we use

| Term | Meaning |
|---|---|
| **Run** | One complete execution of an agent on one task, ending in pass or fail |
| **Step** | One unit inside a run: an LLM call, a tool call, a retrieval, or a state update |
| **Address** | A stable name for a step, e.g. `fx/tool#1`. It lets us match the same step across two runs even when the step numbers shift |
| **Snapshot** | The agent's full state after a step, stored under a hash of its content |
| **Cassette** | The store of every recorded LLM and tool response, looked up by a hash of the exact request |
| **Fork** | A new run made from an old one with one step changed |
| **Cone** | The steps downstream of an edit that actually consume the changed value. Only these re-run |
| **Control** | A fork with no edit. It measures how often a run flips by chance (noise) |
| **Root cause** | The earliest step where a minimal fix turns a failed run into a passing one |
| **Verified** | The edited fork's lower confidence bound is above the unchanged control's upper bound |
| **Refuted** | A valid intervention reached the candidate but repeatedly failed to improve the outcome |
| **Inconclusive** | The evidence is too weak, noisy or under-sampled; the product does not force an answer |
| **Cached / invalidated / live** | A replay step reused an exact-hash response / must re-run because an input changed / made a real model or tool call |

### A4. Architecture

```
 Agents (TripCrew, HopRAG, ShopDesk*)
   │  blackbox SDK: bb.step / bb.chat / bb.tool / bb.State
   ▼
 Recorder ──► SQLite (runs, steps, edges, labels, forks) + content store (snapshots, payloads)
   │                                  │
   ▼                                  ▼
 Replay engine (full | prefix | cone) │   Fault Forge: fork successful runs, inject one fault, label
   │                                  ▼
   │                        Features (DuckDB/Parquet) ──► Diagnoser (LightGBM ranker, optional GNN)
   │                                  │
   ▼                                  ▼
 Verifier + explainer + diff  ◄───────┘
   │
 FastAPI (REST + server-sent events)  ──►  Next.js UI: Runs · Investigate · Fork and fix · Compare · Results
   ├──► local MCP server: get_suspects · get_step · fork_and_verify · export_regression_test
   └──► pytest exporter: verified fork → offline regression test + cassette fixture
```
`*` ShopDesk is a stretch goal.

### A5. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ managed by `uv`; TypeScript for the web app |
| LLM | Groq through its OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`). Agent model `openai/gpt-oss-20b`; judge, fixer and narrator `openai/gpt-oss-120b`. Model IDs live in config |
| Offline LLM | Ollama running `gpt-oss:20b` (the same model weights) on the same client, by changing `base_url` |
| Storage | SQLite in WAL mode + a content-addressed blob store (the existing `blackbox/recorder` Store) |
| Analytics | DuckDB over Parquet exports |
| ML | LightGBM 4.7 (`LGBMRanker`), scikit-learn (IsolationForest, kNN), sentence-transformers (MiniLM), Qwen3-0.6B for surprisal features (Kaggle GPU), PyTorch Geometric (optional GNN) |
| API | FastAPI with built-in server-sent events (SSE) for live replay streaming |
| Agent interface | A small local MCP server over the same service layer; `stdio` first, no hosted MCP dependency |
| UI | Next.js (App Router), Tailwind, shadcn/ui, @xyflow/react + ELK/dagre (fixed graph layout), Monaco (diff and edit), Recharts (charts), TanStack Table and Virtual |
| Tracing interoperability | OpenTelemetry GenAI span names and attributes, so traces can be exported to Arize Phoenix |
| Deploy | Vercel Hobby (web) + Render Free (read-only recorded API, no disk) + Docker Compose (local live/offline demo) |

### A6. Research-derived product decisions

The sweep is a design input, not a list of logos for the deck. These are the concrete decisions
it changes:

| Source | Decision we implement | What remains distinct about Black Box |
|---|---|---|
| LangGraph Studio / LangSmith | Every step has **Fork with edit** and **Re-run unchanged** actions; forks are immutable branches | We invalidate and re-run only the dependency cone, then display calls/tokens/time saved |
| Laminar / AgentOps | Original → control → fixes timeline; every replay step shows **cached / invalidated / live** | The timeline is also causal evidence, not only session playback |
| AgentDebugX / Sentry Seer | Four-stage UI: **Localize → Attribute → Propose → Verify**; separate visible failure from responsible step; show a Suspect steps list | A trained ranker, abstention and replay verdicts replace an unsupported confident answer |
| Pernosco / Docent | Click any scalar value to jump to the producing step and exact source field; every evidence chip deep-links to a field | Recorder-level JSON-pointer provenance makes the link deterministic |
| Chronicle | Export a verified fork as a network-free pytest regression | The test asserts the original failure, the fix, and the exact invalidation cone |
| Raindrop Workshop | Local MCP tools for coding agents | The MCP calls the same confidence-bounded verifier; no separate diagnosis logic |
| Langfuse | Deterministic ELK/dagre graph coordinates and collapsed repeated loops | Replays animate without moving nodes, so causal changes remain legible |
| Lucidic / Muscle-Mem | Report repeated outcomes as “fix 5/5 vs control 0/5”; exact-hash cache hits only | Confidence bounds decide the verdict; semantic similarity is never treated as a cache hit |

The evaluation borrows the strongest controls from the literature:

- **AgenTracer / Who&When:** replay-derived labels and graded relevance: root = 3, ±1 causal
  hop = 2, ±2 causal hops = 1.
- **CAR / Repair or Resample? / AgentLens:** paired K-sample edited and unchanged runs, not a
  single lucky pass.
- **GCJR / GraphTracer / DeFA:** partial replay and explicit dependency edges; report full
  provenance vs protocol-only vs no-edge ablations.
- **MASPrism / AgentRx:** cheap local surprisal plus deterministic checks derived from tool JSON
  schemas (empty query, swallowed error, repeated call) as features and evidence.
- **Conformal attribution / DoVer:** a calibrated suspect set, an abstain path, and exactly three
  public verdicts: **VERIFIED, REFUTED, INCONCLUSIVE**.
- **TelemetrySuffBench:** report full Black Box record vs OTel-style fields vs outputs-only, so
  the pitch demonstrates what the richer recorder contributes.
- **Who&When Pro / TraceElephant:** injected faults are not enough; natural failures remain a
  separate, test-only split.

Before publishing the deck or paper, put the canonical URL, venue/version and access date for
every cited work in a source ledger. Do not claim a reproduced result unless our adapter and
split match the source.

---

## Part B: External tools and access

### B1. Accounts and keys the team needs

| Tool | Why we need it | Who | Access / secret | Cost |
|---|---|---|---|---|
| **Groq Free** | Curated live runs, the demo, a small LLM-judge sample and only the dataset calls that fit the quota | One org owner; everyone else uses a project key | `GROQ_API_KEY` in `.env` only. Stay on Free. On 2026-10-03 the official table lists 1K requests/day and 200K tokens/day per `gpt-oss` model; the console's Limits page is the source of truth | **$0** |
| **GitHub** | Repo and CI | All 4 | Write access to this repo | Free |
| **Hugging Face** | Download the Who&When test set (`Kevin355/Who_and_When`). TRAIL (`PatronusAI/TRAIL`) is optional and needs an accepted gate | P3 | Read token only for gated datasets | Free |
| **MuSiQue / tau2-bench** | Task data for HopRAG and ShopDesk | P2 | Public GitHub downloads (CC BY 4.0 / MIT) | Free |
| **Kaggle (Colab fallback)** | GPU to compute Qwen3-0.6B surprisal features and, if needed, small open-model batch inference | P3 | Personal account; export artifacts after each session | **$0**, quota and availability not guaranteed |
| **Ollama** | Local generation and offline demo; use `gpt-oss:20b` only if the laptop fits it, otherwise a smaller configured open model | P1 laptop | Local install; model ID lives in config | **$0** |
| **Vercel Hobby** | Host the static/Next.js web UI | P4 | Personal project linked to the repo; env `NEXT_PUBLIC_API_URL` | **$0** within Hobby limits |
| **Render Free** | Host a read-only recorded API with the demo DB, model and blobs baked into the image | P1 | Deploy key; **no Groq key and no required disk** | **$0**; cold starts and ephemeral writes accepted |
| Arize Phoenix (optional) | Developer trace viewer during the build | Anyone | Runs locally (`phoenix serve`) | Free |

**Secret rules**
- `.env` is gitignored. Commit a `.env.example` with variable names only.
- The public deployment runs in **recorded mode** and never holds a Groq key.
- Pause bulk data generation during the live pitch. Groq's account Limits page decides the
  effective quota; do not assume that four teammates means four times the quota.

### B2. What Black Box itself is allowed to touch

- **The SDK only sees calls that go through it** (`bb.chat`, `bb.tool`, `bb.State`). It does
  not read the rest of the machine.
- **Agent tools are sandboxed and deterministic.** That means mocked travel APIs, a local BM25
  search index, an in-memory DB copy and a small Python calculator. Tools have no real side
  effects, so replaying a step can't book anything or change anything outside the run.
- **Groq's built-in server-side tools (web search, code execution) are disabled.** They run on
  Groq's side, so we can't record or replay them.
- **Redaction hook.** Before anything is stored, a redactor masks API keys, emails and phone
  numbers in prompts and outputs. It uses regex plus a configurable list.
- **All data stays local by default**: SQLite plus blob files. Sharing a run means exporting it
  or using recorded mode.

### B3. Zero-cost feasibility gate

**Decision: all eight research upgrades fit a four-person, 48-hour build at $0 only in their
thin, end-to-end form.** The previous 1,500-run target, paid Groq Developer tier and hosted
persistent SQLite do not fit that constraint. Public hosting is a read-only showcase; the live
system runs locally.

Free-tier facts checked on 2026-10-03:

- [Groq Free limits](https://console.groq.com/docs/rate-limits) currently list 1K requests/day
  and 200K tokens/day for each `openai/gpt-oss` model; exact account limits can differ.
- [Kaggle notebooks](https://www.kaggle.com/docs/efficient-gpu-usage) provide a free GPU quota
  that is usually around 30 hours/week but varies with demand. Colab is only a fallback because
  its free resources are explicitly not guaranteed.
- [Vercel Hobby](https://vercel.com/docs/plans) is free and pauses projects when included usage
  is exhausted instead of being a paid dependency.
- [Render Free](https://render.com/docs/free) provides 750 instance-hours/month, spins down after
  15 idle minutes and has an ephemeral filesystem. [Persistent disks are paid-only](https://render.com/docs/disks),
  so the demo artifacts are immutable build inputs, not a writable production database.

**Budget guardrails:** do not enter a payment method, upgrade a service, or depend on promotional
credits for the judged path. Any optional credit is a scale-up experiment, never a prerequisite.

| Upgrade | Thin implementation | Added effort | 48h decision |
|---|---|---:|---|
| Click-a-value provenance | JSON-pointer source map + jump/highlight | 4 person-hours | Must ship |
| Three verdicts + abstain | Paired K=5 replay, Wilson 95% bounds, calibrated abstention | 5 person-hours | Must ship |
| Cached/live/invalidated badges | One replay-state enum used by API and graph | 2 person-hours | Must ship |
| Export regression test | One pytest template + cassette fixture + UI button | 4 person-hours | Must ship |
| MCP server | Four local `stdio` tools over the service layer | 3 person-hours | Must ship; MCP Inspector is the $0 demo client |
| DeFA-style edge ablation | Full vs protocol-only vs no edges | 2 person-hours | Must ship |
| Telemetry sufficiency ablation | Full record vs OTel-style vs outputs-only | 2 person-hours | Must ship |
| Corrected slide 2 | Current range, cost/latency/proof framing | 0.5 person-hours | Must ship |

That is about **22.5 person-hours across the team**. Pay for it by removing the light theme,
PDF export, ShopDesk, the GNN, the LLM narrative and fleet validation from the 48-hour build.
Keep Markdown export, TripCrew + a small HopRAG set, the trained ranker, natural failures and the
complete investigate → edit → selective replay → control → verdict → regression-test flow.

---

## Part C: The two guarantees that make this work

### C1. "We know exactly which step went wrong, and why"

**1. Every step is recorded completely.** Each step row stores:
- the exact messages sent to the LLM, or the exact tool arguments;
- the exact response, plus the model's reasoning text (kept as evidence, never used in cache
  keys);
- state before and after, stored as hashes of snapshots;
- the state keys it **read** and **wrote**, with versions;
- which earlier steps produced the values it used (**provenance edges**);
- for each scalar input/output/state value, its RFC 6901 JSON Pointer and the producer's step
  address + pointer, so a click can navigate to the exact origin;
- timing, token counts, errors, retries and finish reason.

Here is one stored step:
```json
{
  "run_id": "TC-0412", "addr": "fx/tool#1", "seq": 9, "kind": "tool", "name": "fx_rate",
  "request_key": "sha256:5e1c…", "input_hash": "sha256:a91f…", "output_hash": "sha256:77d0…",
  "state_before": "sha256:c2b4…", "state_after": "sha256:0f9e…",
  "reads": ["plan.currency@v1"], "writes": ["fx.SGD_INR@v1"],
  "latency_ms": 41, "tokens_in": null, "error_type": null, "cache_status": "live"
}
```
Edges are stored separately, e.g. `fx/tool#1/output/rate ─fx.SGD_INR@v1→ budget/tool#1/input/rate`.

**2. Training labels are intervention-backed.** We create failures ourselves:
- Take a run that passed and change exactly one step (Task 6).
- The run is only labelled if all three of these hold:
  - the original run passed;
  - the paired no-edit controls reproduce the pass;
  - the changed runs reproduce the failure and the edit reaches the expected downstream field.
- Exact injected labels use the edited step as root. Graded training relevance follows causal
  graph distance: root = 3, ±1 causal hop = 2, ±2 causal hops = 1; distractors = 0.

**3. At diagnosis time we rank, then intervene.**
- The model gives each step a probability of being the root cause.
- When calibration cannot isolate a small suspect set, it says **no confident culprit** and
  abstains. The ranked evidence remains inspectable.
- For the top 3 suspects, the verifier applies a valid fix and runs paired edited/control samples
  with identical seeds. K=5 is the default; K=3 is a preview that may remain inconclusive.
- **VERIFIED:** the edited fork's Wilson 95% lower bound is above the control's upper bound.
- **REFUTED:** a known-good or nearest-passing-twin intervention propagated through the candidate,
  but repeated outcomes show no improvement. A failed speculative patch alone does not refute a
  candidate.
- **INCONCLUSIVE:** every other case, including overlapping bounds or an invalid intervention.
- The UI always separates "the model suspects" from "the intervention supports/refutes/has not
  resolved it".

**4. "Why" is shown as evidence, not just stated.** Each diagnosis includes:
- the top 3 reasons the model scored the step high, in plain language;
- the path of damage, from the step through every step that consumed its output to the final
  answer;
- similar past failures and which fix worked for them;
- clickable field-level citations to each source value;
- the edited and unchanged pass rates, intervals, sample count and verdict.

An optional LLM-written summary may only cite step addresses and numbers that appear in that
evidence. A validator rejects anything else.

### C2. "We can continue from that exact step"

Replay re-runs the agent program from the beginning, but **every LLM call and tool call goes
through the cassette**:
- **Steps before the edit** send byte-identical requests, so they get recorded answers instantly
  and make no API calls.
- **The edited step** gets the override (a new output, new arguments, a patched prompt or a
  different model).
- **Steps after the edit:**
  - if a step's request is unchanged, because it doesn't consume the edited value, it is served
    from the recording;
  - if its request changed, it runs live. Those live steps are the **cone**.
  - if a re-run step produces the same output as before, everything after it stays cached. Build
    systems call this "early cutoff".

A cache hit is **only** an exact canonical-request hash match. Embedding similarity or "close
enough" output never counts as cached. Before replay starts, every downstream step in the static
dependency cone is marked **invalidated**; execution then resolves it to **cached** if its exact
request is unchanged, or **live** if a real call is required.

**Proof we're at the exact same state:** at every step before the edit, the replayed state hash
must equal the recorded hash. If any hash differs, replay stops with `ReplayDivergence` instead of
silently continuing from a different state.

**Inspecting state needs no execution.** Any step's before/after snapshot can be opened directly
from the content store.

---

## Part D: Tasks, in order

**Owners:**
- **P1, Infra:** SDK, replay, API, deploy
- **P2, Agents and data:** agents, Fault Forge, dataset
- **P3, ML:** features, model, evaluation, explanations
- **P4, UI and pitch**

The UI starts on day 1 using mock JSON that matches the API contract from Task 1. It does not
wait for the backend.

```
Task 1 ─► Task 2 ─► Task 3 ─► Task 4 ─► Task 6 ─► Task 7 ─► Task 8 ─► Task 9 ─► Task 10 ─► Task 13 ─► Task 14
                              Task 5 ──┘                                                    ▲
Task 1 ─► Task 11 (mock data) ─► Task 12 ──────────────────────────────────────────────────┘
```

---

### Task 1: Repo setup, configuration and contracts
**Owner:** P1, all review · **Est.:** 3h

**Goal:** everyone builds against the same schemas from hour 3.

**Build:**
1. Folder layout:
   ```
   blackbox/            recorder/ (existing CAS store), sdk/, replay/, forge/, ml/, eval/ (existing metrics), explain/, diff/
   agents/              tripcrew/, hoprag/, shopdesk/
   server/              FastAPI app
   web/                 Next.js app
   data/                gitignored: blackbox.db, blobs/, parquet/, models/, eval/
   ```
2. `blackbox/config.py`. Reads `.env`: `GROQ_API_KEY`, `LLM_BASE_URL`, `AGENT_MODEL`,
   `JUDGE_MODEL`, `MODE=live|recorded|offline`, `DATA_DIR`.
3. `blackbox/llm.py`. One async OpenAI-compatible client with:
   - a token-bucket rate limiter driven by the `x-ratelimit-remaining-*` headers;
   - retry with jitter on 429 errors, honouring `retry-after`.
4. SQLite schema (`blackbox/store/schema.sql`), WAL mode, one writer queue:
   - `runs(run_id, parent_run_id, fork_id, agent, agent_version, task_id, model, seed, mode, outcome, score, checker_reason, started_at, ended_at)`
   - `steps(step_id, run_id, addr, seq, kind, name, agent_role, request_key, input_hash, output_hash, reasoning_hash, state_before, state_after, reads_json, writes_json, cache_status, tokens_in, tokens_out, tokens_cached, latency_ms, finish_reason, error_type, retries)`
   - `edges(run_id, src_addr, src_pointer, dst_addr, dst_pointer, value_hash, key, version, kind)`,
     where pointers are RFC 6901 paths and kind is `state`, `message` or `inferred`
   - `cassette(request_key PRIMARY KEY, kind, response_hash, model, created_at)`
   - `forks(fork_id, base_run_id, branch_name, edits_json, mode, samples, reexec_steps, cached_steps, invalidated_steps, tokens_saved, ms_saved, fix_pass_rate, fix_ci_low, fix_ci_high, control_pass_rate, control_ci_low, control_ci_high, verdict)`
   - `labels(run_id, root_addr, fault_type, source, recovered, manifest_addr, verified)`, where source is `injected`, `natural_auto`, `human` or `verified`. **Kept out of every feature query.**
   - `diagnoses(run_id, model_version, ranking_json, conformal_set_json, abstain, evidence_json, created_at)`
   - `regression_exports(export_id, fork_id, test_hash, fixture_hash, created_at)`
5. API contract (`server/contract.md` + pydantic models) and matching mock JSON fixtures in
   `web/mocks/`: run list, run detail, diagnosis, fork events, diff, eval.
6. `.env.example`, Makefile targets (`setup`, `check`, `dev-api`, `dev-web`), and a CI job for
   web lint.

**Test:**
- `make check` passes.
- `python -m blackbox.llm ping` returns a reply from Groq.
- Creating the DB from the schema succeeds.
- The web app renders the mock fixtures.

---

### Task 2: Recorder SDK
**Owner:** P1 · **Est.:** 6h · **Reuses:** `blackbox/recorder.Store`, `canonical_json`

**Goal:** wrap any Python agent so that every step is recorded with exact inputs, outputs, state
and provenance.

**Build** (`blackbox/sdk/`):
- `bb.run(agent, task_id, seed)`: a context manager that opens a run and writes the outcome at
  the end.
- `bb.step(addr, kind)`: a decorator or context manager that groups the calls inside a logical
  step.
- `await bb.chat(messages, model, tools=None, response_format=None, **params)`:
  - normalizes the request (sorted JSON, tool-call IDs rewritten as `tc_{addr}_{i}`, floats
    canonicalized);
  - hashes it into a `request_key`;
  - calls the LLM through the memo layer (Task 3);
  - stores the response and reasoning in the blob store.
- `await bb.tool(fn, **args)`: the same for tools. The key is `H(name, tool_version, canonical args)`.
- `bb.State`: a dict-like proxy.
  - On read it records `key@version`. On write it bumps the version and records the write.
  - After each step it saves a **Merkle snapshot**, i.e. `{key: blob_hash}`, then hashes the
    sorted pairs. Unchanged keys are shared between snapshots, so a snapshot per step is cheap.
  - **Provenance edges** come from reads matched to the step that last wrote each key, plus the
    message IDs passed into `bb.chat`.
  - A leaf-value walker records the producer and consumer JSON Pointers for scalar values. If
    several producers are possible, store every candidate and mark the edge `inferred`; never
    invent a single origin.
- `bb.now()`, `bb.uuid()`, `bb.random()`: recorded values, so replays see identical time and
  randomness.
- Redaction hook applied before payloads are stored.
- Optional OpenTelemetry export (`gen_ai.*` attributes, `openinference.span.kind`) for viewing in
  Phoenix.

**Test** (`tests/test_sdk.py`):
- Record a 3-step toy agent. The DB has 3 steps, correct reads and writes, and 2 edges.
- Recording the same toy agent twice gives identical `request_key` and `state_after` hashes.
- Snapshot round trip: loading `state_after` of step 2 returns exactly the dict the agent had.
- A string matching the API-key pattern is masked in the stored blobs.
- In a nested tool payload, `/quote/total_inr` resolves to the producing step and exact output
  pointer; an ambiguous repeated value returns all candidates.

---

### Task 3: Replay engine (checkpointed replay and alternative execution core)
**Owner:** P1 · **Est.:** 7h

**Goal:** resume from any step's exact state, apply an edit, and re-run only the affected steps.

**Build** (`blackbox/replay/`):
- `replay(base_run_id, edits=[], mode="cone"|"prefix"|"full", samples=1, control=False) -> fork_ids`
- Every fork is an immutable named branch. **Fork with edit** and **Re-run unchanged** are separate
  commands, and neither mutates the base run.
- CLI parity: `blackbox fork <run> --step <addr> --patch <fix.json> --samples 5 --control`.
- **Edit types:**
  - `override_output(addr, value)`
  - `patch_tool_args(addr, args)`
  - `patch_prompt(addr, system_or_message_patch)`
  - `swap_model(addr, model)`
  - `patch_tool_result(addr, value)`
- **Effect resolution**, applied on every `bb.chat` / `bb.tool` during replay:
  1. If `addr` has an edit, apply it.
  2. In mode `full`, always call live.
  3. In mode `prefix`, call live if this step comes at or after the edit.
  4. If the exact `request_key` is in the base run's cassette, return the recorded response.
  5. Otherwise call live: this step is in the cone. Store the result under `request_key + sample_idx`.
- **Divergence check:** for steps before the first edit, the replayed `state_before` must equal
  the recorded one. Otherwise raise `ReplayDivergence(addr)`.
- **Samples and control:**
  - Run K samples of the edited fork. With `control=True`, also run K samples with no edit,
    resampled from the earliest edit point with paired seeds.
  - Require K ≥ 3; verification defaults to K=5. Store pass rates and Wilson 95% confidence
    intervals. K=3 is allowed to finish as **INCONCLUSIVE**.
- **Savings:** count re-executed vs cached steps, tokens and milliseconds. Save these to `forks`.
- **Replay state:** mark the static downstream cone `invalidated`, then resolve each step to
  `cached` or `live` at runtime. Keep lifecycle (`queued|running|done|diverged`) separate from
  cache state (`cached|invalidated|live|edited`).
- Emit progress events through an async callback. Task 10 turns them into SSE.

**Test** (`tests/test_replay.py`):
- Replaying with no edits gives **100% cached**, 0 LLM calls, and an identical outcome and final
  state hash.
- On a toy DAG `A → {B, C} → D`, editing B re-executes `{B, D}`, and C is cached.
- If the edited B produces the same output as before, D is cached (early cutoff).
- Changing a recorded tool output on disk raises `ReplayDivergence`.
- `prefix` mode re-executes everything after the edit. The test asserts that `cone` re-runs
  fewer steps than `prefix`.
- A near-match request never hits the cassette; only its exact canonical hash does.
- A K=5 fixture with fix 5/5 and control 0/5 is `VERIFIED`; overlapping intervals are
  `INCONCLUSIVE`.

---

### Task 4: TripCrew, the demo agent
**Owner:** P2 · **Est.:** 7h

**Goal:** a multi-agent trip planner whose branches are independent, so the "re-run only the
affected steps" behaviour is real and visible.

**Build** (`agents/tripcrew/`):
- **Mock APIs** (`mock_apis.py`) over a seeded synthetic DB:
  - `search_flights`, `search_hotels`, `get_weather`, `fx_rate`, `visa_rules`;
  - each response carries an `as_of` date;
  - `fx_rate` supports `fresh=true`.
- **Scenarios:** about 300, built from templates. Each one is an Indian origin city × a world
  destination × dates × travellers × a ₹ budget × constraints (vegetarian, refundable, no
  red-eye flights). A solver checks each scenario is feasible and precomputes the correct total.
- **Agent graph.** Every step goes through `bb.step`, and each worker sees only its own
  sub-task, so contexts stay scoped. The flow runs from left to right:
  - `planner/chat` parses the request into a constraints JSON.
  - Then 4 workers run in parallel: `flight/{chat,tool,chat}` ∥ `hotel/{chat,tool,chat}` ∥
    `weather/tool` ∥ `fx/tool` (+ `visa/tool`).
  - `budget/tool` is a Python calculator.
  - `writer/chat` drafts the itinerary.
  - `verifier/chat` reviews it.
  - `final` produces the plan.
  - That is about 16–18 steps in total.
- **Checker:** passes only if every hard constraint is met and the total in INR is within ₹1 of
  the solver's total. It returns a reason string.
- **Difficulty:** tune it so `gpt-oss-20b` passes about 70% of the time. We need passing runs to
  fork and some natural failures.

**Test:**
- 20 scenarios run end to end and the pass rate is logged. The checker agrees with a manual check
  of 5 runs.
- Replay test: editing `fx/tool#1` re-executes only {fx, budget, writer, verifier}. The flight,
  hotel and weather branches are cached.

---

### Task 5: HopRAG (second agent) and ShopDesk (stretch)
**Owner:** P2 · **Est.:** 4h, plus 4h for ShopDesk

**Goal:** a second architecture, so we can test whether the model generalizes to a different
kind of agent.

**Build:**
- **HopRAG** (`agents/hoprag/`):
  - 2–4-hop questions from MuSiQue-Ans;
  - tools: `search(query)` (BM25 over each question's 20 paragraphs, with `rank_bm25`), `read(doc_id)`
    and `answer(text)`;
  - the agent is a sequence of decompose → search/read per hop → answer;
  - the checker uses exact match or F1 ≥ 0.8 against the answer aliases;
  - the gold sub-answers are stored for oracle fixes.
- **ShopDesk** (stretch):
  - a single-agent ReAct loop over the tau2-bench retail environment (tools + DB imported, our
    own instrumented loop, cached user simulator);
  - the checker is tau2's DB-hash reward;
  - airline tasks are held out.

**Test:**
- 50 HopRAG questions recorded, with the pass rate logged.
- The replay no-edit test passes on HopRAG.
- ShopDesk (if built): 20 tasks recorded.

---

### Task 6: Fault Forge and the dataset
**Owner:** P2 · **Est.:** 8h (generation runs in the background)

**Goal:** thousands of failed runs where we know exactly which step caused each failure, plus
hard negatives.

**Build** (`blackbox/forge/`):
- **Fault operators** (`operators.py`). Each declares which field it changes and applies to
  specific step kinds.

  | Family | Seen in training | **Held out (unseen)** |
  |---|---|---|
  | Tool | T1 wrong value, T3 empty/404, T4 timeout/500 | **T2 stale data**, **T5 unit/schema drift** |
  | Retrieval | R1 irrelevant documents | **R2 poisoned fact** |
  | Decision | D1 wrong arguments, D2 wrong tool, D4 stops too early | **D3 hallucinated intermediate value** |
  | Coordination | C1 instruction misread, C4 repeated-step loop | **C2 constraint dropped in a hand-off between agents**, **C3 state corruption** |

- **Fork and perturb** (`inject.py`). Each fork is one call to the Task 3 replay engine:
  1. Pick a passing run and a step k at a random position.
  2. Apply the operator to step k.
  3. Resume.
  4. Check the outcome.
  5. Run a paired no-edit control for every accepted label. During exploration use K=1; before
     dataset freeze, re-check every kept example with K=3 or mark it low-confidence.
- **Labels** (`label.py`):
  - **POSITIVE:** the edited fork reproducibly fails and the paired control passes. Root = k.
  - **RECOVERED:** the fork still passes. This is a hard negative.
  - **FLAKY:** the control fails. Discard the fork and report the rate.
  - Also store the **manifestation step**: the first step whose action differs from the
    original run after the fault.
- **Distractors:** in 30% of positive forks, also inject a recoverable fault at another step
  (T4 retried, or a C4 loop that recovers). A separate control must show that the distractor
  alone does not cause failure. This teaches the model the difference between "suspicious" and
  "causal", which is the step-9 moment in the demo.
- **Anti-cheating rules:**
  - Decision faults use **ghost hints**: a hidden instruction is added to the step's input, the
    LLM writes the wrong action in its own style, and the hint is **not** stored.
  - Tool faults stay schema-valid and plausible: replacement values are sampled from the real
    distribution.
  - Fault positions are randomized.
  - An assertion checks that the forked state differs from the base run only in the declared
    fields.
- **Natural failures** (`natural_label.py`). These come from base runs that failed with no
  injection.
  1. gpt-oss-120b gets the gold answers (the solver output for TripCrew, sub-answers for HopRAG).
  2. It proposes a minimal fix at each candidate step, earliest first, and replays.
  3. Screen candidates at K=3, then run K=5 on the earliest promising step. It becomes the
     candidate label only if its edited lower bound beats the paired control's upper bound; if
     bounds still overlap, keep it as `natural_inconclusive`, not truth.
  4. If no step flips it, the run is marked "unattributable".
  5. These runs are **test-only**.
- **Bulk runner** (`make forge`): async workers with a concurrency cap, resumable, with progress
  logging.

**Free-first data targets** (scale only while the measured free quotas allow it):

| | TripCrew | HopRAG | ShopDesk* |
|---|---|---|---|
| Base runs | 80–120 | 40–80 | stretch only |
| Accepted fault forks | 240–360 | 120–240 | stretch only |
| Natural failures (test only) | 25–40 | 20–35 | stretch only |

The must-ship floor is **360 accepted injected failures, at least 20 examples per evaluated fault
family, and 45 natural failures**. The target is 600 accepted failures if free compute allows it.
Report the actual N everywhere; wider confidence intervals are an honest limitation, not a reason
to buy compute or hide the result. Reuse passing base-run cassettes, prefer deterministic tool
faults, run LightGBM locally, and use free Groq only for the calls the replay cone genuinely
invalidates. **Freeze the dataset** at a fixed hour by exporting Parquet and saving its hash in
`data/DATASET_VERSION`.

**Test:**
- Unit test per operator: only the declared field changes.
- 20 forks per operator with outcome counts printed.
- The flaky rate is below 10%.
- Every POSITIVE label has a root address that exists in the run.
- Every frozen label has paired seed IDs, reproduction counts and control counts.
- The natural labeller reproduces the known root on 10 injected runs given to it as if they were
  natural.

---

### Task 7: Features, splits and baselines
**Owner:** P3 · **Est.:** 6h · **Reuses:** `blackbox/eval.localization_metrics`

**Goal:** turn each step into a row of numbers, without leaking the answer.

**Build** (`blackbox/ml/features.py`, built with DuckDB from the Parquet exports). The groups
below are about 90 features, one row per step, and each group can be switched off for ablation.

| Group | Examples | Why it helps |
|---|---|---|
| A. Structure | position (relative), step kind, tool name, graph depth, number of downstream steps, on the path to the final answer, repeat count | Where the step sits in the run |
| B. Telemetry | latency, tokens, retries, errors, `finish_reason=length`, each z-scored against passing runs of the same tool | Unusual behaviour |
| C. Validity | arguments/output schema valid, NaN or out-of-range values, unit tokens; AgentRx-style schema rules such as empty query, swallowed error and repeated identical call | Broken data with deterministic evidence |
| D. Grounding | share of numbers and entities in the output that appear in no input, conflicts with earlier values, embedding drift from the task | Hallucinated or contradictory values |
| E. Surprisal | Qwen3-0.6B negative log-likelihood (mean, max, p90, numbers only, change from previous step). Computed on Kaggle and cached by output hash | Groq gives no logprobs, so we compute our own |
| F. Novelty | IsolationForest score, kNN distance and Mahalanobis distance vs **passing training runs only** | "This doesn't look like a good run" |
| G. Context | the same signals at steps i−1 and i+1, errors appearing after this step, "first anomaly in the run" | Symptoms show up after the cause |
| H. Lineage | number of downstream consumers, reaches the final answer, later overwritten, causal hop distance, field-level fan-out | How far the damage spread |
| I. Embedding | 16-dimensional PCA of a MiniLM embedding of the output | Semantic signal |

- Within-run percentile copies of the key features. This helps the model transfer to new agents.
- Feature-group dropout: randomly blank the telemetry and validity groups in 30% of training
  runs, so the model also works on text-only traces like Who&When.
- **Splits** (`blackbox/eval/splits.py`). Splits are always grouped by `task_id` and source run,
  so forks of one run never land in two splits.

  | Split | Purpose |
  |---|---|
  | S0 | Known fault types on new tasks |
  | **S1** | **Unseen fault types** (the held-out column above) |
  | S3 | **Unseen agent**: leave one agent out |
  | S4 | **Natural failures** |
  | S5 | **Who&When** zero-shot, using a text-only adapter (`whowhen_adapter.py`) |

- **Baselines** (`blackbox/eval/baselines.py`):
  - random step, last step;
  - first step with an error;
  - position-only model;
  - IsolationForest maximum;
  - LLM judge (gpt-oss-120b) in the three Who&When styles: all-at-once, step-by-step and
    binary search.
- **Observability views** are materialized from the same frozen traces so comparisons are paired:
  - `full`: every Black Box feature;
  - `otel`: standard span timing/status/name/attributes, with field provenance and snapshots removed;
  - `outputs_only`: ordered step names and outputs only.
- **Graph views** are also paired:
  - `full_edges`: field provenance + protocol/time-order edges;
  - `protocol_edges`: call/sequence/agent edges only;
  - `no_edges`: all graph-derived features blanked.

**Test:**
- An assertion fails the build if any `labels` column, fault type or run ID appears in the
  feature matrix.
- No `task_id` appears in two splits, and S1 fault types are absent from training.
- Baselines produce numbers on S0 and S1.
- The three observability views and three graph views contain identical run IDs and labels; only
  allowed feature groups differ.

---

### Task 8: Train and evaluate the diagnoser
**Owner:** P3 · **Est.:** 8h

**Goal:** a model that, given a failed run, ranks its steps by blame, with honest numbers.

**How training works:**
1. **Training examples.**
   - Every failed run in the training split is one group, and each of its steps is one row.
   - Label: root step = 3; a step ±1 causal hop on the same provenance path = 2; ±2 hops = 1.
     Everything else = 0, including distractor steps.
2. **Model.** `LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03,
   num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, lambdarank_truncation_level=5)`.
   - It learns to put the root step at the top of each run's list.
   - Training uses early stopping on validation ndcg@1.
   - It takes seconds on a laptop CPU.
3. **Probabilities.**
   - A softmax over each run's step scores, with temperature T fitted on validation.
   - Result: e.g. "step 7: 82%".
4. **Run-level detector.** An `LGBMClassifier` on run aggregates (passing + recovered runs vs
   failed). It powers the risk column in the runs list.
5. **Conformal sets.** Calibrated on validation, so we can say "the root is in {6, 7, 9} with
   90% empirical coverage". If the top-1 margin is small or the set exceeds 3 steps, the UI
   abstains with "no confident culprit" instead of presenting a top-1 as fact.
6. **Optional GNN** (stretch):
   - GATv2 over the step graph, with edge types for time order, data dependency, same agent and
     parent call;
   - added to an ensemble only if it improves validation.
7. **Saved artifacts** (`data/models/diagnoser-v1/`): model file, feature list, T, the conformal
   threshold, the dataset hash and the git commit.
8. **Retraining loop.** Each time the verifier marks a root cause VERIFIED (Task 9), it becomes a new
   `verified` label. `make train` picks it up next time, adding it to the training split only.

**Evaluation** (`make eval` → `data/eval/*.json`, which the Results page reads):
- **Metrics:**
  - Top-1, Top-3 and ±1-step accuracy, MRR;
  - run-level AUROC;
  - blame on distractors (should be low);
  - Actionable@1: fixing the top suspect flips the run;
  - replay savings;
  - latency per trace.
- **Tables:**
  1. Leaderboard × splits S0/S1/S3/S4/S5, with 95% bootstrap confidence intervals over runs.
  2. Ablation per feature group.
  3. **Recorder-value ablations:** full provenance vs protocol-only vs no edges; full record vs
     OTel-style view vs outputs-only. Show paired bootstrap differences, not just bars.
  4. Integrity checks:
     - **artifact audit**: a surface-only classifier that tries to find injected steps should
       score AUROC ≤ 0.65;
     - **shuffled labels**: should score about random;
     - **human agreement**: κ on 40–60 runs labelled in the UI.
- **Expected ranges** (targets, not promises):

  | Split | Top-1 |
  |---|---|
  | S0 | 0.75–0.90 |
  | **S1 (unseen faults)** | **0.40–0.60** (above 0.85 means something is leaking) |
  | Who&When (algorithm-generated subset) | Report measured value; literature pipelines span roughly 0.29–0.53, but are not directly comparable until splits match |

- The Who&When result is computed **once**, after the model is frozen.

**Test:**
- Beats the first-error and LLM-judge baselines on S1, with confidence intervals reported.
- Produces both recorder-value ablations; if full provenance does not win, report that result and
  do not claim the recorder itself caused the gain.
- The shuffled-label run scores about random.
- The artifact audit is ≤ 0.65 (or the fix is documented).
- `make eval` regenerates every JSON file from scratch.

---

### Task 9: Explanation, verifier, trace comparison and regression export
**Owner:** P3 (explain, verifier) + P1 (diff, export) · **Est.:** 9h split across both owners

**Goal:** turn the ranking into evidence and proof, and compare any two runs.

**Build:**
- **Reasons** (`explain/reasons.py`):
  - TreeSHAP (`pred_contrib=True`) gives the top 3 features for the suspect.
  - A template table maps each feature to a sentence. For example, `fresh_age_z` becomes
    "Data is 7 months older than in passing runs."
  - Deterministic schema-rule failures are evidence lines, not model prose. Every line cites
    `{addr, json_pointer}` and the UI highlights that exact field.
- **Path of damage:** walk the provenance edges forward from the suspect to the final answer.
  Every step on the path is tagged as root, symptom or unaffected.
- **Similar past cases:**
  - kNN (k=12) over the root steps in the training data;
  - output looks like: "Looks like 9 of 12 past stale-data failures. `fresh=true` fixed 7 of
    them."
  - The same lookup picks the fault type and the fix template.
  - For failed runs, find the nearest passing twin by agent, task embedding and graph shape. It is
    the preferred source of a known-good replacement value and a side-by-side comparison.
- **Verifier** (`explain/verifier.py`):
  1. Take the top 3 suspects.
  2. Prefer a schema-derived fix, oracle value or nearest-passing-twin value. A free-tier LLM
     proposal is optional and never sufficient evidence by itself.
  3. Run paired `cone` replay and no-edit control with identical seeds, K=5 by default.
  4. Compute fix/control pass rates and Wilson 95% intervals.
  5. Mark **VERIFIED** only when fix LCB > control UCB. Mark **REFUTED** only when a known-good
     intervention propagated but reproducibly did not improve the outcome. Otherwise mark
     **INCONCLUSIVE** and show what additional samples or edit quality would resolve it.
- **Narrative** (optional):
  - gpt-oss-120b turns the evidence bundle into an incident-report-style summary, using a strict
    JSON schema of claims, each with `cites: ["fx/tool#1", …]`;
  - a validator rejects unknown addresses and numbers that are not in the bundle;
  - after one failed retry, it falls back to a template text.
- **Diagnosis object:** `{ranking, conformal_set, abstain, reasons, damage_path, precedents, verification, narrative}`,
  saved in `diagnoses`.
- **Trace comparison** (`blackbox/diff/`):
  1. Align the two runs by address. If the control flow changed, use LCS over
     `(kind, name, args_hash)` instead.
  2. Classify each step as same, cached, changed, new or removed.
  3. Find the first divergence.
  4. Compute a per-key state diff using the snapshot hashes.
  5. Compute the outcome diff (checker reason before and after).
  6. Also provide "nearest passing run of the same task" for any failed run.
- **Regression exporter** (`blackbox/export/regression.py`):
  1. Accept only a `VERIFIED` fork.
  2. Emit `tests/regressions/test_<run>_<addr>.py` plus a minimal cassette/fixture directory.
  3. The generated test runs with `MODE=recorded` and network disabled.
  4. Assert the original fixture fails, the patch passes, the state hash before the edit matches,
     and the observed invalidation cone equals the saved cone.
  5. Include source run/fork IDs, dataset/model hashes and a human-readable patch in comments.

**Test:**
- On 30 S0 test runs, report VERIFIED/REFUTED/INCONCLUSIVE counts separately; at least 60% of
  known roots should be VERIFIED and no more than 5% of wrong roots may be falsely VERIFIED.
- Every narrative claim cites an address that exists.
- Diffing a run against itself gives all "same".
- Diffing the demo fork gives first divergence = `fx/tool#1`, the outcome flips, and only cone
  steps are marked changed.
- A generated regression test passes with network access blocked and fails if its saved fix is
  removed.

---

### Task 10: API and local MCP server
**Owner:** P1 · **Est.:** 7h

**Goal:** one backend the UI talks to, which streams replays live.

**Endpoints** (FastAPI, pydantic models from Task 1):

| Method | Path | Returns |
|---|---|---|
| GET | `/runs?agent=&outcome=&split=&q=` | Paged list: status, agent, task, steps, duration, cost, risk, top suspect |
| GET | `/runs/{id}` | Run detail + steps + edges |
| GET | `/runs/{id}/steps/{addr}` | Full step: input, output, reasoning, state before/after |
| GET | `/runs/{id}/provenance?pointer=` | Producer step/pointer candidates and consumer path for one value |
| GET | `/runs/{id}/diagnosis` | Diagnosis object. Cached; computed on first request |
| POST | `/runs/{id}/verify` | Starts the verifier and returns a `job_id` |
| POST | `/forks` | `{base_run_id, edits, mode, samples, control}` → `fork_id` |
| GET | `/forks/{id}/stream` | **SSE** events (below) |
| GET | `/diff?a=&b=` | Aligned comparison |
| POST | `/forks/{id}/export-test` | Generated pytest + cassette fixture for a VERIFIED fork |
| GET | `/eval` | Precomputed evaluation JSON |
| POST | `/labels` | Human label (Label mode) |
| POST | `/v1/traces` | OpenTelemetry trace ingest. Diagnosis and comparison only for external agents |

**SSE events:**
- `step`: `{addr, phase: queued|running|done|diverged, cache_status: cached|invalidated|live|edited, ms, tokens}`
- `outcome`: `{sample, passed, reason}`
- `summary`: `{reexecuted, cached, tokens_saved, ms_saved, fix_rate, fix_ci, control_rate, control_ci, verdict}`

The browser connects directly to FastAPI (with CORS), not through a Next.js proxy, so events
are not buffered. In `MODE=recorded`, live calls are refused and only cassette hits are allowed.

**Local MCP server** (`blackbox/mcp_server.py`, `stdio` transport) calls the same Python service
functions as FastAPI; it does not call HTTP or duplicate business logic:

| Tool | Result |
|---|---|
| `get_suspects(run_id)` | Ranked/calibrated suspects, abstain state and evidence citations |
| `get_step(run_id, addr)` | Step payload, field provenance and damage path |
| `fork_and_verify(run_id, addr, patch, samples=5)` | Immutable edited/control forks, savings, intervals and verdict |
| `export_regression_test(fork_id)` | Paths and hashes for the generated offline pytest fixture |

Use MCP Inspector for the $0 judged demo. A Claude Code or other coding-agent integration is an
optional recording only if the team already has access; it is not a dependency.

**Test:**
- `pytest server/` passes.
- `curl -N /forks/{id}/stream` shows `step`* → `outcome` → `summary` in order.
- In recorded mode, a fork that would need a live call returns a clear error instead of hanging.
- MCP contract tests invoke all four tools over `stdio`; the verifier response matches the REST
  response byte-for-byte after canonical JSON serialization.

---

### Task 11: UI part 1, the app shell, Runs and Investigate
**Owner:** P4 · **Est.:** 10h (starts on mocks after Task 1)

**Goal:** a clean screen that answers "what failed, and which step caused it?" within seconds.

**Design system** (set up first, used everywhere):
- **Theme:** one polished dark "cockpit" theme for the 48-hour build. Near-black background,
  two surface levels, 0.5px borders, no gradients or glow. Light theme is post-hackathon.
- **Colour has meaning and nothing else:**

  | Colour | Used for |
  |---|---|
  | Orange (#FF4F00) | The suspect and the single primary action |
  | Red | Failed |
  | Green | Passed |
  | Amber | Re-running |
  | Grey | Cached / unaffected |

  Colour is always paired with an icon and a text label, so the UI works for colour-blind users.
- **Fonts:** Inter for text, JetBrains Mono for values, hashes and code.
- **Layout:** 8px spacing grid. Smallest supported screen is 1366×768.
- **Every term has a tooltip** ("cached", "cone", "control").
- **Keyboard shortcuts:** `j`/`k` next and previous step, `f` fork and fix, `c` compare, `/`
  search, `?` shortcut help.
- **Empty, loading (skeleton) and error states** on every panel. Errors say what to do next.

**Screens:**
1. **App shell.**
   - Left nav: Runs, Results, Label.
   - Top bar: agent picker, search, and a **LIVE / RECORDED** badge so viewers know whether
     calls are real.
2. **Runs.**
   - A virtualized table with columns: status, run ID, agent, task summary, steps, duration,
     cost, risk score, top suspect (e.g. "fx_rate · 82%"), time.
   - Filter chips: agent, outcome, split.
   - **Group by failure type**, e.g. "14 runs fail like this", so a team can triage many
     failures at once.
   - Clicking a row opens Investigate.
3. **Investigate**, the main screen, in three panes:
   - **Header:** task, outcome badge, the checker's reason in plain words ("Total ₹1,31,400
     exceeds budget ₹1,20,000"), and buttons *Compare with a passing run* and *Fork and fix*.
   - **Left, the step timeline.**
     - One row per step: number, icon by kind (LLM / tool / retrieval / state), name, duration,
       a **suspicion bar**, and chips (error, recovered, cached/live/invalidated during replay).
     - Mark the **visible failure** separately from the **responsible suspect** so a late symptom
       is never visually confused with the proposed root.
     - A toggle switches between **List** and **Graph**. Graph is a React Flow DAG, coloured by
       suspicion, with provenance edges.
     - Hovering a step highlights what it depends on and what depends on it.
     - ELK/dagre coordinates are computed once per base run and reused by every fork; repeated
       loops collapse into a stable group, so replay animation never shifts the graph.
   - **Centre, the step inspector.** Tabs:
     - *Input*: the exact prompt messages or tool arguments.
     - *Output*.
     - *State*: the before/after diff with changed keys highlighted.
     - *Reasoning*: the model's reasoning text.
     - *Raw*: JSON.
     - Every scalar renders as a `ValueCell`. Clicking it calls the provenance endpoint, jumps to
       the producer step, opens the right tab and highlights the exact source field. Ambiguous
       origins open a short chooser instead of guessing.
     - Copy buttons on every tab.
   - **Right, the diagnosis panel.**
     - A four-stage rail: **Localize → Attribute → Propose → Verify**.
     - *Suspect steps* list: step, confidence, and "likely among 6, 7, 9" when uncertain. When
       calibration abstains, the headline is **No confident culprit**.
     - *Why*: 3 plain-language reasons.
     - *Path of damage*: a mini graph.
     - *Seen before*: similar past cases.
     - *Proof*: **VERIFIED / REFUTED / INCONCLUSIVE**, fix and control rates, 95% intervals and K,
       or a **Verify** button.
     - Each item links to the cited step and exact field.
4. **Crash report.** A printable page with these sections:
   - Synopsis
   - Sequence of events
   - Probable cause
   - Contributing factors
   - Findings (every claim has a clickable step chip)
   - Recommended fix
   - Verification

   Export to Markdown. PDF export is post-hackathon.

**Test:**
- With the real API, any failed run from the Runs list opens Investigate within 1s.
- The suspect is highlighted, and clicking each evidence/value chip jumps to the right step and
  exact field.
- The graph node coordinates do not change between original, control and edited forks.
- Works at 1366×768 in the dark theme.
- The keyboard shortcuts work.

---

### Task 12: UI part 2, Fork and fix, Compare, Results, Label
**Owner:** P4 (P1 helps with SSE) · **Est.:** 10h

**Screens:**
1. **Fork and fix**, a drawer opened from the suspect or any step.
   - Two explicit, non-destructive actions: **Fork with edit** and **Re-run unchanged**. The
     unchanged branch is the control and is never hidden behind an advanced menu.
   - **Left:** the original value, read-only.
   - **Right:** a Monaco editor pre-filled with the **suggested fix**.
   - **Edit type selector:** change output / tool arguments / prompt / model.
   - **Replay mode:** *Smart* (cone, the default), *From this step* (prefix) or *Full*. It shows
     a **prediction before running**, e.g. "Will re-run 4 of 18 steps, about 2s."
   - **Samples:** 3/5/10; K=5 is the default. *Include paired control* is on and required for a
     verdict; turning it off labels the result `preview`, never VERIFIED.
   - **During the run, the graph animates:**
     - cached steps fade to grey;
     - invalidated steps receive an outlined warning badge;
     - live steps pulse amber;
     - finished steps turn green or red;
     - a live counter bar shows *steps re-run · LLM calls saved · tokens saved · time*.
   - **Result card:**
     - before → after outcome;
     - pass rate with edit vs without (control), e.g. "5/5 vs 0/5", both 95% intervals and the
       **VERIFIED / REFUTED / INCONCLUSIVE** verdict;
     - buttons: *Compare runs*, *Save verified fix*, and **Export as regression test**. Save/export
       are disabled unless the verdict is VERIFIED.
   - **Fork timeline:** original → unchanged control → each edited fork, with score, pass rate,
     verdict and savings. Switching branches preserves the graph layout and inspector selection.
2. **Compare.**
   - Two aligned columns, one row per step.
   - Row colours: same, cached, changed, new or removed.
   - A **first divergence** marker.
   - Clicking a changed row opens a Monaco side-by-side diff of input and output.
   - A state diff table, an outcome diff at the bottom, and a *Show only changes* toggle.
3. **Results** (reads `/eval`):
   - **Headline cards:** unseen-fault top-1 accuracy vs the best baseline, natural-failure
     top-1, replay savings, ms per trace.
   - **Leaderboard** with confidence-interval whiskers.
   - **Generalization matrix** (method × split heatmap).
   - Two required paired ablation charts: **full provenance / protocol-only / no edges** and
     **full Black Box record / OTel-style fields / outputs-only**.
   - Feature-group ablation bars, a reliability/abstention chart, and the integrity-checks panel
     (artifact audit, shuffled labels, κ).
   - Every number shows its sample size and a tooltip explaining it.
4. **Label mode.**
   - Shows a failed run with the model's guess hidden. The annotator clicks the step they think
     is the root and saves.
   - Two teammates label the same 40–60 runs, which gives κ.

**Test:**
- The full demo flow (Runs → Investigate → Fork and fix → result → Compare → Results) works
  without touching code.
- The SSE animation matches the server events.
- Results numbers match `data/eval/*.json`.
- A recorded 5/5 edited vs 0/5 control demo shows VERIFIED; a noisy fixture shows INCONCLUSIVE;
  the UI never renders `confirmed` or an unsupported binary verdict.
- Exported pytest appears only for VERIFIED forks and passes with network disabled.
- A first-time user, e.g. a friend from another team, can find the culprit and test a fix in
  under 2 minutes without help. Note where they get stuck and fix it.

---

### Task 13: Deploy, offline mode and hardening
**Owner:** P1 · **Est.:** 5h

**Build:**
- **Docker Compose** (`make demo-offline`): API + SQLite + model + static web build, with
  `MODE=offline` pointing at the configured Ollama model. Works with Wi-Fi off; use
  `gpt-oss:20b` only on a machine that fits it.
- **Public deploy:**
  - web on Vercel Hobby;
  - API on Render Free with a curated, read-only `blackbox.db`, blobs and model baked into the
    image. Runtime writes go to temporary storage and may disappear without affecting the demo;
  - `MODE=recorded`, no Groq key.
  - demo controls, fixes, verdicts and regression exports are pre-recorded immutable artifacts;
  - a static JSON fallback ships with the web build if the free API is waking from a cold start.
- **Demo scenarios:** pick 3 failed runs (TripCrew stale FX, a HopRAG poisoned fact, a natural
  failure). Each fix must flip the outcome in **10 of 10** re-runs. Record their cassettes.
- **Fallbacks:**
  - warm the Render URL before judging; if it is cold or unavailable, switch visibly to the
    static recorded data without pretending it is live;
  - a 90-second screen recording of the whole demo.
- **Operations:** restrict CORS to the web origin, add a simple rate limit on `POST /forks`, and
  a health check endpoint.

**Test:**
- With Wi-Fi off, `make demo-offline` runs the full demo.
- The public URL works in an incognito window on a phone and a laptop.
- Restarting the Render service loses no required demo state because every judged artifact is in
  the image or static web fallback.
- No secrets appear in the deployed environment or the repo (`git grep -i groq_api_key` finds
  only `.env.example`).

---

### Task 14: Demo and pitch
**Owner:** P4 + all · **Est.:** 4h

- **Deck** (9 slides + backup):
  1. Title
  2. Problem: "AI agents don't crash. They just quietly give wrong answers." One failure can
     hide in 200+ spans, and even the newest LLM localization pipelines are roughly 29–53% on
     their reported setups while requiring many model calls and providing no replay proof. Do
     **not** reuse the obsolete blanket claim "LLM judges get 14.2%."
  3. Insight: record → learn → replay; the replay engine also produces our training data.
  4. Architecture
  5. Live demo
  6. Data: 2 agents × 14 fault types, held-out fault families and natural test-only failures
  7. Results with confidence intervals, abstention, edge ablation and telemetry-sufficiency ablation
  8. Replay savings
  9. Impact: export a verified crash as an offline regression test; local MCP lets a coding agent
     inspect and repair it
  10. Backup: threats to validity
- **Demo script (4:00):**
  - **0:00–0:20, hook:** "Every aircraft carries a black box. Your AI agents carry nothing."
  - **0:20–0:50, the crash:** run TripCrew live; it fails.
  - **0:50–1:30, investigate:** visible failure vs responsible step; click the stale rate value
    to jump to its producer; show the recovered step 9 as not causal.
  - **1:30–2:25, fork and fix:** original → K=5 control → edit; 4 of 18 steps invalidate, exact
    cache hits settle to grey, live steps run, and the verdict shows "5/5 vs 0/5 · VERIFIED".
  - **2:25–3:10, rigor:** Results page, unseen faults vs the LLM judge, confidence intervals,
    abstention, edge/telemetry ablations and natural failures.
  - **3:10–3:40, a second agent:** a natural failure, caught by the same model.
  - **3:40–4:00, close:** export the verified fork as a network-free pytest. "Black Box finds
    the step, shows where its value came from, and proves the fix without re-flying the flight."
- **Q&A sheet:**
  - "vs LangSmith/Langfuse/Laminar": they provide excellent trace and replay UX. We train the
    localizer, invalidate only the dependency cone, and verify edits against a control.
  - "vs AgentDebugX": its diagnose → attribute → recover → rerun pipeline is closest; our claim
    is the trained localizer, unseen-family/natural evaluation and confidence-bounded proof.
  - "Nondeterminism?": exact-hash cassettes, paired K samples, unchanged controls, intervals and
    an INCONCLUSIVE state.
  - "Why not just ask an LLM?": show our measured latency/call count and verdict evidence. Do not
    claim a 1000× advantage unless the experiment in this repo actually measures it.
  - "Synthetic data?": cite Who&When Pro / TraceElephant, then show held-out fault types, κ, the
    artifact audit, the separate natural-failure result and the injected-only vs natural gap.
- **Citation/source ledger:** canonical links and exact versions for AgenTracer, DoVer, CAR,
  GCJR, GraphTracer, MASPrism, conformal attribution, Who&When Pro, TraceElephant, DeFA,
  BranchPoint-Latent, TelemetrySuffBench, Repair or Resample?, AgentLens and AgentRx. Attribute
  the HN quotes to their original threads; X snippets have no engagement-count claims.
- **Rehearse 3 times** with a timer. One person drives the demo, one presents.

**Test:** a full run-through under 4:00, done 3 times, including once on the offline stack.

---

## Part E: How a real developer uses it

1. **Install and instrument (about 5 minutes).** `pip install` the package, wrap LLM calls with
   `bb.chat` and tools with `bb.tool`, and keep shared state in `bb.State`.
   - Agents that already emit OpenTelemetry traces can send them to `/v1/traces` instead and get
     diagnosis and comparison, but no replay.
2. **Run the agent as usual.** Every run is recorded locally. There is no extra cost beyond the
   agent's own calls.
3. **When a run fails:**
   - open Runs, then the failed run;
   - read the suspect set, click any evidence value to inspect its origin, and accept an abstention
     when the model is not confident;
   - fork with an edit and run the paired unchanged control;
   - read VERIFIED / REFUTED / INCONCLUSIVE with pass rates and intervals, then Compare;
   - for VERIFIED fixes, export a network-free regression test.
   - Typically under 2 minutes, compared with reading hundreds of log lines.
4. **For a team:**
   - group failures by type to fix the most common cause first;
   - save verified fixes, which also improves the model;
   - run exported crash regressions in CI with no model/API calls;
   - let a coding agent call the same workflow through the local MCP server.

**Why it is practical:**
- Diagnosis runs on a laptop CPU in milliseconds and costs nothing.
- Replay only pays for the cone.
- Verification spends K samples deliberately and always shows the unchanged control cost.
- Data stays local.
- Secrets are redacted before storage.
- Recorded mode lets you share a run without sharing keys.

---

## Part F: Schedule (48h) and cut lines

| Hours | P1 Infra | P2 Agents / data | P3 ML | P4 UI / pitch |
|---|---|---|---|---|
| 0–3 | Task 1 | Task 1 review, mock APIs | Who&When adapter, baselines skeleton | Task 1 mocks, design system |
| 3–10 | Task 2 including field provenance | Task 4 (TripCrew) | Feature code + schema rules on early traces | Task 11 shell + Runs |
| 10–16 | Task 3 immutable forks, exact cache, replay badges | Task 4 tests, small HopRAG seed | Task 7 | Task 11 Investigate + value jumps |
| **16** | **Gate 1:** click a value → producer, fork → exact cone replay works on TripCrew | | | |
| 16–24 | Task 10 REST/SSE + MCP skeleton | Task 6 operators + free-first generation | Task 7 splits + paired edge/telemetry views | Task 12 Fork and fix (mock SSE) |
| 24–32 | Task 9 diff + regression exporter | Natural labels, κ labelling, **freeze data at h30** | Task 8 + required ablations | Task 12 Compare + Results + branch timeline |
| **32** | **Gate 2:** real data → model → diagnosis → fork → compare, end to end in the UI | | | |
| 32–40 | Task 10 MCP contract, Task 13 | Demo controls/fixes (10/10), source ledger | Task 9 K=5 verifier + abstention, final eval | Verdict UI, regression button, corrected deck |
| **40** | **Feature freeze:** bug fixes only | | | |
| 40–48 | Free deploy, offline/MCP test | Backup scenarios | Q&A numbers + claim audit | Rehearsals |

**MVP (must ship):**
- Tasks 1–4 and 6–12 with TripCrew + a small HopRAG evaluation.
- LightGBM.
- Splits S0, S1 and S4.
- Field-level provenance jumps; three verdicts + abstain; cached/live/invalidated replay; paired
  control; both recorder-value ablations; regression export; four local MCP tools.
- Free recorded-mode deploy plus local live/offline demo.

**Stretch:**
- ShopDesk (S3 leave-one-agent-out with 3 agents).
- GNN.
- Full Who&When adapter beyond a small compatibility sample.
- LLM narrative.
- Fleet fix validation.
- Light theme and PDF crash-report export.

**24-hour finale version:**
- drop ShopDesk, the GNN and surprisal;
- keep the three verdicts, unchanged control, provenance jump and regression export; cut MCP UI
  demo first, while retaining its contract tests;
- start data generation by hour 8;
- freeze data at hour 18;
- feature freeze at hour 20.

---

## Part G: Risks

| Risk | Plan |
|---|---|
| Groq free limits or model changes | Never upgrade for the judged path. Read live limits, enforce a hard token/request budget, keep model IDs in config, use exact cassettes, local Ollama/Kaggle when available, and shrink N transparently |
| Replay isn't deterministic | Exact-hash cassette for unaffected steps; paired K samples + control; three-way verdict; demo cases verified 10/10 |
| The model learns injection tricks instead of real patterns | Ghost hints, plausible values, the artifact audit, held-out fault types, natural-failure test set |
| Data generation runs late | Data freeze hour; LightGBM trains in seconds; evaluation JSON is precomputed |
| Free hosting loses SQLite writes or cold-starts | Public image is immutable and recorded; static JSON fallback; warm Render; all real writes happen locally |
| Free GPU is unavailable | Surprisal is one removable feature group; run the no-surprisal ablation locally and keep the rest of the evaluation |
| Demo Wi-Fi fails | Local recorded/offline stack, backup video, phone hotspot |
| Too much to build | Follow the gates. Cuts are predeclared: light theme, PDF, narrative, ShopDesk, GNN and scale. Never cut the control, verdict honesty or end-to-end flow |
