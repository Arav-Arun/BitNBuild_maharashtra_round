# Black Box: build plan

Black Box is a flight recorder for AI agents. It records every step an agent takes. When a
run fails, it points to the step that most likely caused the failure, shows why with evidence
from the recording, and lets you fix that step and re-run from exactly that point. Only the
steps the fix actually affects are re-run.

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
 FastAPI (REST + server-sent events)  ──►  Next.js UI: Runs · Investigate · Report · Fork and fix · Compare · Results
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
| UI | Next.js (App Router), Tailwind, shadcn/ui, @xyflow/react (graph), Monaco (diff and edit), Recharts (charts), TanStack Table and Virtual |
| Tracing interoperability | OpenTelemetry GenAI span names and attributes, so traces can be exported to Arize Phoenix |
| Deploy | Vercel (web) + Render/Railway/Fly.io (API with a disk) + Docker Compose (offline demo) |

---

## Part B: External tools and access

### B1. Accounts and keys the team needs

| Tool | Why we need it | Who | Access / secret | Cost |
|---|---|---|---|---|
| **Groq** | Running agents and generating data; the LLM-judge baseline; fixes and narratives | One org owner; everyone else uses a project key | `GROQ_API_KEY` in `.env` only. Upgrade to the **Developer tier** and set a spend limit. The free tier is capped at about 200K tokens/day per model, which is too small | About $10–35 total |
| **GitHub** | Repo and CI | All 4 | Write access to this repo | Free |
| **Hugging Face** | Download the Who&When test set (`Kevin355/Who_and_When`). TRAIL (`PatronusAI/TRAIL`) is optional and needs an accepted gate | P3 | Read token only for gated datasets | Free |
| **MuSiQue / tau2-bench** | Task data for HopRAG and ShopDesk | P2 | Public GitHub downloads (CC BY 4.0 / MIT) | Free |
| **Kaggle (or Colab)** | GPU to compute Qwen3-0.6B surprisal features | P3 | Personal account | Free (about 30 GPU-hours/week on Kaggle) |
| **Ollama** | Offline fallback LLM for the demo | P1 laptop | Local install; download `gpt-oss:20b` | Free |
| **Vercel** | Host the web UI | P4 | Project linked to the repo; env `NEXT_PUBLIC_API_URL` | Free |
| **Render / Railway / Fly.io** | Host the API, the SQLite file and the trained model | P1 | Deploy key; **no Groq key in public deploys** | Free/hobby tier |
| Arize Phoenix (optional) | Developer trace viewer during the build | Anyone | Runs locally (`phoenix serve`) | Free |

**Secret rules**
- `.env` is gitignored. Commit a `.env.example` with variable names only.
- The public deployment runs in **recorded mode** and never holds a Groq key.
- Pause bulk data generation during the live pitch. Groq rate limits are per organization, so
  generation would compete with the demo.

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
Edges are stored separately, e.g. `fx/tool#1 ─fx.SGD_INR@v1→ budget/tool#1`.

**2. Training labels are exact.** We create failures ourselves:
- Take a run that passed and change exactly one step (Task 6).
- The run is only labelled if all three of these hold:
  - the original run passed;
  - a no-edit re-run (control) also passes;
  - the changed run fails.
- We then know for certain which step caused the failure, because it was the only thing that
  changed.

**3. At diagnosis time we rank, then prove.**
- The model gives each step a probability of being the root cause.
- For the top 3 suspects, the verifier applies a fix and replays.
- A suspect is **confirmed** only if the fix flips the run to pass on 2 of 2 samples **and**
  the no-edit control stays failed.
- The UI always separates "the model thinks" (a probability) from "the replay proved" (a
  verified result).

**4. "Why" is shown as evidence, not just stated.** Each diagnosis includes:
- the top 3 reasons the model scored the step high, in plain language;
- the path of damage, from the step through every step that consumed its output to the final
  answer;
- similar past failures and which fix worked for them;
- the replay proof.

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
   - `edges(run_id, src_addr, dst_addr, key, version, kind)`, where kind is `state`, `message` or `inferred`
   - `cassette(request_key PRIMARY KEY, kind, response_hash, model, created_at)`
   - `forks(fork_id, base_run_id, edits_json, mode, samples, reexec_steps, cached_steps, tokens_saved, ms_saved, flip_rate, control_flip_rate)`
   - `labels(run_id, root_addr, fault_type, source, recovered, manifest_addr, verified)`, where source is `injected`, `natural_auto`, `human` or `verified`. **Kept out of every feature query.**
   - `diagnoses(run_id, model_version, ranking_json, evidence_json, created_at)`
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

---

### Task 3: Replay engine (checkpointed replay and alternative execution core)
**Owner:** P1 · **Est.:** 6h

**Goal:** resume from any step's exact state, apply an edit, and re-run only the affected steps.

**Build** (`blackbox/replay/`):
- `replay(base_run_id, edits=[], mode="cone"|"prefix"|"full", samples=1, control=False) -> fork_ids`
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
  4. If the `request_key` is in the base run's cassette, return the recorded response.
  5. Otherwise call live: this step is in the cone. Store the result under `request_key + sample_idx`.
- **Divergence check:** for steps before the first edit, the replayed `state_before` must equal
  the recorded one. Otherwise raise `ReplayDivergence(addr)`.
- **Samples and control:**
  - Run K samples of the edited fork. With `control=True`, also run K samples with no edit,
    resampled from the earliest edit point.
  - Store `flip_rate` and `control_flip_rate` with Wilson confidence intervals.
- **Savings:** count re-executed vs cached steps, tokens and milliseconds. Save these to `forks`.
- Emit progress events through an async callback. Task 10 turns them into SSE.

**Test** (`tests/test_replay.py`):
- Replaying with no edits gives **100% cached**, 0 LLM calls, and an identical outcome and final
  state hash.
- On a toy DAG `A → {B, C} → D`, editing B re-executes `{B, D}`, and C is cached.
- If the edited B produces the same output as before, D is cached (early cutoff).
- Changing a recorded tool output on disk raises `ReplayDivergence`.
- `prefix` mode re-executes everything after the edit. The test asserts that `cone` re-runs
  fewer steps than `prefix`.

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
  5. On 15% of forks, also run a no-edit control.
- **Labels** (`label.py`):
  - **POSITIVE:** the fork fails and the control passes. Root = k.
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
  3. The earliest step that flips the run to pass becomes the label.
  4. If no step flips it, the run is marked "unattributable".
  5. These runs are **test-only**.
- **Bulk runner** (`make forge`): async workers with a concurrency cap, resumable, with progress
  logging.

**Data targets** (Developer tier, about 4–6 hours of generation):

| | TripCrew | HopRAG | ShopDesk* |
|---|---|---|---|
| Base runs | 300 | 400 | 280 |
| Fault forks | ~650 | ~600 | ~450 |
| Natural failures (test only) | ~90 | ~150 | ~120 |

That is about 1,500 labelled failed runs and about 22K step rows. **Freeze the dataset** at a
fixed hour by exporting Parquet and saving its hash in `data/DATASET_VERSION`.

**Test:**
- Unit test per operator: only the declared field changes.
- 20 forks per operator with outcome counts printed.
- The flaky rate is below 10%.
- Every POSITIVE label has a root address that exists in the run.
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
| C. Validity | arguments/output schema valid, NaN or out-of-range values, unit tokens | Broken data |
| D. Grounding | share of numbers and entities in the output that appear in no input, conflicts with earlier values, embedding drift from the task | Hallucinated or contradictory values |
| E. Surprisal | Qwen3-0.6B negative log-likelihood (mean, max, p90, numbers only, change from previous step). Computed on Kaggle and cached by output hash | Groq gives no logprobs, so we compute our own |
| F. Novelty | IsolationForest score, kNN distance and Mahalanobis distance vs **passing training runs only** | "This doesn't look like a good run" |
| G. Context | the same signals at steps i−1 and i+1, errors appearing after this step, "first anomaly in the run" | Symptoms show up after the cause |
| H. Lineage | number of downstream consumers, reaches the final answer, later overwritten | How far the damage spread |
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

**Test:**
- An assertion fails the build if any `labels` column, fault type or run ID appears in the
  feature matrix.
- No `task_id` appears in two splits, and S1 fault types are absent from training.
- Baselines produce numbers on S0 and S1.

---

### Task 8: Train and evaluate the diagnoser
**Owner:** P3 · **Est.:** 8h

**Goal:** a model that, given a failed run, ranks its steps by blame, with honest numbers.

**How training works:**
1. **Training examples.**
   - Every failed run in the training split is one group, and each of its steps is one row.
   - Label: root step = 3. A neighbour on the same data path (±1) = 1, so near-misses get partial
     credit. Everything else = 0, including distractor steps.
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
   90% coverage". If the top-1 margin is small, the UI shows the set instead of one answer.
6. **Optional GNN** (stretch):
   - GATv2 over the step graph, with edge types for time order, data dependency, same agent and
     parent call;
   - added to an ensemble only if it improves validation.
7. **Saved artifacts** (`data/models/diagnoser-v1/`): model file, feature list, T, the conformal
   threshold, the dataset hash and the git commit.
8. **Retraining loop.** Each time the verifier confirms a root cause (Task 9), it becomes a new
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
  3. Integrity checks:
     - **artifact audit**: a surface-only classifier that tries to find injected steps should
       score AUROC ≤ 0.65;
     - **shuffled labels**: should score about random;
     - **human agreement**: κ on 40–60 runs labelled in the UI.
- **Expected ranges** (targets, not promises):

  | Split | Top-1 |
  |---|---|
  | S0 | 0.75–0.90 |
  | **S1 (unseen faults)** | **0.40–0.60** (above 0.85 means something is leaking) |
  | Who&When (algorithm-generated subset) | 0.18–0.30 (GPT-4o judge: 0.125) |

- The Who&When result is computed **once**, after the model is frozen.

**Test:**
- Beats the first-error and LLM-judge baselines on S1, with confidence intervals reported.
- The shuffled-label run scores about random.
- The artifact audit is ≤ 0.65 (or the fix is documented).
- `make eval` regenerates every JSON file from scratch.

---

### Task 9: Explanation, verifier and trace comparison
**Owner:** P3 (explain, verifier) + P1 (diff) · **Est.:** 6h

**Goal:** turn the ranking into evidence and proof, and compare any two runs.

**Build:**
- **Reasons** (`explain/reasons.py`):
  - TreeSHAP (`pred_contrib=True`) gives the top 3 features for the suspect.
  - A template table maps each feature to a sentence. For example, `fresh_age_z` becomes
    "Data is 7 months older than in passing runs."
- **Path of damage:** walk the provenance edges forward from the suspect to the final answer.
  Every step on the path is tagged as root, symptom or unaffected.
- **Similar past cases:**
  - kNN (k=12) over the root steps in the training data;
  - output looks like: "Looks like 9 of 12 past stale-data failures. `fresh=true` fixed 7 of
    them."
  - The same lookup picks the fault type and the fix template.
- **Verifier** (`explain/verifier.py`):
  1. Take the top 3 suspects.
  2. Apply a template fix for the predicted fault type, or a gpt-oss-120b proposal if no
     template fits.
  3. Run a `cone` replay with 2 samples.
  4. Run 1 no-edit control.
  5. Mark the suspect `confirmed` if it passes 2/2 and the control fails; otherwise `refuted`.
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

**Test:**
- On 30 S0 test runs, the verifier confirms the true root in at least 60% of cases and the
  control flips in at most 10%.
- Every narrative claim cites an address that exists.
- Diffing a run against itself gives all "same".
- Diffing the demo fork gives first divergence = `fx/tool#1`, the outcome flips, and only cone
  steps are marked changed.

---

### Task 10: API server
**Owner:** P1 · **Est.:** 4h

**Goal:** one backend the UI talks to, which streams replays live.

**Endpoints** (FastAPI, pydantic models from Task 1):

| Method | Path | Returns |
|---|---|---|
| GET | `/runs?agent=&outcome=&split=&q=` | Paged list: status, agent, task, steps, duration, cost, risk, top suspect |
| GET | `/runs/{id}` | Run detail + steps + edges |
| GET | `/runs/{id}/steps/{addr}` | Full step: input, output, reasoning, state before/after |
| GET | `/runs/{id}/diagnosis` | Diagnosis object. Cached; computed on first request |
| POST | `/runs/{id}/verify` | Starts the verifier and returns a `job_id` |
| POST | `/forks` | `{base_run_id, edits, mode, samples, control}` → `fork_id` |
| GET | `/forks/{id}/stream` | **SSE** events (below) |
| GET | `/diff?a=&b=` | Aligned comparison |
| GET | `/eval` | Precomputed evaluation JSON |
| POST | `/labels` | Human label (Label mode) |
| POST | `/v1/traces` | OpenTelemetry trace ingest. Diagnosis and comparison only for external agents |

**SSE events:**
- `step`: `{addr, status: cached|running|done|override|diverged, ms, tokens}`
- `outcome`: `{sample, passed, reason}`
- `summary`: `{reexecuted, cached, tokens_saved, ms_saved, flip_rate, control_flip_rate}`

The browser connects directly to FastAPI (with CORS), not through a Next.js proxy, so events
are not buffered. In `MODE=recorded`, live calls are refused and only cassette hits are allowed.

**Test:**
- `pytest server/` passes.
- `curl -N /forks/{id}/stream` shows `step`* → `outcome` → `summary` in order.
- In recorded mode, a fork that would need a live call returns a clear error instead of hanging.

---

### Task 11: UI part 1, the app shell, Runs and Investigate
**Owner:** P4 · **Est.:** 10h (starts on mocks after Task 1)

**Goal:** a clean screen that answers "what failed, and which step caused it?" within seconds.

**Design system** (set up first, used everywhere):
- **Theme:** dark by default, "cockpit" style, with a light theme too. Near-black background,
  two surface levels, 0.5px borders, no gradients or glow.
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
       a **suspicion bar**, and chips (error, recovered).
     - A toggle switches between **List** and **Graph**. Graph is a React Flow DAG, coloured by
       suspicion, with provenance edges.
     - Hovering a step highlights what it depends on and what depends on it.
   - **Centre, the step inspector.** Tabs:
     - *Input*: the exact prompt messages or tool arguments.
     - *Output*.
     - *State*: the before/after diff with changed keys highlighted.
     - *Reasoning*: the model's reasoning text.
     - *Raw*: JSON.
     - Copy buttons on every tab.
   - **Right, the diagnosis panel.**
     - *Probable cause* card: step, confidence, and "likely among 6, 7, 9" when uncertain.
     - *Why*: 3 plain-language reasons.
     - *Path of damage*: a mini graph.
     - *Seen before*: similar past cases.
     - *Proof*: the verifier result, or a **Verify** button.
     - Each item links to the step it cites.
4. **Crash report.** A printable page with these sections:
   - Synopsis
   - Sequence of events
   - Probable cause
   - Contributing factors
   - Findings (every claim has a clickable step chip)
   - Recommended fix
   - Verification

   Export to Markdown or PDF.

**Test:**
- With the real API, any failed run from the Runs list opens Investigate within 1s.
- The suspect is highlighted, and clicking each evidence chip jumps to the right step.
- Works at 1366×768 and in light and dark themes.
- The keyboard shortcuts work.

---

### Task 12: UI part 2, Fork and fix, Compare, Results, Label
**Owner:** P4 (P1 helps with SSE) · **Est.:** 10h

**Screens:**
1. **Fork and fix**, a drawer opened from the suspect or any step.
   - **Left:** the original value, read-only.
   - **Right:** a Monaco editor pre-filled with the **suggested fix**.
   - **Edit type selector:** change output / tool arguments / prompt / model.
   - **Replay mode:** *Smart* (cone, the default), *From this step* (prefix) or *Full*. It shows
     a **prediction before running**, e.g. "Will re-run 4 of 18 steps, about 2s."
   - **Samples:** 1/3/5, plus an *Include control* toggle.
   - **During the run, the graph animates:**
     - cached steps fade to grey;
     - re-running steps pulse amber;
     - finished steps turn green or red;
     - a live counter bar shows *steps re-run · LLM calls saved · tokens saved · time*.
   - **Result card:**
     - before → after outcome;
     - pass rate with edit vs without (control), e.g. "3/3 vs 0/3";
     - buttons: *Compare runs*, *Save as confirmed fix* (writes a `verified` label).
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
   - Ablation bars, a reliability chart, and the integrity-checks panel (artifact audit,
     shuffled labels, κ).
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
- A first-time user, e.g. a friend from another team, can find the culprit and test a fix in
  under 2 minutes without help. Note where they get stuck and fix it.

---

### Task 13: Deploy, offline mode and hardening
**Owner:** P1 · **Est.:** 5h

**Build:**
- **Docker Compose** (`make demo-offline`): API + SQLite + model + static web build, with
  `MODE=offline` pointing at Ollama `gpt-oss:20b`. Works with Wi-Fi off.
- **Public deploy:**
  - web on Vercel;
  - API on Render/Railway/Fly.io with a persistent disk holding `blackbox.db`, the blobs and the
    model;
  - `MODE=recorded`, no Groq key.
  - Demo forks for the curated runs are pre-recorded, so the public site can replay them.
- **Demo scenarios:** pick 3 failed runs (TripCrew stale FX, a HopRAG poisoned fact, a natural
  failure). Each fix must flip the outcome in **10 of 10** re-runs. Record their cassettes.
- **Fallbacks:**
  - a live call that takes longer than 3s falls back to the recorded response, with the badge
    showing it;
  - a 90-second screen recording of the whole demo.
- **Operations:** restrict CORS to the web origin, add a simple rate limit on `POST /forks`, and
  a health check endpoint.

**Test:**
- With Wi-Fi off, `make demo-offline` runs the full demo.
- The public URL works in an incognito window on a phone and a laptop.
- No secrets appear in the deployed environment or the repo (`git grep -i groq_api_key` finds
  only `.env.example`).

---

### Task 14: Demo and pitch
**Owner:** P4 + all · **Est.:** 4h

- **Deck** (9 slides + backup):
  1. Title
  2. Problem: one bad step sinks a run; LLM judges find the step only 14.2% of the time on
     Who&When.
  3. Insight: record → learn → replay; the replay engine also produces our training data.
  4. Architecture
  5. Live demo
  6. Data: 3 agents × 14 fault types
  7. Results with confidence intervals
  8. Replay savings
  9. Impact and roadmap: crash tests for agents in CI, a regression suite built from past
     crashes
  10. Backup: threats to validity
- **Demo script (4:00):**
  - **0:00–0:20, hook:** "Every aircraft carries a black box. Your AI agents carry nothing."
  - **0:20–0:50, the crash:** run TripCrew live; it fails.
  - **0:50–1:30, investigate:** heatmap, step 7, evidence, the recovered step 9 shown as not
    causal, the crash report.
  - **1:30–2:20, fork and fix:** 4 of 18 steps re-run, the counters update, the run turns green,
    Compare.
  - **2:20–3:10, rigor:** Results page, unseen faults vs the LLM judge, CIs, the artifact audit,
    κ.
  - **3:10–3:40, a second agent:** a natural failure, caught by the same model.
  - **3:40–4:00, close:** "Three lines to instrument your agent. Black Box finds the step,
    explains why, and proves the fix without re-flying the flight."
- **Q&A sheet:**
  - "vs LangSmith/Langfuse": they show traces. We localize the step and replay only the cone.
  - "vs AgentDebugX": it has no trained model and no evaluation on unseen faults.
  - "Synthetic data?": held-out fault types, natural failures that are test-only, κ, and the
    artifact audit.
  - "Nondeterminism?": the cassette, K samples, control forks, and a measured flaky rate.
  - "Why not just ask an LLM?": the leaderboard, plus about 1000× less cost and latency.
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
   - read the suspect and its evidence;
   - click Fork and fix, review the suggested edit, and run it;
   - see pass or fail, then Compare.
   - Typically under 2 minutes, compared with reading hundreds of log lines.
4. **For a team:**
   - group failures by type to fix the most common cause first;
   - save confirmed fixes, which also improves the model;
   - (roadmap) replay saved crash cases after every code change, like unit tests for agents.

**Why it is practical:**
- Diagnosis runs on a laptop CPU in milliseconds and costs nothing.
- Replay only pays for the cone.
- Data stays local.
- Secrets are redacted before storage.
- Recorded mode lets you share a run without sharing keys.

---

## Part F: Schedule (48h) and cut lines

| Hours | P1 Infra | P2 Agents / data | P3 ML | P4 UI / pitch |
|---|---|---|---|---|
| 0–3 | Task 1 | Task 1 review, mock APIs | Who&When adapter, baselines skeleton | Task 1 mocks, design system |
| 3–10 | Task 2 | Task 4 (TripCrew) | Feature code on early traces | Task 11 shell + Runs |
| 10–16 | Task 3 | Task 4 tests, Task 5 HopRAG | Task 7 | Task 11 Investigate |
| **16** | **Gate 1:** record → replay with cone works on TripCrew | | | |
| 16–24 | Help Task 6 runner, start Task 10 | Task 6 operators + bulk generation | Task 7 surprisal (Kaggle), splits | Task 11 report, Task 12 Fork and fix (mock SSE) |
| 24–32 | Task 10, Task 9 diff | Natural labels, κ labelling, **freeze data at h30** | Task 8 | Task 12 Compare + Results |
| **32** | **Gate 2:** real data → model → diagnosis → fork → compare, end to end in the UI | | | |
| 32–40 | Task 13 | Demo scenario selection (10/10 flips) | Task 9 verifier + explanations, final eval | Task 12 Label mode, Task 14 deck |
| **40** | **Feature freeze:** bug fixes only | | | |
| 40–48 | Deploy, offline test | Backup scenarios | Q&A numbers sheet | Rehearsals |

**MVP (must ship):**
- Tasks 1–4 and 6–12 with TripCrew + HopRAG.
- LightGBM.
- Splits S0, S1 and S4.
- Recorded-mode deploy.

**Stretch:**
- ShopDesk (S3 leave-one-agent-out with 3 agents).
- GNN.
- Who&When.
- Conformal sets.
- LLM narrative.
- Fleet fix validation.

**24-hour finale version:**
- drop ShopDesk, the GNN and surprisal;
- start data generation by hour 8;
- freeze data at hour 18;
- feature freeze at hour 20.

---

## Part G: Risks

| Risk | Plan |
|---|---|
| Groq rate limits or model changes | Developer tier, header-driven rate limiter, model IDs in config, prompt caching (stable prefixes), and the cassette for everything already run |
| Replay isn't deterministic | Cassette for unaffected steps; K samples + control; demo cases verified 10/10 |
| The model learns injection tricks instead of real patterns | Ghost hints, plausible values, the artifact audit, held-out fault types, natural-failure test set |
| Data generation runs late | Data freeze hour; LightGBM trains in seconds; evaluation JSON is precomputed |
| Demo Wi-Fi fails | Offline stack via Ollama, recorded mode, backup video, phone hotspot |
| Too much to build | Follow the gates. If Gate 2 slips, cut the stretch goals first, never the end-to-end flow |
