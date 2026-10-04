# Black Box — technical pitch (consolidated, speakable)

**One-breath opener**
> "Black Box is a flight recorder and failure debugger for AI agents. It records every
> step, points at the step that *caused* the failure, re-runs only the steps that depend
> on the fix, and proves the fix against an unchanged control."

**The problem, in one line (say this before any jargon):** an agent run can fail in its
last step while the real mistake happened ten steps earlier.

---

## How to walk through it (flow at a glance)

Record → Provenance → Determinism → Replay → Diagnosis → Training data → Verification → Evaluation

Rule of thumb: **every section ends by naming the artifact it produces** (a step record, an
edge, a replay status, a ranked list, a label, a verdict, a number). That gives the judge a
hook for each block instead of a wall of terms.

---

## 1. Recording and addresses — *the foundation*

Every step gets a **stable address** like `fx/tool#1`: agent role, kind of step, and which
occurrence it is (the first FX **tool** call). Addresses stay stable even when step numbers
shift — if one run has an extra retry, "step 7" stops meaning the same thing, but `fx/tool#1`
always does. That is what lets you line up the same step across two runs.

Each step stores:
- **input hash and output hash** — a short fingerprint of the content; identical content
  always gives the same fingerprint, so you compare two things instantly without reading them;
- **state before / after as content-hashed snapshots** — a copy of everything the agent
  "knows" at that moment, filed under its fingerprint, so identical snapshots are stored once
  (a Merkle checkpoint);
- **bookkeeping**: cache status, tokens, latency, retries, errors.

*Artifact:* one row per step. Code: `blackbox/store/schema.sql` (`steps`), `blackbox/recorder`
(content-addressed `Store` + `save_checkpoint`).

**Say it out loud:** "Two runs of the same task — one had a retry, so they have different step
counts. But `fx/tool#1` names the first FX tool call in **both**, so I can line them up. And every
step's output hash is a fingerprint, so I can tell two steps are identical without reading them."

**Transition:** "So we can record a run. The next question is: which step's output actually
reached which other step?"

## 2. Provenance edges — *follow a value back to its origin*

The `edges` table links a **producer step** (created a value) to a **consumer step** (used it).
Three kinds:
- **state** — a step wrote to the agent's shared memory and another read it (memory entries are
  versioned, so you know exactly which version was read);
- **message** — one step passed output directly to another;
- **inferred** — no explicit link existed, but a distinctive value from step A shows up in
  step B's input, so we infer B used A's output.

The `inferred` kind is deliberately conservative: it only fires on **distinctive values**.
Common values like `1` or `true` would create false links, so only rare, identifiable values
count. This powers "select a value and follow its origin" in the UI.

*Artifact:* the dependency graph. Code: `store/schema.sql` (`edges`, `kind IN
('state','message','inferred')`), surfaced as `Trace.edges` in `blackbox/ml/dataset.py`.

**Say it out loud:** "I click the exchange rate inside the budget step and Black Box walks back to
`fx/tool#1`. That's a **message** edge if the value was passed straight in, and a **state** edge
if the budget step read the rate from shared memory. Where there's no explicit link but a
distinctive number like `64` shows up downstream, it's an **inferred** edge."

**Transition:** "Edges tell us what *should* have happened. But agents run steps in parallel,
and that could move the hashes around — so we had to pin that down."

## 3. Determinism in parallel steps — *why the hashes are trustworthy*

TripCrew runs its workers in parallel (flight, hotel, weather, FX and visa lookups at the same
time). The code **waits for every call in a stage to finish, then commits results in a fixed
order**. That way, async timing — whichever call happens to return first — cannot change
snapshot hashes. Without this, replay would report differences that are pure scheduling noise.

*Artifact:* stable pre-edit snapshot hashes. Code: `docs/tripcrew.md`,
`agents/tripcrew/agent.py`.

**Say it out loud:** "One time the weather lookup returns first, another time the hotel lookup does.
It doesn't matter: the stage waits for all five lookups, then commits them in a fixed order, so the
snapshot hash is identical. Without that, replay would flag differences that are just timing."

**Transition:** "Now that the recording is stable, we can safely replay a piece of it."

## 4. Cassette and replay — *re-run only what depends on the edit*

The **cassette** maps a hash of the exact request (`request_key`) to the recorded response: a
tape of every model and tool answer, indexed by exactly what was asked.

Replay has three modes: **full** (re-run everything), **prefix** (reuse up to a chosen step,
run live after it), **cone** (re-run only what depends on the edit).

For a **fork** (a new run branched from an old one with one change) the engine edits one step,
then walks the **dependency graph**. Steps that don't consume the changed value are served from
the cassette (**cached** — no new model or tool call). Steps in the **cone** — the downstream
steps that actually consume the changed value — are **invalidated** (marked stale because an
input changed) and re-executed (**live**). That is the "only re-run what depends on it" claim.

Unchanged replay makes **zero new calls** and reproduces the final snapshot exactly — a sanity
check that the recording is faithful. The docs report this held for **all 50 HopRAG runs**
(50/50 fully cached, 50/50 matching final state and outcome).

**One caution to state out loud:** the SDK currently labels **state-only steps** (plain program
code, no model or tool call) as `live`. So "live" does *not* always mean "made a paid call" —
distinguish re-execution from call count.

*Artifact:* a replay with per-step `cached / invalidated / live` status. Code:
`blackbox/replay/__init__.py` (`ReplayMode = cone | prefix | full`).

**Say it out loud:** "I fix the stale FX quote at `fx/tool#1`. The budget, writer and verifier
steps consume that rate, so they re-run **live**; the flight, hotel, weather and visa calls don't,
so they're served from the cassette — **cached**, zero new calls. That's the whole claim: re-run
only the cone."


---

## 5. Diagnosis — *localize the cause, and know when not to guess*

The ML side has three parts.

**(a) Features per step** — one row of numbers per step, in switchable groups:
- **A structure** (where the step sits), **B telemetry** (error/retry/finish signals),
  **C validity** (error payloads, unparsable JSON, missing keys), **D grounding** (values that
  contradict the step's own input or earlier steps), **F novelty**, **G context**,
  **H lineage** (how far the output travelled through the provenance graph).
- **Novelty** uses **robust z-scores** (how unusual a value is vs normal runs, in a way outliers
  can't distort) measured against a **per-address reference profile** fitted **only on passing
  runs**, so "unusual" always means "unlike healthy runs of this step," never "unlike the test
  set." **Staleness** is one of these signals.
- `assert_no_leakage` is a guard that stops label information from sneaking into the features —
  otherwise the model would "cheat" and the accuracy would be fake.

**(b) Ranker** — **LightGBM lambdarank**. LightGBM is a fast gradient-boosted decision-tree
library; lambdarank trains it to put the right answer near the top of a list, not to classify
steps independently. It uses **graded relevance** (root = 3, ±1 hop = 2, ±2 hops = 1,
distractors = 0) and `lambdarank_truncation_level = 5`, i.e. it focuses on getting the top 5
right. Raw scores go through a **softmax with a temperature** (fitted on validation runs) so
they become calibrated percentages, not just an ordering.

**Abstention:** a **split-conformal set** picks a probability threshold so the set of steps
above it contains the true root at least `1 − alpha` of the time. If that set is larger than
`max_set` (3), the evidence can't isolate a culprit and the diagnosis **abstains** instead of
naming one. Calibration uses **leave-one-fault-family-out**, so confidence stays honest on
faults the model has never seen.

**(c) Explanation** — deterministic rules produce inspectable evidence (plain if-then checks,
not a model, so you can point to exactly why a step was flagged):
- `reasons.py` — TreeSHAP contributions in plain language ("the model thinks");
- `evidence.py` — citable findings `{addr, json_pointer}` ("the data says");
- `damage.py` — the **path of damage**: from the suspect forward to the final answer;
- `precedents.py` — similar past cases (k-NN over training roots) and the fix they suggest.

So the answer is not only a black-box score. Code: `blackbox/ml/features.py`,
`blackbox/ml/model.py`, `blackbox/explain/*`.

**Say it out loud:** "At the budget step the model sees the rate is a big robust z-score versus
healthy runs and its date is months old, so it ranks the FX step first. The reason line reads like
a sentence — 'the exchange rate was dated seven months ago' — and the evidence points at
`fx/tool#1` and the exact field. If the evidence can't separate the steps, it abstains instead of
guessing."

**Transition:** "A ranker needs labelled failures — and real ones are scarce. So we manufactured
them."

## 6. Training data — Fault Forge

Real labelled failures are scarce, so `forge` takes a passing run, injects **exactly one**
fault, and labels that step as the root cause. The answer key comes for free because we caused
the fault ourselves. Four families (codes in `blackbox/forge/operators.py`):
- **T — tool faults:** T1 wrong value, T2 stale data, T3 empty/404, T4 timeout/500,
  T5 schema drift (a tool's output format quietly changes);
- **R — retrieval faults:** R1 irrelevant documents, R2 poisoned fact;
- **D — decision faults:** D1 wrong arguments, D2 wrong tool, D3 hallucinated value,
  D4 stops too early;
- **C — coordination faults:** C1 instruction misread, C2 constraint dropped, C3 state
  corruption, C4 repeated loop.

`seen_operators()` vs `held_out_operators()` gives the **unseen-fault-type test**: train on some
fault kinds, test on kinds the model never saw (this checks whether it learned a general pattern
or just memorized the injected faults). **Natural failures** — like the stale FX quote that
occurred without injection — are labelled separately by **oracle counterfactual replay**: find
the earliest step where a minimal fix makes the run pass, by actually trying fixes. They are
used for **test only, never training**.

**Say it out loud:** "Take a passing trip, inject exactly one fault — a stale FX quote — and the run
now blows the budget. Because we caused it, the answer key is free: that step is the label. Then
train on some families (wrong-value, timeout) and hold out others (stale-data, poisoned-fact) to
test whether it generalizes."

**Transition:** "We can now propose a cause. But proposing isn't proving."

## 7. Verification — *a fix needs a control*

The fix and an unchanged **control** (a replay with no edit, which measures how often a run flips
by pure chance) run on **paired seeds** — the same random seeds, so any difference comes from the
edit and not from luck.

- **VERIFIED**: the fix's **Wilson 95% lower bound** beats the control's **upper bound**. The
  Wilson interval is a standard way to put an error bar on a pass rate from few samples; the
  *worst plausible* pass rate of the fix beats the *best plausible* pass rate of the control. The
  ranges don't overlap, so it's unlikely to be chance.
- **REFUTED**: a **known-good** intervention reached the step but repeatedly didn't help.
- **INCONCLUSIVE**: not enough evidence (including when upstream damage already reached the
  step, so replacing its output would hide the real root).
- `samples_needed()` estimates how many more runs would settle it.

VERIFIED fixes export as **offline pytest checks** (automated regression tests) with outcome and
hash assertions, so they run with no network. VERIFIED is evidence for the tested task and setup,
**not universal proof**. Code: `blackbox/explain/verifier.py`, `wilson_interval` in
`blackbox/replay/__init__.py`, `blackbox/export/regression.py`.

**Say it out loud:** "I re-run the fix five times and an unchanged control five times on the same
seeds. The fix passes 5/5 and the control 0/5 — the Wilson bounds don't overlap, so it's
**VERIFIED**. That gets frozen into a pytest file that replays with no network."

**Transition:** "Now the honest part — how well does the localization actually work?"

## 8. Evaluation — *where the questions will concentrate*

**Splits** (`blackbox/eval/splits.py`): every split is grouped by `task_bucket(task_id)`, a
**salted hash**, so the same task never appears in both train and test. There is also
`leave_one_agent_out` (train on all *other* agents, test on the held-out agent entirely).
- **S0** — injected, **seen** fault types, test tasks: *known faults on new tasks*.
- **S1** — injected, **held-out** fault types, non-train tasks: *unseen fault families*.
- **S4** — **natural** failures: test-only, never trained on.
- **S3** — leave-one-agent-out (only when the corpus has two or more agents).

**Baselines** — simple strategies the model must beat to justify itself: random step, last step,
first error, and max anomaly score.

**Numbers** (frozen artifacts, 615 traces). Top-1 = the true culprit was ranked first; n = sample
size:

| Split | What it tests | Top-1 | n |
|---|---|---:|---:|
| S0 | familiar faults | 0.88 | 75 |
| S1 | held-out fault types | 0.552 | 96 |
| S4 | natural failures | 0.247 | 85 |

The best baseline on **S1 is 0.625**, which **beats the model**.

**Honest framing (use these words):** the ranker is strong on familiar faults, weak on unseen
ones, and weakest on real natural failures. Don't claim it generalizes. Say: *"The contribution
is the full loop of localize, replay, and verify, and the evaluation is transparent about where
the ranker needs work."*

**Say it out loud:** "On faults like the ones it trained on, top-1 is 0.88. On held-out fault
types it drops to 0.552, and on real natural failures to 0.247 — while a plain baseline scores
0.625 on the held-out set, which beats us. We put that number on the Results page instead of
hiding it."


---

## Term cheat-sheet (for follow-up questions)

- **Stable address** — `agent/kind#occurrence`, e.g. `fx/tool#1`; aligns the same step across
  runs even when step counts change.
- **Content hash** — fingerprint of content; identical content → identical hash.
- **Snapshot / Merkle checkpoint** — state saved under its hash; identical states stored once.
- **Provenance edge** — producer→consumer link: `state`, `message`, or `inferred` (distinctive
  values only).
- **Cassette** — `request_key` → recorded response; the tape that makes replay free.
- **Replay modes** — `full`, `prefix`, `cone`; per-step status `cached / invalidated / live`.
- **Robust z-score** — "how unusual vs healthy runs," computed so outliers can't distort it.
- **Lambdarank** — learning-to-rank objective for LightGBM; graded relevance, truncation at 5.
- **Temperature** — softmax scaling that turns rank scores into calibrated probabilities.
- **Conformal set / abstention** — threshold set that can cover the true root; too big → "not sure."
- **Fault Forge** — inject one fault into a passing run to get a free label (T/R/D/C families).
- **Oracle counterfactual replay** — find the earliest minimal fix that makes a run pass.
- **Wilson interval** — error bar on a pass rate from small samples; used for VERIFIED/REFUTED.
- **Paired seeds** — same randomness for fix and control, so the edit is the only difference.

## One line per section (rehearsal cue cards)

1. **Recording** — every step gets a stable address, content hashes and hashed state snapshots.
2. **Provenance** — edges (`state`/`message`/`inferred`) let you follow a value to its origin.
3. **Determinism** — parallel stages commit in a fixed order, so hashes don't move with timing.
4. **Replay** — cassette + dependency cone: non-consumers cached, the cone invalidated and live.
5. **Diagnosis** — features → LightGBM lambdarank → calibrated probabilities → conformal
   abstention, plus deterministic explanation rules.
6. **Fault Forge** — inject one fault per passing run for a free label; held-out operators test
   generalization; natural failures labelled by counterfactual replay (test-only).
7. **Verification** — fix vs paired control, Wilson bounds, VERIFIED/REFUTED/INCONCLUSIVE,
   exported as offline pytest.
8. **Evaluation** — salted task splits (S0/S1/S4, plus S3), four baselines, and honest limits:
   strong on familiar faults, weak on unseen and natural ones.

## If a judge presses on exact split definitions

S0–S4 are defined in `blackbox/eval/splits.py` (`make_splits`, `leave_one_agent_out`,
`check_splits`). Read that file before answering. In short: **S0** = injected faults with a
*seen* operator on test-bucket tasks; **S1** = injected faults whose operator is marked
`held_out`, on tasks that are **not** in the train bucket; **S4** = `natural_auto` failures,
test-only; **S3** = leave-one-agent-out (needs at least two agents).

