# Black Box: Step-by-Step Approach

This plan turns the problem statement into an ordered build. Each phase has tasks a team can split. A later phase should start only after the previous phase’s “done when” checks pass, except where a task is marked parallel.

## What we are building

Black Box records an agent run as a sequence of steps and checkpoints. It learns, from many successful and failed runs, which step is most likely responsible when a new run fails. A person can then replay from that step, try a different action, and compare the new outcome with the original run without redoing the steps that were not affected.

## Success criteria

The demo is complete when all of the following are true:

1. Successful and failed runs are stored as structured traces with a checkpoint after every step.
2. A trained model ranks steps in a failed run and names the most likely failure-causing step.
3. The diagnosis includes evidence drawn from that run and from comparable runs.
4. A suspected step can be replayed from the checkpoint immediately before it.
5. An alternative action at that step can be executed forward, and the final outcome can be compared with the original trace.
6. On a held-out set of runs whose failing step is known, we report how often the model ranks that step first and inside the top 3.
7. The same metrics are reported on a second held-out set whose failure types were not used in training.

## Scope for this build

Use one tool-using agent and a fixed suite of tasks. Inject known faults so every evaluation run has a ground-truth failing step. Do not try to debug arbitrary third-party agents in this round. The recorder, model, replay, and evaluation all sit on the same trace format.

Suggested agent loop: a planner that calls a small set of tools (retrieve, calculate, write a structured answer) and stops with a success or failure verdict that a checker can score automatically.

---

## Phase 0 — Align on the trace and the demo task

**Goal:** Freeze the data contract and the agent everyone else builds against.

### Task 0.1 — Define the step schema

Write one schema for a single execution step. Every later component reads and writes this schema.

Required fields:

- `run_id`, `step_index`, `parent_step_index`
- `step_type`: `model_call`, `tool_call`, `retrieval`, or `state_update`
- `input`, `output`, `tool_name`, `retrieved_context`
- `state_before_ref`, `state_after_ref` (pointers to checkpoints, not full copies inside the step)
- `status`: `ok` or `error`
- `started_at`, `ended_at`
- `label` fields used only in training data: `run_outcome` (`success` or `fail`) and, when known, `is_culprit`

**Done when:** A short schema file exists and a sample JSON trace of about 8 steps validates against it.

### Task 0.2 — Define the task suite and the checker

Pick 8–15 tasks the agent can finish in under about 20 steps. Each task has an automatic checker that returns success or failure plus a short reason. Examples: answer a numeric question from a local document, fill a form from retrieved facts, or complete a multi-tool lookup where the final JSON must match an expected shape.

**Done when:** Every task can be scored without a human, and at least one task is known to pass with a correct agent.

### Task 0.3 — List the faults we will inject

Faults are how we get ground truth. Each fault changes exactly one step and records that step index as the culprit.

Start with these fault types:

- Wrong tool chosen
- Tool arguments corrupted
- Retrieved passage swapped for an irrelevant one
- Tool result altered before the model sees it
- A correct intermediate value overwritten in state

Hold two fault types out of training so Phase 6 can test generalization.

**Done when:** A table lists fault type, which step field it changes, and which fault types are train versus held-out.

---

## Phase 1 — Capture execution data

**Goal:** Every agent run, success or failure, becomes a complete trace. This covers **Execution Data**.

### Task 1.1 — Build the recorder

Wrap the agent loop so each model call, tool call, retrieval, and state update emits one step in the schema from Task 0.1. The wrapper must not change the agent’s decision except when a fault from Task 0.3 is explicitly enabled.

**Done when:** One successful run and one failed run are written to disk as traces, and every step has inputs, outputs, and status.

### Task 1.2 — Store checkpoints

After each step, save a restorable snapshot of agent state: message history, tool observations, and any task memory. Store the snapshot by content hash and point `state_after_ref` at it.

**Done when:** Loading checkpoint `k` restores the state that existed immediately after step `k`, verified by comparing a hash of the restored state with the stored hash.

### Task 1.3 — Record run metadata

Each run stores `run_id`, task id, outcome from the checker, fault type if any, culprit step index if any, model name, and a seed.

**Done when:** A run can be loaded by id and the metadata matches the checker result and the injected fault.

### Task 1.4 — Generate the corpus (parallel with Phase 2 once 1.1–1.3 pass)

Run the suite many times:

- Clean runs that succeed
- The same tasks with one injected fault so they fail at a known step
- A smaller set of naturally failed runs with no injected label, kept for qualitative demo only

Target on the order of a few hundred runs, not thousands, as long as every train fault type appears many times and step counts vary.

**Done when:** The corpus summary shows both outcomes, all train fault types, and no missing culprit index on injected-fault runs.

---

## Phase 2 — Checkpointed replay

**Goal:** Investigation starts at an intermediate state. Unaffected earlier steps are not repeated. This covers **Checkpointed Replay**.

### Task 2.1 — Replay loader

Given `run_id` and `step_index`, load the checkpoint from the end of `step_index - 1` (or the initial state when the index is 0) and the original step record.

**Done when:** A script prints the restored messages and state for any chosen step of a stored run.

### Task 2.2 — Resume execution

From the loaded checkpoint, run the agent forward to a terminal outcome using the same tools and checker as the original run.

**Done when:** Replaying a clean successful run from step 0 reproduces a success. Replaying from a mid-run checkpoint of a clean run also finishes, and the prefix of the trace up to that checkpoint is unchanged.

### Task 2.3 — Partial-trace stitching

The new run stores only the suffix it actually executed. The full comparable trace is the original prefix through the checkpoint, plus the new suffix. Mark which steps were reused and which were recomputed.

**Done when:** A replay from step `k` does not call tools for steps `0 .. k-1`, and the stored trace still shows those steps as reused.

---

## Phase 3 — Train the failure diagnosis model

**Goal:** Rank steps of a failed run by how likely each one is to be the cause. This covers **Failure Diagnosis**.

### Task 3.1 — Build step features

Compute features that can be known at diagnosis time without using the culprit label. Useful groups:

- Identity: step type, tool name, depth, position in the run (fraction, not raw index only)
- Outcome signals: tool error, empty retrieval, schema-invalid model output
- Change signals: size of state delta, whether the final answer fields changed at this step
- Contrast signals: how unusual this step is compared with steps at a similar position in successful runs of the same task (different tool, low overlap with typical retrieved text, novel argument pattern)

**Done when:** A feature vector is produced for every step in the corpus and the culprit label is stored beside it but not inside the feature vector.

### Task 3.2 — Split the data

Split by **run**, never by step, so steps from one run do not leak across splits.

- Train: clean successes plus injected failures from the train fault types
- Validation: same fault types, different runs
- Test A (known failures): held-out runs of the train fault types
- Test B (unseen failures): held-out fault types never trained on

**Done when:** No `run_id` appears in more than one split, and Test B fault types are absent from train.

### Task 3.3 — Train a step ranker

Train a model that scores each step of a failed run. The training target is the known culprit step on injected failures. Success runs are used for the contrast features and as negatives (no culprit).

Start with a gradient-boosted tree or logistic regression on the feature vectors. Keep the model small enough to retrain quickly. A prompted LLM judge may be implemented as a baseline to compare against, not as the only system.

At inference, sort steps by score and return the top step plus the top 3.

**Done when:** On the validation split, the model returns a ranking for every failed run and beats a trivial baseline that always blames the last step.

### Task 3.4 — Calibrate a simple acceptance rule

Decide how the demo presents uncertainty. Example: if the top score is far above the second, call it the primary suspect; otherwise show the top 3 as a shortlist.

**Done when:** The rule is a documented function of the scores, and the UI or CLI uses that function.

---

## Phase 4 — Explain the diagnosis

**Goal:** Every accused step comes with evidence from the trace. This covers **Failure Explanation**.

### Task 4.1 — Evidence extractors

For the top suspected step, attach concrete evidence, for example:

- The step’s input, output, and error text
- The state fields that changed at that step
- The nearest successful run of the same task and the step where the two traces first diverge
- Which features contributed most to the score (tool mismatch, bad retrieval overlap, large unexpected state write)

**Done when:** A diagnosis record contains the suspected step index, the score, and at least three evidence items that each quote real fields from a stored trace.

### Task 4.2 — Narrative summary

Turn the evidence record into a short paragraph a person can read: what the step did, why it looks unlike successful runs, and what outcome followed it. The paragraph must only mention facts present in the evidence record.

**Done when:** Reading the paragraph, a teammate can point at the cited step in the raw trace and confirm each claim.

---

## Phase 5 — Alternative execution

**Goal:** Change the suspected step and see whether the run recovers, without replaying the unaffected prefix. This covers **Alternative Execution**.

### Task 5.1 — Step editor

Support a small set of edits at one step:

- Replace tool name or arguments
- Replace the retrieved context
- Replace the model’s action before tools run
- Restore a state field the step overwrote

**Done when:** An edit produces a new step record and does not mutate the original run.

### Task 5.2 — Counterfactual resume

Apply the edit on top of the checkpoint before the suspected step, then resume with Task 2.2. Score the new run with the same checker.

**Done when:** At least one injected failure is repaired by editing the culprit step, the checker returns success, and steps before the edit are marked reused.

### Task 5.3 — Negative control

Apply an edit at a non-culprit step of the same failed run and resume. This shows that changing an arbitrary step is not treated as a fix.

**Done when:** The experiment log contains both a repair at the culprit and a non-repair (or a different outcome) at a different step.

---

## Phase 6 — Evaluate the model

**Goal:** Show that localization works on known failures and degrades honestly on unseen ones. This covers **Model Evaluation**.

### Task 6.1 — Localization metrics

On Test A and Test B separately, compute:

- Recall@1: culprit is the top-ranked step
- Recall@3: culprit is in the top 3
- Mean reciprocal rank of the culprit

Also report the last-step baseline and a random-step baseline on the same splits.

**Done when:** A results table lists both baselines and the trained model for Test A and Test B.

### Task 6.2 — Repair-rate check

On a subset of Test A failures, take the model’s top step, apply the known corrective edit for that fault type, and resume. Report how often the checker flips from fail to success. This ties diagnosis quality to the replay system.

**Done when:** The table includes repair rate when intervening at the predicted step versus intervening at the true culprit (upper bound) and at a random step.

### Task 6.3 — Error notes

Inspect runs where Recall@1 fails. Group them (late symptom blamed instead of the cause, two steps look similar, unseen fault type). Write a short note in the results doc. Do not hide weak Test B numbers.

**Done when:** The results doc states the metrics and the main failure modes.

---

## Phase 7 — Trace comparison

**Goal:** Show how an edit changed the path and the outcome. This covers **Trace Comparison**.

### Task 7.1 — Align two traces

Align the original failed trace and the alternative trace. Shared prefix steps match by index through the checkpoint. After the edited step, align loosely by step type and tool name, and mark insertions, deletions, and changed outputs.

**Done when:** A comparison object lists unchanged prefix length, the first diverging step, and the two final checker results.

### Task 7.2 — Outcome diff

Summarize state and answer differences at the end: which fields flipped, whether the checker reason changed, and which suffix steps appeared only in the alternative.

**Done when:** The demo can show original versus alternative side by side for one repaired run and one run that stayed failed.

---

## Phase 8 — Demo path

**Goal:** One scripted story that uses every key feature in order.

### Task 8.1 — End-to-end script

Script this sequence and keep the commands or clicks stable:

1. Show a failed run’s trace.
2. Show the model’s top suspect and the evidence paragraph.
3. Open the checkpoint before that step.
4. Apply the alternative edit and resume.
5. Show the comparison and the new checker result.
6. Show the evaluation table for Test A and Test B.

**Done when:** A teammate who did not write the code can run the script and reach both a repaired outcome and the metrics table.

### Task 8.2 — Minimal interface

A CLI is enough if it is readable. A simple local page is better if time allows: trace timeline, highlighted suspect, evidence panel, replay button, comparison view. The interface reads stored runs; it does not reimplement diagnosis.

**Done when:** The scripted story in Task 8.1 is clickable or commandable without editing code between steps.

---

## Suggested order and parallelism

| Order | Tasks | Can run in parallel with |
| --- | --- | --- |
| 1 | 0.1, 0.2, 0.3 | — |
| 2 | 1.1, 1.2, 1.3 | — |
| 3 | 1.4 corpus generation | 2.1, 2.2, 2.3 |
| 4 | 3.1, 3.2, 3.3, 3.4 | 5.1 once replay works |
| 5 | 4.1, 4.2 | 5.2, 5.3 |
| 6 | 6.1, 6.2, 6.3 | 7.1, 7.2 |
| 7 | 8.1, 8.2 | — |

## Repo layout to grow into

```text
blackbox/
  schema/          # step and run schema
  agent/           # task suite, tools, checker, fault injection
  recorder/        # trace writer and checkpoints
  replay/          # load checkpoint, resume, stitch
  model/           # features, training, ranking
  explain/         # evidence and narrative
  compare/         # trace alignment and outcome diff
  eval/            # splits, metrics, results
  data/            # generated traces (not hand-edited)
```

## Risks to watch

- **Labels leak into features.** The culprit index and fault type must not be model inputs.
- **Split leak.** Splitting steps instead of runs inflates Recall@1.
- **Replay drift.** If checkpoints omit tool observations or message history, resume will not match a real investigation.
- **Blaming the symptom.** The last step often looks wrong because it emits the bad answer. Contrast against successful runs of the same task is what points earlier.
- **Unseen faults.** Test B should be allowed to score worse than Test A. That gap is part of the result, not a bug to hide.
