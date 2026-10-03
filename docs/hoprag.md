# HopRAG (Task 5)

HopRAG adds a sequential retrieval architecture alongside TripCrew's parallel
travel workers. It predicts a decomposition, searches and reads evidence for each
hop, predicts intermediate answers, and emits a final answer. All calls go
through the recorder SDK, with stable step addresses and versioned state reads.

## Data and attribution

The source dataset is [MuSiQue: Multi-hop Questions via Single-hop Question
Composition](https://github.com/StonyBrookNLP/musique), by Harsh Trivedi, Niranjan
Balasubramanian, Tushar Khot and Ashish Sabharwal (TACL 2022). The authors distribute
it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original
download link is in their repository. Data is used without changing the questions
or paragraphs; the local adapter removes gold annotations from the agent view.

For a smaller, direct JSONL download, `hoprag download` fetches the MuSiQue-Ans v1.0
development file from the [bdsaglam/musique community mirror](https://huggingface.co/datasets/bdsaglam/musique),
revision `22873a405dd809893b22ada0b499299fb612d2df`. It verifies the mirror's published
LFS SHA-256 before using the file:

```text
15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b
```

The 30,439,728-byte file has 2,417 answerable questions: 1,252 two-hop, 760
three-hop and 405 four-hop. Most have 20 paragraphs, but 16 rows have 17–19.
The adapter preserves those smaller supplied corpora. It rejects duplicate IDs,
invalid references, unanswerable rows, and decompositions outside 2–4 hops.
Use `--dataset /path/to/musique_ans_v1.0_dev.jsonl` for the authors' original file
or another official-format split. Do not combine dev questions with training
data when implementing the later ML splits.

Dataset files and traces are ignored by Git. The tests use original synthetic
fixtures and require no download or network connection.

## Commands

```sh
make setup
make hoprag-data
make hoprag
```

The first command installs the locked `rank-bm25` dependency alongside existing
dependencies. `hoprag-data` is the explicit network download; repeated invocations
verify and reuse the local copy. `hoprag` runs 50 seeded, balanced examples and
checks unchanged replay for every completed agent run. The default report is
`data/hoprag/report.json`, and traces are in `data/hoprag/blackbox.db` and `content/`.
Each invocation creates new base runs and immutable replay forks.

For live Groq runs, configure `GROQ_API_KEY` locally, then run:

```sh
uv run --locked --extra dev python -m agents.hoprag run --client groq --count 50 \
  --verify-replay --report data/hoprag/live.json
```

The runner uses `AGENT_MODEL` and the existing shared async client. It makes
`predicted hops + 2` model calls per successful base run. Invalid outputs are
recorded as failures; there is no fallback to gold answers. A report is saved
after each completed question and marked interrupted if the suite aborts.

Other options: `--seed`, `--data-dir`, `--report`, `--read-k 1|2|3`. The offline
client additionally accepts `--hops 2|3|4`; its default is three. The live model
predicts its own 2–4-hop plan without being told the dataset's gold hop count.

## Data boundary and tools

`data.py` separates a public `Question` (question text, paragraph IDs, titles and
text) from an `Example` containing final answers, aliases and gold sub-answers.
HopRAG and its retrieval tools receive only `Question`. Gold decomposition and
supporting flags never enter prompts or state snapshots. The evaluator owns the
gold values and scores the final prediction after agent execution.

The selected questions' complete gold decomposition is exported to a separate
content-addressed JSON file under `data/hoprag/oracles/`. Each entry preserves
sub-question ID, sub-question text, sub-answer and supporting paragraph ID for
future oracle fixes. The file includes the dataset hash. It is not loaded into
the agent or the feature store.

Tools use [rank_bm25's BM25Okapi](https://github.com/dorianbrown/rank_bm25):

- `search(query)`: lowercase Unicode word tokenization over titles and paragraph
  text, ranked within this question's corpus only. Equal scores are ordered by ID.
- `read(doc_id)`: returns only the requested paragraph's ID, title and text.
- `answer(text)`: emits the stripped final answer.

All SDK requests also carry a `corpus_id` fingerprint of the question and its
paragraphs. Thus a read of paragraph 0 in two different questions cannot share a
cassette entry, and a changed corpus invalidates the request key.

## Recorded flow

```text
decompose/chat#1
  → hop1/search#1 → hop1/read#1 → hop1/answer#1
  → hop2/search#1 → hop2/read#1 → hop2/answer#1
  → optional hops 3 and 4
  → final/chat#1 → answer/tool#1
```

With the default one paragraph per hop, there are 9–15 steps. Each hop substitutes
earlier predicted answers into its `#1`, `#2` references. Search retrieves rankings;
read prefers the highest-ranked paragraph not read yet, falling back to a previously
read paragraph when the corpus is exhausted. A larger `--read-k` adds separately
recorded reads. Intermediate answers may only cite documents read during that hop;
the final answer may cite any document read in the run. Unknown or missing citations
on a nonempty answer fail validation. Empty answers represent abstention.

The checker uses the normalization and answer-metric definitions from the authors'
[answer evaluator](https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py):
lowercase, remove ASCII punctuation and articles, collapse whitespace, and calculate
multiset token F1. It takes the best exact match and F1 over the main answer and
aliases. Task 5's pass rule is exact match **or** F1 ≥ 0.8. The report logs both
metrics and the pass decision.

## Measured validation

No Groq key was available during implementation. The offline client is an explicit,
small lexical baseline: fixed-hop query expansion and sentence/entity extraction.
It sees only the same retrieved evidence as the live reader. Its scores are not
presented as LLM results.

The development sample at seed 7 contains 17 two-hop, 17 three-hop and 16 four-hop
questions. The recorded implementation run (`data/hoprag/task5-offline.json`) produced:

| Check | Result |
| --- | ---: |
| Real MuSiQue base runs | 50 |
| Pipeline errors | 0 |
| Passed (exact match or F1 ≥ 0.8) | 2/50 (4%) |
| Mean answer F1 | 0.064 |
| Unchanged replays fully cached | 50/50 |
| Matching final state and outcome | 50/50 |

The weak answer accuracy is retained honestly. A meaningful model-quality result
requires running `--client groq`; the offline run establishes recording, scoring
and replay behavior on real data. The unit tests also validate 2-, 3- and 4-hop
scripted executions, gold-data isolation, deterministic sampling, exact/F1 scoring,
corpus isolation, invalid citations, and a network-disabled scored CLI suite.

Task 5's mandatory HopRAG path is implemented. ShopDesk remains the explicitly
optional stretch goal; no tau2-bench runs or rewards are claimed. Repeated
stochastic fix verification remains dependent on the existing replay engine's
sample-isolation work; this task validates unchanged single-sample replay.
