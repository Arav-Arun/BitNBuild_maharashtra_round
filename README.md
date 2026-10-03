# Black Box

A Python scaffold for the agent flight recorder described in
[PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) and [APPROACH.md](APPROACH.md).
Requires Python 3.11+. The scaffold has no runtime dependencies or API keys.

For development, run `make setup` (requires `uv`), then `make check`.
This installs a local `.venv` from `uv.lock` and runs linting, formatting checks,
tests, and an installed-CLI smoke check. See [CONTRIBUTING.md](CONTRIBUTING.md)
for formatting, packaging, and dependency updates. GitHub Actions runs these
checks on Python 3.11–3.14 after pushing the workflow to GitHub.

## Quick start

Run directly from the repository:

```sh
python3 -m blackbox sample
# Use the run_id printed above:
python3 -m blackbox show <run_id>
python3 -m blackbox checkpoint <run_id> 4
python3 -m unittest discover -s tests -v
```

The checkpoint command restores state **before** the selected zero-based step;
it does not execute a replay. The sample is a synthetic eight-step successful
trace, not an LLM-backed agent or a learned diagnosis demo.

Optional editable installation provides the `blackbox` command:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# When implementing training or linting:
python -m pip install -e '.[ml,dev]'
```

Use `python3 -m blackbox --data-dir /tmp/blackbox sample` to change storage.
Generated traces and checkpoints under `blackbox/data/` are ignored by Git.
Run IDs are unique and existing runs cannot be overwritten. Checkpoints contain
JSON state and are addressed and verified by SHA-256 of canonical JSON bytes.
Local storage is intended for single-process development; concurrent writers and
crash-safe writes remain production hardening work.

## Layout and implementation status

| Module | Included | Next work |
| --- | --- | --- |
| `schema/` | Step, labels, run contracts and validation | Evolve/version contract as needed |
| `agent/` | Eight local tasks, checker, fault registry, fixture runner | Real tool loop and five fault injectors |
| `recorder/` | Trace persistence, checkpoint hashing and integrity checks | Generic recording wrapper |
| `replay/` | Restore checkpoint before any recorded step | Resume, suffix storage, prefix stitching |
| `model/` | Ranker protocol and leakage constraints | Features, training, ranking, uncertainty |
| `explain/` | Evidence requirements | Extractors and grounded narrative |
| `compare/` | Alignment requirements | Trace alignment and outcome diff |
| `eval/` | Recall@1, Recall@3, MRR | Run splits, baselines, repair rate, reports |

`examples/sample_trace.json` is a checked-in eight-step contract example. Generate
a fresh sample to obtain its actual checkpoint files. The schema source is
`blackbox/schema/__init__.py`; load external JSON with `Run.from_dict()` and call
`to_dict()` before persistence. State snapshots hold message history, tool
observations, and task memory. Only JSON-serializable state is supported.

The existing approach document remains the implementation roadmap. Start next
with the instrumented agent loop and fault injection in Phase 1; the current
scaffold does not claim to complete diagnosis, counterfactual replay, or evaluation
against a trained model. Fault split assignments are in [docs/faults.md](docs/faults.md).
