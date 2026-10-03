# Black Box

A flight recorder for AI agents. It records agent runs, learns which step caused a
failure, explains why, and proves the fix by replaying only the steps the change
affects.

- [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md): Bit N Build 2026, problem statement #2
- [PLAN.md](PLAN.md): architecture, tech stack, data and model design, evaluation
  protocol, 48h timeline, demo script

## What exists so far

| Module | Status |
| --- | --- |
| `blackbox/recorder/` | Content-addressed checkpoint store (SHA-256 of canonical JSON, integrity-checked) |
| `blackbox/eval/` | Localization metrics: Recall@1, Recall@3, MRR |

Everything else is still to be built, following [PLAN.md](PLAN.md).

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
make setup   # install the locked dev environment into .venv
make check   # ruff lint + format check, unit tests
make format  # apply ruff formatting and import sorting
```

After editing `pyproject.toml`, run `uv lock` and commit `uv.lock`. Use
`uv sync --locked --extra dev --extra ml` for ML work.

Ground rules (details in PLAN.md):
- Keep fault metadata and labels out of model features.
- Split by task or source run, never by step.
- Fit preprocessing and success-run references on training data only.
- Never mutate an original run; record replay provenance separately.
- Generated data (`data/`, `*.db`), models and credentials stay out of Git.
