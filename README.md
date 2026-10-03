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
| `blackbox/config.py`, `blackbox/llm.py` | `.env` configuration and rate-limit-aware OpenAI-compatible async client |
| `blackbox/store/` | WAL-mode SQLite schema with serialized access |
| `blackbox/recorder/` | Content-addressed blobs and Merkle state checkpoints |
| `blackbox/sdk/` | Run/step recording, LLM and tool wrappers, versioned state, redaction and provenance |
| `blackbox/replay/` | Immutable cone/prefix/full replay, exact caching, controls and verdict intervals |
| `blackbox/eval/` | Localization metrics: Recall@1, Recall@3, MRR |
| `server/models.py`, `web/` | Typed API contract fixtures and the first Next.js recorded-runs shell |

Tasks 1–3 are implemented. Tasks 4 onward follow [PLAN.md](PLAN.md).

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
make setup   # install the locked dev environment into .venv
make check   # ruff lint + format check, unit tests
make format  # apply ruff formatting and import sorting
npm --prefix web ci
npm --prefix web run check
npm --prefix web run build
```

Minimal recorder usage:

```python
import asyncio
import blackbox as bb


async def main():
    recorder = bb.configure("data", mode="live")
    try:
        with bb.run("my-agent", "task-1", seed=7) as run:
            with bb.step("lookup/tool#1", "tool"):
                result = await bb.tool(lookup, query="example")
                run.state["result"] = result
            run.set_outcome(True)
    finally:
        recorder.close()


asyncio.run(main())
```

After editing `pyproject.toml`, run `uv lock` and commit `uv.lock`. Use
`uv sync --locked --extra dev --extra ml` for ML work.

Ground rules (details in PLAN.md):
- Keep fault metadata and labels out of model features.
- Split by task or source run, never by step.
- Fit preprocessing and success-run references on training data only.
- Never mutate an original run; record replay provenance separately.
- Generated data (`data/`, `*.db`), models and credentials stay out of Git.
