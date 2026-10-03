# Development

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then run `make setup` and `make check`. The local `.venv` and generated data are
ignored by Git. No credentials or external services are needed.

`make format` applies Ruff formatting and import sorting. `make check` checks
formatting, lints, runs the unit tests, and exercises the installed CLI using a
temporary data directory. `make build` creates wheel and source distributions.
Without Make, the commands are listed in the Makefile and can be run directly.

Commit `uv.lock` with dependency changes. After editing `pyproject.toml`, run
`uv lock`, `make setup`, and `make check`. The optional ML dependencies are locked
but are not installed by default; use `uv sync --locked --extra dev --extra ml`
when working on training. Include both extras in later `uv run` commands to retain
that environment. CI uses the same locked development dependencies on Python
3.11–3.14. Its setup follows the [uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/).

Follow APPROACH.md for implementation order. Preserve these invariants:

- Keep fault metadata and labels out of model features.
- Split by run; fit preprocessing and successful-run references on train only.
- Restore the checkpoint immediately before the replayed step.
- Preserve original runs and record replay provenance separately.
- Score final answers with the checker; injection alone does not imply failure.

Add behavior-focused tests for changes to recording, replay, schema, or evaluation.
Keep large corpora, checkpoints, trained models, and local credentials out of Git.
The eight-step example is a small contract fixture; change it alongside deliberate
schema changes. Update the README status when a planned component is implemented.
