# Development and deployment

## Checks

Install the locked dependencies using the README setup commands, then run:

```sh
make check
npm --prefix web test
make web-check
make web-build
uv run --locked --extra dev --extra server python -m server.contract_export --check
```

`make check` runs Ruff and the unittest suite. `pytest` discovers `tests/` only, so
local regression exports under `data/exports/` do not interfere with collection.
Run an exported regression explicitly using the command returned by its export.

The API contract source is `server/models.py`. `make contract` regenerates the
committed JSON Schema and TypeScript types. Both outputs remain in Git so the
web app can build without a running API.

## Data and models

`make dataset` generates a smaller deterministic TripCrew corpus, injects faults,
freezes labels, trains the LightGBM diagnoser, and writes evaluation artifacts.
`make dataset-expanded` adds the route-diverse corpus used for the published travel
evaluation without overwriting the original corpus. Both use local stand-ins and need
no model-provider key. `make eval` retrains and evaluates the existing dataset.

| Local path | Contents |
|---|---|
| `data/tripcrew/` | Base SQLite recordings, content blobs, labels and replay metadata |
| `data/expanded/tripcrew/` | Additional route-diverse travel recordings for the full benchmark |
| `data/tripcrew/prompt-scenarios/` | Constraints and catalog seed for user-entered requests |
| `data/models/` | Trained diagnoser, references and related model artifacts |
| `data/eval/` | Measured evaluation results |
| `data/exports/` | Offline regression tests and their self-contained replay fixtures |

These directories are ignored by Git. `scripts/build_dataset.sh --fresh` removes
and rebuilds the training data and model directories; omit `--fresh` to retain them.

## Containers

`docker compose up --build` starts the API and web app and mounts the local `data/`
directory. Generate that data first. The API can start without a dataset but reports
degraded health; the labelled web fallback is read-only.

For a hosted API, `make data-archive` produces `blackbox-data.tar.gz`. SQLite files
are backed up through SQLite's backup API, and per-fork exports are excluded. Upload
the archive to a location you control, then provide its URL as the Docker build
argument `DATA_URL`, or mount the dataset at `/app/data`.

Set the web build's `NEXT_PUBLIC_API_URL` to the API origin and the API's
`CORS_ORIGINS` to the web origin. `render.yaml` configures recorded mode, which does
not make new provider calls. A web build with no API URL uses the static showcase.

## Integrations

`make mcp` starts the local stdio MCP server. The same service functions support
inspection, diagnosis, paired verification and regression export through HTTP and MCP.

`POST /v1/traces` imports OTLP/HTTP JSON traces. Imported traces are read-only:
recorded spans alone do not include application code or adapters needed for replay.
The built-in TripCrew tools have no external side effects. Other agents must provide
appropriate replay adapters for their tools.
