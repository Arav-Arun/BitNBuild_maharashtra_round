<div align="center">

# 📦 Black Box

### A flight recorder and failure debugger for AI agents

**Record every call. Rank the likely culprit. Replay only what changed. Prove the fix.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-diagnoser-2E8B57)
![SQLite](https://img.shields.io/badge/SQLite-store-003B57?logo=sqlite&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-tools-8A2BE2)

</div>

---

## 🤔 The problem

An AI agent run can contain dozens of model calls, tool calls and state changes, and fail because of **one bad intermediate step**. Ordinary traces show *what* happened. They don't tell you *which step caused the failure*, or whether your fix actually works.

## 💡 What Black Box does

Black Box learns from recorded agent runs and answers three questions:

1. **Where did it go wrong?** It ranks the most likely failure-causing steps, with evidence.
2. **Why do we think so?** It shows the dependency graph, the rules that fired and the recorded payloads, and lets you follow any value back to its origin.
3. **Does the fix work?** It edits one step and replays *only the affected part* of the run, side by side with an unchanged control.

## ✨ Features

| | Feature | What you get |
|---|---|---|
| 🔎 | **Runs** | Search and filter indexed recordings by agent, outcome, split and origin. |
| 🕵️ | **Investigate** | Dependency graph, ranked suspects, rule evidence, recorded payloads and value provenance. |
| 🛠️ | **Fork and fix** | Edit a recorded step and stream a selective *cone replay* next to an unchanged control. |
| ⚖️ | **Compare** | Align runs by stable step address and inspect changed payloads, state and outcomes. |
| 📊 | **Results** | Measured evaluation artifacts with intervals, ablations and explicit not-run items. |
| 🏷️ | **Label** | Review a failed run with model and oracle labels hidden, then store your judgment. |
| 📝 | **Crash report** | View and download a Markdown incident report. |
| ✈️ | **New task** | Describe a TripCrew trip in plain language and record it as an inspectable run. |

Also included: OTLP/HTTP JSON GenAI span import (read-only) and MCP tools for coding agents.

## 🏗️ How it fits together

```
 instrumented agent ──► recorder / SDK ──► SQLite store ──► feature extraction
 (TripCrew)             (calls, state,      (immutable        │
                         provenance)         runs)            ▼
                                                       LightGBM diagnoser
                                                        + rule evidence
                                                              │
        Next.js UI  ◄──────────  FastAPI  ◄───────────────────┘
   graph · fork · compare        service layer ──► selective cone replay
                                       │
                                       └──► MCP server (blackbox-mcp)
```

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript, React Flow + ELK.js (trace graph), TanStack Table/Virtual, Recharts, Monaco editor |
| **Backend** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, httpx, SQLite |
| **ML and retrieval** | LightGBM, scikit-learn, NumPy, BM25 (`rank-bm25`) |
| **LLMs** | Groq-hosted `gpt-oss-20b` (agent) and `gpt-oss-120b` (judge), optional and only in live mode |
| **Tooling** | uv, ruff, pytest, Docker Compose, MCP |
| **Hosting** | Vercel (web), Render (recorded-mode API) |

## 🚀 Quick start

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Node.js 22+ and npm.

```sh
cp .env.example .env
make setup
npm --prefix web ci

# Build the local example recordings and model if data/ is empty
./scripts/build_dataset.sh --fresh

# Terminal 1
make dev-api
# Terminal 2
make dev-web
```

Open <http://localhost:3000>. API docs are at <http://127.0.0.1:8000/docs>.

### Run modes

| `MODE` | Behaviour |
|---|---|
| `recorded` | Reuses stored responses only. No network, no keys. Used in deployment. |
| `offline` | Deterministic local stand-ins. The **New task** page uses a small synthetic catalog, not live booking inventory. |
| `live` | Calls the configured OpenAI-compatible endpoint. Put keys in `.env`, never in the browser or a deployment. |

> **No data? No problem.** If you only want the UI, start both servers without `data/`. The API reports degraded health and the browser shows clearly labelled static fixtures from `web/mocks/`.

The dataset build includes deterministic recordings, injected and natural failures, a trained diagnoser and evaluation artifacts. Data, model files and reports are intentionally ignored by Git. A fresh build may download the public MuSiQue-Ans dataset.

## 📈 Evaluation, honestly reported

The frozen artifacts contain **615 evaluated traces**.

| Split | Ranker top-1 | n |
|---|---|---|
| S0 | **0.88** | 75 |
| S1 | 0.552 (best baseline: 0.625) | 96 |
| S4 (natural failures) | 0.247 | 85 |

The current results **do not support a claim that the ranker wins on unseen fault types.** Black Box reports that plainly: the Results page keeps the split and sample size visible next to every number. Regenerate everything with `make eval`.

## 🔌 API

`GET` `/health` · `/agents` · `/runs` · `/failure-groups` · `/runs/{id}` · `/runs/{id}/steps/{addr}` · `/runs/{id}/provenance` · `/runs/{id}/diagnosis` · `/runs/{id}/forks` · `/runs/{id}/twin` · `/runs/{id}/report` · `/runs/{id}/report.md` · `/diff` · `/eval` · `/forks/{id}` · `/forks/{id}/stream` · `/jobs/{id}` · `/labels/queue`

`POST` `/tasks/run` · `/forks` · `/replay/predict` · `/runs/{id}/verify` · `/forks/{id}/export-test` · `/labels` · `/v1/traces`

Every request and response is defined in [server/models.py](server/models.py). The generated TypeScript types live in [web/lib/contract.ts](web/lib/contract.ts), and errors share one JSON envelope.

### MCP for coding agents

```sh
uv run --project /path/to/DeployForGood_maharashtra_round blackbox-mcp
```

Configure it as a stdio server and set `DATA_DIR` to your generated data path.

## 🧪 Checks and commands

```sh
make check                 # lint, format verification, unit tests
npm --prefix web run check # TypeScript
npm --prefix web run build # production Next.js build
make eval                  # regenerate model and evaluation artifacts
make build                 # Python source and wheel
make demo-offline          # Docker Compose showcase
```

## ☁️ Deployment

The public setup is a **Next.js frontend on Vercel** and a **recorded-mode FastAPI service on Render**.

**1. Package the dataset.** `data/` is git-ignored, so host it where the build can download it, for example as a GitHub Release asset:

```sh
tar -czf blackbox-data.tar.gz --exclude=musique_ans_v1.0_dev.jsonl -C data .
```

**2. Render (API).** Settings for the Python runtime:

| Setting | Value |
|---|---|
| Build command | `pip install ".[ml,server]" && python scripts/fetch_data.py "$DATA_URL" data` |
| Start command | `uvicorn server.app:app --host 0.0.0.0 --port $PORT` |
| Health check | `/health` |
| Env | `PYTHON_VERSION=3.12.8`, `MODE=recorded`, `DATA_DIR=data`, `DATA_URL=<archive link>`, `CORS_ORIGINS=<frontend origin>` |

A Docker deploy also works: set `DATA_URL` as a build argument and the [Dockerfile](Dockerfile) unpacks the dataset into `/app/data`.

**3. Vercel (web).** Set **Root Directory** to `web` and add `NEXT_PUBLIC_API_URL` set to the Render URL. This value is baked in at build time.

**4. CORS.** Set Render's `CORS_ORIGINS` to the exact Vercel origin, with no trailing slash. Separate several origins with commas.

Without the dataset the API starts in degraded mode, and the browser falls back to labelled mocks. **Never add a live provider key to Render or Vercel.**

```sh
docker compose up --build   # run API and web locally in containers
```

## 🛡️ Data and safety boundaries

- The SDK sees only calls routed through it, and the included agent tools are local deterministic fixtures.
- Redaction masks common key, email and phone patterns before persistence.
- Original runs are immutable.
- Exported regression artifacts contain recorded outcome and hash assertions. They run without network access and never call a live LLM or external tool.

## 📚 More docs

[Demo script](docs/demo-script.md) · [QA](docs/qa.md) · [Deck outline](docs/deck-outline.md) · [TripCrew](docs/tripcrew.md) · [HopRAG](docs/hoprag.md) · [Problem statement](PROBLEM_STATEMENT.md)
