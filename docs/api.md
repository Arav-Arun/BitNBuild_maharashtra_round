# Black Box API contract

The Python source of truth is `server/models.py`. These resources are exposed:

- `GET /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/steps/{addr}` (addresses contain `/`, e.g. `fx/tool#1`; URL-encode `#`)
- `GET /runs/{run_id}/provenance?addr=...&pointer=...` (pointer is relative to the step: `/input/...`, `/output/...`)
- `GET /runs/{run_id}/diagnosis`
- `POST /tasks/run` — submit a bounded offline TripCrew prompt; returns a recorded run ID.
- `POST /forks`
- `GET /forks/{fork_id}/stream`
- `GET /diff?a=...&b=...`
- `GET /eval`

The fixtures in `web/mocks/` are frozen static data for the web fallback and tests; `tests/test_contracts.py` validates them against the Pydantic models. See `/docs` on the running API for the complete endpoint reference.
Replay events keep execution phase separate from cache state:

```json
{
  "event": "step",
  "data": {
    "addr": "fx/tool#1",
    "phase": "running",
    "cache_status": "live"
  }
}
```

## Live research tasks

`GET /research/questions` returns downloaded question IDs, text and source attribution, without reference answers or supporting-fact labels.

`POST /tasks/run` accepts `workflow: "research"`, `prompt`, and optional `inject_empty_retrieval`. It requires live mode, a configured provider, and downloaded documents. The default `workflow: "tripcrew"` preserves the existing travel request contract. New task uses Research by default.

Research runs use the same run, diagnosis, fork, comparison and regression-export endpoints. The research pilot is separate from the synthetic TripCrew evaluation. A custom research question receives quotation-grounding checks only.
