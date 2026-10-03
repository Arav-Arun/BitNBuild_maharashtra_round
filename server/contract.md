# Black Box API contract

The Python source of truth is `server/models.py`. Task 10 will expose these resources:

- `GET /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/steps/{addr}`
- `GET /runs/{run_id}/provenance?pointer=...`
- `GET /runs/{run_id}/diagnosis`
- `POST /forks`
- `GET /forks/{fork_id}/stream`
- `GET /diff?a=...&b=...`
- `GET /eval`

The fixtures in `web/mocks/` must validate against the Pydantic models before UI work merges.
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
