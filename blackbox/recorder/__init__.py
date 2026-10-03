"""Local immutable checkpoint and trace storage."""
import hashlib
import json
import re
from pathlib import Path

from blackbox.schema import Run


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for folder in ("checkpoints", "runs"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state: dict) -> str:
        payload = canonical_json(state)
        ref = hashlib.sha256(payload).hexdigest()
        path = self.root / "checkpoints" / f"{ref}.json"
        if path.exists():
            self.load_checkpoint(ref)
        else:
            path.write_bytes(payload)
        return ref

    def load_checkpoint(self, ref: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{64}", ref):
            raise ValueError("invalid checkpoint reference")
        payload = (self.root / "checkpoints" / f"{ref}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != ref:
            raise ValueError("checkpoint integrity check failed")
        return json.loads(payload)

    def _run_path(self, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("invalid run id")
        return self.root / "runs" / f"{run_id}.json"

    def save_run(self, run: Run):
        payload = run.to_dict()
        for ref in {run.initial_state_ref} | {r for s in run.steps
                                             for r in (s.state_before_ref, s.state_after_ref)}:
            self.load_checkpoint(ref)
        with self._run_path(run.run_id).open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)

    def load_run(self, run_id: str) -> Run:
        return Run.from_dict(json.loads(self._run_path(run_id).read_text()))
