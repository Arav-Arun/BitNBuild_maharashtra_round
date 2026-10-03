"""Content-addressed checkpoint store keyed by SHA-256 of canonical JSON."""

import hashlib
import json
import re
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)

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
