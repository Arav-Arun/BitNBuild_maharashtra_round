"""Content-addressed JSON blobs and Merkle state checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid canonical JSON")
        return 0.0 if value == 0 else value
    return value


def canonical_json(value: Any) -> bytes:
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _validate_ref(ref: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", ref):
        raise ValueError("invalid content reference")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _save_bytes(self, directory: str, payload: bytes) -> str:
        ref = hashlib.sha256(payload).hexdigest()
        path = self.root / directory / f"{ref}.json"
        if path.exists():
            self._load_bytes(directory, ref)
        else:
            path.write_bytes(payload)
        return ref

    def _load_bytes(self, directory: str, ref: str) -> bytes:
        _validate_ref(ref)
        path = self.root / directory / f"{ref}.json"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != ref:
            raise ValueError(f"{directory.rstrip('s')} integrity check failed")
        return payload

    def save_json(self, value: Any) -> str:
        return self._save_bytes("blobs", canonical_json(value))

    def load_json(self, ref: str) -> Any:
        return json.loads(self._load_bytes("blobs", ref))

    def save_checkpoint(self, state: dict[str, Any]) -> str:
        manifest = {
            "format": "blackbox-merkle-v1",
            "entries": {key: self.save_json(value) for key, value in sorted(state.items())},
        }
        return self._save_bytes("checkpoints", canonical_json(manifest))

    def load_checkpoint(self, ref: str) -> dict[str, Any]:
        manifest = json.loads(self._load_bytes("checkpoints", ref))
        if manifest.get("format") != "blackbox-merkle-v1":
            raise ValueError("unsupported checkpoint format")
        return {
            key: self.load_json(value_ref) for key, value_ref in manifest.get("entries", {}).items()
        }
