"""Environment-backed Black Box configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str | None
    llm_base_url: str
    agent_model: str
    judge_model: str
    mode: str
    data_dir: Path

    @classmethod
    def load(
        cls,
        env_path: str | Path = ".env",
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        file_values = _read_env_file(Path(env_path))
        source = dict(file_values)
        source.update(dict(os.environ if environ is None else environ))
        mode = source.get("MODE", "recorded").strip().lower()
        if mode not in {"live", "recorded", "offline"}:
            raise ValueError("MODE must be live, recorded, or offline")
        key = source.get("GROQ_API_KEY", "").strip() or None
        return cls(
            groq_api_key=key,
            llm_base_url=source.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            agent_model=source.get("AGENT_MODEL", "openai/gpt-oss-20b"),
            judge_model=source.get("JUDGE_MODEL", "openai/gpt-oss-120b"),
            mode=mode,
            data_dir=Path(source.get("DATA_DIR", "data")),
        )
