"""Public recorder SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blackbox.config import Settings
from blackbox.sdk.runtime import Recorder, RunSession, StepScope, current_run
from blackbox.sdk.state import State

_default_recorder: Recorder | None = None


def configure(data_dir: str | Path | None = None, **kwargs: Any) -> Recorder:
    global _default_recorder
    if _default_recorder is not None:
        _default_recorder.close()
    settings = kwargs.pop("settings", None) or Settings.load()
    _default_recorder = Recorder(data_dir or settings.data_dir, settings=settings, **kwargs)
    return _default_recorder


def recorder() -> Recorder:
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = configure()
    return _default_recorder


def run(agent: str, task_id: str, seed: int, **kwargs: Any) -> RunSession:
    return recorder().run(agent, task_id, seed, **kwargs)


def step(addr: str, kind: str, name: str | None = None, **kwargs: Any) -> StepScope:
    return current_run().step(addr, kind, name, **kwargs)


async def chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return await current_run().chat(*args, **kwargs)


async def tool(*args: Any, **kwargs: Any) -> Any:
    return await current_run().tool(*args, **kwargs)


async def now() -> str:
    return await current_run().now()


async def uuid() -> str:
    return await current_run().uuid()


async def random() -> float:
    return await current_run().random()


__all__ = [
    "Recorder",
    "RunSession",
    "State",
    "chat",
    "configure",
    "now",
    "random",
    "recorder",
    "run",
    "step",
    "tool",
    "uuid",
]
