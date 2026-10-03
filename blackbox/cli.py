"""Black Box command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from blackbox.replay import Edit, ReplayEngine
from blackbox.sdk import Recorder


def _load_callable(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("agent must use module:function syntax")
    return getattr(importlib.import_module(module_name), attribute)


async def _fork(args: argparse.Namespace) -> None:
    patch = json.loads(Path(args.patch).read_text(encoding="utf-8"))
    kind = patch.get("kind", "override_output")
    edit = Edit(args.step, kind, patch.get("value"), bool(patch.get("known_good")))
    recorder = Recorder(args.data_dir, mode=args.runtime_mode)
    try:
        batch = await ReplayEngine(recorder).replay(
            args.run,
            _load_callable(args.agent),
            edits=[edit],
            mode=args.replay_mode,
            samples=args.samples,
            control=args.control,
        )
        print(
            json.dumps(
                {
                    "fork_id": batch.fork_id,
                    "verdict": batch.verdict,
                    "fix_pass_rate": batch.fix_pass_rate,
                    "control_pass_rate": batch.control_pass_rate,
                },
                indent=2,
            )
        )
    finally:
        recorder.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="blackbox")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fork = subparsers.add_parser("fork", help="fork and replay a recorded run")
    fork.add_argument("run")
    fork.add_argument("--step", required=True)
    fork.add_argument("--patch", required=True)
    fork.add_argument("--agent", required=True, help="import path module:function")
    fork.add_argument("--data-dir", type=Path, default=Path("data"))
    fork.add_argument("--replay-mode", choices=["cone", "prefix", "full"], default="cone")
    fork.add_argument("--runtime-mode", choices=["live", "recorded", "offline"], default="live")
    fork.add_argument("--samples", type=int, default=5)
    fork.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.command == "fork":
        asyncio.run(_fork(args))


if __name__ == "__main__":
    main()
