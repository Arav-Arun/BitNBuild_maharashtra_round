"""Local scaffold smoke path: record, inspect, and restore a checkpoint."""

import argparse
import json

from blackbox.agent.sample import record_sample
from blackbox.recorder import Store
from blackbox.replay import load_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="blackbox/data", help="local generated-data directory"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample", help="record a synthetic successful eight-step trace")
    show = commands.add_parser("show", help="validate and print a stored trace")
    show.add_argument("run_id")
    checkpoint = commands.add_parser("checkpoint", help="restore state immediately before a step")
    checkpoint.add_argument("run_id")
    checkpoint.add_argument("step_index", type=int)
    args = parser.parse_args()
    try:
        store = Store(args.data_dir)
        if args.command == "sample":
            run = record_sample(store)
            result = {"run_id": run.run_id, "outcome": run.outcome, "steps": len(run.steps)}
        elif args.command == "show":
            result = store.load_run(args.run_id).to_dict()
        else:
            result, _ = load_before(store, args.run_id, args.step_index)
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"blackbox: {exc}\n")
