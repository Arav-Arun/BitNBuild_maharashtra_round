"""CLI entry point for Fault Forge bulk dataset generation."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from blackbox.config import Settings
from blackbox.forge.inject import FaultInjector
from blackbox.forge.operators import all_operators, held_out_operators, seen_operators
from blackbox.forge.runner import ForgeRunner
from blackbox.sdk import Recorder

logger = logging.getLogger("blackbox.forge")


def make_agent_factory(recorder: Recorder, agent_name: str = "tripcrew"):
    """Create agent function factory for replay."""
    if agent_name == "tripcrew":
        from agents.tripcrew import TravelAPI, TripCrew, generate_scenarios

        # Cache scenarios
        scenarios_cache = {s.scenario_id: s for s in generate_scenarios(300)}

        def factory(run_id: str):
            row = recorder.database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            if not row:
                raise ValueError(f"Run {run_id} not found in database")
            scenario_id = row["task_id"]
            scenario = scenarios_cache.get(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            api = TravelAPI([scenario])
            agent = TripCrew(scenario, api, model=row.get("model") or "tripcrew-fixture-v1")
            return agent

        return factory

    raise ValueError(f"Unsupported agent: {agent_name}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fault Forge dataset generator")
    parser.add_argument("--target", "-n", type=int, default=360, help="Target positive forks (default 360)")
    parser.add_argument("--max-attempts", type=int, default=1000, help="Maximum injection attempts")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrency limit (default 4)")
    parser.add_argument("--agent", type=str, default="tripcrew", help="Target agent (default tripcrew)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default data)")
    parser.add_argument("--checkpoint-dir", type=str, default="data/forge", help="Checkpoint directory")
    parser.add_argument("--per-operator", type=int, default=None, help="Run N forks per operator instead")
    parser.add_argument("--distractor-rate", type=float, default=0.3, help="Distractor rate (default 0.3)")
    parser.add_argument("--samples", type=int, default=1, help="Replay samples per fork (default 1)")
    parser.add_argument("--no-control", action="store_true", help="Skip control runs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--freeze", action="store_true", help="Freeze dataset upon completion")
    parser.add_argument("--freeze-dir", type=str, default="data", help="Output directory for frozen dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    settings = Settings.load()
    data_path = Path(args.data_dir)
    recorder = Recorder(data_path, mode="offline", settings=settings)

    try:
        agent_factory = make_agent_factory(recorder, args.agent)
        injector = FaultInjector(
            recorder,
            agent_factory,
            default_samples=args.samples,
            default_control=not args.no_control,
        )
        runner = ForgeRunner(
            injector,
            concurrency=args.concurrency,
            checkpoint_dir=Path(args.checkpoint_dir),
            distractor_rate=args.distractor_rate,
            seed=args.seed,
        )

        if args.per_operator is not None:
            await runner.run_per_operator(
                count_per_operator=args.per_operator,
                agent=args.agent,
                samples=args.samples,
                control=not args.no_control,
            )
        else:
            await runner.run(
                target_positive=args.target,
                max_attempts=args.max_attempts,
                agent=args.agent,
                samples=args.samples,
                control=not args.no_control,
            )

        if args.freeze:
            freeze_path = Path(args.freeze_dir)
            runner.freeze_dataset(freeze_path)
    finally:
        recorder.close()


if __name__ == "__main__":
    asyncio.run(main())
