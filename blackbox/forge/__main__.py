"""CLI entry point for Fault Forge dataset generation.

Commands (``inject`` is the default, so ``python -m blackbox.forge --target 360`` still works):

- ``inject``   fork passing base runs, inject one fault each, replay, and label
- ``natural``  attribute naturally failed base runs with oracle fixes (test-only labels)
- ``freeze``   export the labels recorded so far and write DATASET_VERSION

``inject --quota N`` runs every operator (or those named by ``--operators``) until it has N
positive forks, walking each operator's own seeded list of sites without repeating one.

Base runs are read from ``--data-dir`` (default ``data/<agent>``, where ``make tripcrew``
records them). Replays call the same LLM backend that recorded each
run: the deterministic test double for fixture runs, Groq for everything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from blackbox.config import Settings
from blackbox.forge.adapters import (
    ADAPTERS,
    AGENT_ERRORS,
    AgentAdapter,
    RoutingClient,
    TripCrewAdapter,
    make_adapter,
)
from blackbox.forge.inject import FaultInjector
from blackbox.forge.natural_label import NaturalLabeler, NaturalLedger
from blackbox.forge.operators import all_operators
from blackbox.forge.runner import ForgeRunner
from blackbox.sdk import Recorder

__all__ = [
    "ADAPTERS",
    "AGENT_ERRORS",
    "AgentAdapter",
    "RoutingClient",
    "TripCrewAdapter",
    "make_adapter",
    "main",
]

logger = logging.getLogger("blackbox.forge")


def _parse_variants(text: str) -> dict[str, int]:
    variants = {}
    for item in text.split(","):
        code, _, count = item.partition("=")
        if not code.strip() or not count.strip().isdigit() or int(count) < 1:
            raise argparse.ArgumentTypeError(f"expected CODE=COUNT with COUNT >= 1, got {item!r}")
        variants[code.strip().upper()] = int(count)
    return variants


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fault Forge dataset generator")
    parser.add_argument(
        "command", nargs="?", choices=["inject", "natural", "freeze"], default="inject"
    )
    parser.add_argument("--agent", choices=sorted(ADAPTERS), default="tripcrew")
    parser.add_argument("--data-dir", type=Path, help="Recorder directory (default data/<agent>)")
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Checkpoint directory (default <data-dir>/forge)"
    )
    parser.add_argument(
        "--freeze-dir", type=Path, help="Frozen dataset directory (default <data-dir>/frozen)"
    )
    parser.add_argument("--freeze", action="store_true", help="Freeze dataset after injecting")
    parser.add_argument("--target", "-n", type=int, default=360, help="Target positive forks")
    parser.add_argument(
        "--max-attempts", type=int, default=1000, help="Maximum attempts, counting resumed ones"
    )
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Replays in flight")
    parser.add_argument(
        "--per-operator", type=int, default=None, help="Run N forks per operator instead"
    )
    parser.add_argument(
        "--quota",
        type=int,
        default=None,
        help="Run each operator until it has N positive forks (or runs out of sites)",
    )
    parser.add_argument(
        "--max-attempts-per-operator",
        type=int,
        default=None,
        help="With --quota: cap on sites tried per operator (default: every site)",
    )
    parser.add_argument(
        "--variants",
        type=_parse_variants,
        default=None,
        help="With --quota: independent edits to draw per site, e.g. R2=3,T1=2 (default 1)",
    )
    parser.add_argument(
        "--operators",
        type=lambda text: [code.strip().upper() for code in text.split(",") if code.strip()],
        default=None,
        help="Comma-separated operator codes to run (default: all)",
    )
    parser.add_argument("--distractor-rate", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=1, help="Replay samples per fork")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="natural: examine at most N failed runs in total (a seeded sample)",
    )
    parser.add_argument("--no-control", action="store_true", help="Skip control runs")
    parser.add_argument(
        "--no-ghost-hints",
        action="store_true",
        help="Always apply operator edits, even on runs recorded with a real LLM",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--restart", action="store_true", help="Discard the checkpoint instead of resuming"
    )
    tripcrew = parser.add_argument_group("tripcrew")
    tripcrew.add_argument(
        "--stale-fx", action="store_true", help="Base runs were recorded with --stale-fx"
    )
    tripcrew.add_argument(
        "--stale-fx-seeds",
        type=lambda text: tuple(int(seed) for seed in text.split(",") if seed.strip()),
        default=(),
        help="Comma-separated scenario seeds whose runs used a stale FX cache",
    )
    return parser


def _selected_operators(codes: list[str] | None):
    operators = all_operators()
    if codes is None:
        return operators
    unknown = sorted(set(codes) - {operator.spec.code for operator in operators})
    if unknown:
        raise SystemExit(f"unknown operator codes: {', '.join(unknown)}")
    return [operator for operator in operators if operator.spec.code in codes]


async def main() -> None:
    args = _parser().parse_args()
    args.data_dir = args.data_dir or Path("data") / args.agent
    args.checkpoint_dir = args.checkpoint_dir or args.data_dir / "forge"
    args.freeze_dir = args.freeze_dir or args.data_dir / "frozen"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    settings = Settings.load()
    client = RoutingClient(settings)
    recorder = Recorder(
        args.data_dir,
        mode="live" if settings.groq_api_key else "offline",
        settings=settings,
        llm_client=client,
    )
    try:
        adapter = make_adapter(recorder, args)
        if args.command == "natural":
            ledger = NaturalLedger(args.checkpoint_dir / "natural_labels.jsonl")
            labeler = NaturalLabeler(recorder, adapter.factory, ledger=ledger)
            await labeler.label_all(args.agent, adapter.oracle_fixes, limit=args.limit)
            report = args.checkpoint_dir / "natural_labels.json"
            labels = list(ledger.labels.values())
            report.write_text(json.dumps([asdict(label) for label in labels], indent=2) + "\n")
            logger.info("Wrote %d natural labels to %s", len(labels), report)
            return

        injector = FaultInjector(
            recorder,
            adapter.factory,
            default_samples=args.samples,
            default_control=not args.no_control,
            ghost_hints=not args.no_ghost_hints,
        )
        runner = ForgeRunner(
            injector,
            concurrency=args.concurrency,
            checkpoint_dir=args.checkpoint_dir,
            distractor_rate=args.distractor_rate,
            seed=args.seed,
            resume=not args.restart,
        )
        if args.command == "inject":
            operators = _selected_operators(args.operators)
            if args.quota is not None:
                await runner.run_quotas(
                    {operator.spec.code: args.quota for operator in operators},
                    agent=args.agent,
                    samples=args.samples,
                    control=not args.no_control,
                    max_attempts_per_operator=args.max_attempts_per_operator,
                    variants=dict(args.variants or {}),
                )
            elif args.per_operator is not None:
                await runner.run_per_operator(
                    count_per_operator=args.per_operator,
                    operators=operators,
                    agent=args.agent,
                    samples=args.samples,
                    control=not args.no_control,
                )
            else:
                await runner.run(
                    target_positive=args.target,
                    max_attempts=args.max_attempts,
                    operators=operators,
                    agent=args.agent,
                    samples=args.samples,
                    control=not args.no_control,
                )
        if args.command == "freeze" or args.freeze:
            runner.freeze_dataset(args.freeze_dir)
    finally:
        await client.aclose()
        recorder.close()


if __name__ == "__main__":
    asyncio.run(main())
