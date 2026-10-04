"""Explain, verify, compare and export (Task 9).

python -m blackbox.explain diagnose <run_id> [--verify] [--narrate llm] [--save]
python -m blackbox.explain diff <left_run_id> <right_run_id>
python -m blackbox.explain export <fork_id>                # VERIFIED forks only
python -m blackbox.explain verify-eval [--n 30]            # data/eval/verifier.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from blackbox.config import Settings
from blackbox.diff import compare
from blackbox.explain import Explainer, contract
from blackbox.explain.verifier import Verifier
from blackbox.sdk import Recorder

logger = logging.getLogger("blackbox.explain")


def _adapter_args(agent: str, args: argparse.Namespace) -> argparse.Namespace:
    # replay.json (written by scripts/build_dataset.sh) names the seeds recorded with a stale
    # FX cache; without them a natural stale-FX failure would be rebuilt with fresh rates.
    seeds: tuple[int, ...] = ()
    config = Path(args.data_dir) / "replay.json"
    if config.is_file():
        seeds = tuple(json.loads(config.read_text(encoding="utf-8")).get("stale_fx_seeds", ()))
    return argparse.Namespace(agent=agent, stale_fx=args.stale_fx, stale_fx_seeds=seeds)


class Adapters:
    """One forge adapter per agent, built lazily."""

    def __init__(self, recorder: Recorder, args: argparse.Namespace) -> None:
        from blackbox.forge.__main__ import make_adapter

        self._make = make_adapter
        self.recorder = recorder
        self.args = args
        self._adapters: dict[str, Any] = {}

    def get(self, run_id: str) -> Any:
        row = self.recorder.database.one("SELECT agent FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        agent = row["agent"]
        if agent not in self._adapters:
            self._adapters[agent] = self._make(self.recorder, _adapter_args(agent, self.args))
        return self._adapters[agent]

    def factory(self, run_id: str) -> Any:
        return self.get(run_id).factory(run_id)

    def oracle(self, run_id: str) -> Any:
        return self.get(run_id).oracle_fixes(run_id)


def _recorder(data_dir: Path) -> tuple[Recorder, Any]:
    from blackbox.forge.__main__ import RoutingClient

    settings = Settings.load()
    client = RoutingClient(settings)
    recorder = Recorder(
        data_dir,
        mode="live" if settings.groq_api_key else "offline",
        settings=settings,
        llm_client=client,
    )
    return recorder, client


async def _diagnose(args: argparse.Namespace) -> None:
    recorder, client = _recorder(args.data_dir)
    try:
        adapters = Adapters(recorder, args)
        verifier = Verifier(
            recorder, adapters.factory, oracle=adapters.oracle, samples=args.samples
        )
        explainer = Explainer.from_model_dir(args.model_dir, recorder, verifier=verifier)
        llm = None
        if args.narrate == "llm":
            settings = Settings.load()
            if settings.groq_api_key:
                from blackbox.llm import AsyncLLMClient

                llm = AsyncLLMClient(api_key=settings.groq_api_key, base_url=settings.llm_base_url)
        try:
            bundle = await explainer.investigate(
                args.run_id,
                verify=args.verify,
                narrate=args.narrate,
                llm_client=llm,
                llm_model=Settings.load().judge_model,
            )
        finally:
            if llm is not None:
                await llm.aclose()
        if args.save:
            explainer.save(bundle)
        print(json.dumps(bundle if args.full else contract(bundle), indent=2, default=str))
    finally:
        await client.aclose()
        recorder.close()


async def _verify_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the top-3 suspects of N S0 test runs; count verdicts for true and wrong roots.

    Runs against a scratch copy of the recorder directory, so the hundreds of
    verification forks never land in the real dataset.
    """
    from blackbox.eval.splits import make_splits
    from blackbox.ml.dataset import load_corpus

    corpus = load_corpus([args.data_dir])
    splits = make_splits(corpus)
    pool = sorted(splits.s0)
    chosen = sorted(random.Random(args.seed).sample(pool, min(args.n, len(pool))))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory) / args.data_dir.name
        shutil.copytree(args.data_dir, scratch, ignore=shutil.ignore_patterns("forge", "*.log"))
        recorder, client = _recorder(scratch)
        try:
            adapters = Adapters(recorder, args)
            verifier = Verifier(
                recorder, adapters.factory, oracle=adapters.oracle, samples=args.samples
            )
            explainer = Explainer.from_model_dir(args.model_dir, recorder, verifier=verifier)
            runs = []
            for index, run_id in enumerate(chosen, 1):
                bundle = await explainer.investigate(run_id, record_labels=False)
                root = corpus.labels[run_id].root_addr
                runs.append(
                    {
                        "run_id": run_id,
                        "fault_type": corpus.labels[run_id].fault_type,
                        "root": root,
                        "root_in_top3": any(s["addr"] == root for s in bundle["suspects"]),
                        "abstain": bundle["abstain"],
                        "verdict": bundle["verdict"],
                        "verified_addr": bundle["verified_addr"],
                        "twin_same_task": bundle["twin_same_task"],
                        "twin_divergence": bundle["twin_divergence"],
                        "candidates": [
                            {
                                "addr": c["addr"],
                                "proposed_by": c["proposed_by"],
                                "is_root": c["addr"] == root,
                                "verdict": c["verdict"],
                                "source": c["source"],
                                "masked": c["masked_by"] is not None,
                                "fix_rate": c.get("fix_rate"),
                                "control_rate": c.get("control_rate"),
                            }
                            for c in bundle["verification"]
                        ],
                    }
                )
                logger.info(
                    "[%d/%d] %s root=%s verdict=%s (%s)",
                    index,
                    len(chosen),
                    run_id,
                    root,
                    bundle["verdict"],
                    bundle["verified_addr"],
                )
        finally:
            await client.aclose()
            recorder.close()

    # The plan's targets are about the model's suspects; twin-diff candidates are
    # reported separately so they cannot inflate the model's verification rate.
    candidates = [c for r in runs for c in r["candidates"] if c["proposed_by"] == "model"]
    twin_diff = [c for r in runs for c in r["candidates"] if c["proposed_by"] == "twin_diff"]
    true_roots = [c for c in candidates if c["is_root"]]
    wrong = [c for c in candidates if not c["is_root"]]

    def counts(items: list[dict[str, Any]]) -> dict[str, int]:
        tally = Counter(c["verdict"] for c in items)
        return {v: tally.get(v, 0) for v in ("VERIFIED", "REFUTED", "INCONCLUSIVE")}

    root_verified = counts(true_roots)["VERIFIED"] / len(true_roots) if true_roots else None
    false_verified = counts(wrong)["VERIFIED"] / len(wrong) if wrong else None
    report = {
        "n_runs": len(runs),
        "samples": args.samples,
        "root_in_top3": sum(r["root_in_top3"] for r in runs),
        "true_roots": {"n": len(true_roots), **counts(true_roots)},
        "wrong_roots": {"n": len(wrong), **counts(wrong)},
        "true_root_verified_rate": root_verified,
        "wrong_root_false_verified_rate": false_verified,
        "targets": {"true_root_verified_rate": 0.6, "wrong_root_false_verified_rate": 0.05},
        "passed": bool(
            root_verified is not None and root_verified >= 0.6 and (false_verified or 0.0) <= 0.05
        ),
        "twin_diff_candidates": {
            "n": len(twin_diff),
            "true_roots": sum(c["is_root"] for c in twin_diff),
            **counts(twin_diff),
        },
        "run_verdicts": dict(Counter(r["verdict"] for r in runs)),
        "runs_with_root_verified": sum(r["verified_addr"] == r["root"] for r in runs),
        "sources": dict(Counter(str(c["source"]) for c in candidates)),
        "masked_wrong_roots": sum(c["masked"] for c in wrong),
        "seconds": round(time.perf_counter() - started, 1),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["diagnose", "diff", "export", "verify-eval"])
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tripcrew"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/diagnoser-v1"))
    parser.add_argument("--samples", "-k", type=int, default=5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--narrate", choices=["template", "llm"], default="template")
    parser.add_argument("--save", action="store_true", help="Store in the diagnoses table")
    parser.add_argument("--full", action="store_true", help="Print the whole evidence bundle")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/eval/verifier.json"))
    parser.add_argument("--tests-dir", type=Path, default=Path("tests/regressions"))
    parser.add_argument("--stale-fx", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    needed = {"diagnose": 1, "diff": 2, "export": 1, "verify-eval": 0}[args.command]
    if len(args.ids) != needed:
        parser.error(f"{args.command} takes {needed} id(s)")

    if args.command == "diagnose":
        args.run_id = args.ids[0]
        asyncio.run(_diagnose(args))
    elif args.command == "diff":
        recorder = Recorder(args.data_dir, mode="recorded")
        try:
            print(json.dumps(compare(recorder.database, recorder.store, *args.ids), indent=2))
        finally:
            recorder.close()
    elif args.command == "export":
        from blackbox.export.regression import export

        recorder = Recorder(args.data_dir, mode="recorded")
        try:
            result = export(
                recorder,
                args.ids[0],
                args.tests_dir,
                adapter_args={"stale_fx": args.stale_fx},
            )
        finally:
            recorder.close()
        print(f"Wrote {result.test_path} and {result.fixture_dir}")
    else:
        report = asyncio.run(_verify_eval(args))
        print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
