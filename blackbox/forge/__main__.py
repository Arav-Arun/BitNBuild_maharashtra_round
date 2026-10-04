"""CLI entry point for Fault Forge dataset generation.

Commands (``inject`` is the default, so ``python -m blackbox.forge --target 360`` still works):

- ``inject``   fork passing base runs, inject one fault each, replay, and label
- ``natural``  attribute naturally failed base runs with oracle fixes (test-only labels)
- ``freeze``   export the labels recorded so far and write DATASET_VERSION

Base runs are read from ``--data-dir`` (default ``data/<agent>``, where ``make tripcrew``
and ``make hoprag`` record them). Replays call the same LLM backend that recorded each
run: the deterministic test doubles for fixture/heuristic runs, Groq for everything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from blackbox.config import Settings
from blackbox.forge.inject import FaultInjector
from blackbox.forge.natural_label import NaturalLabeler
from blackbox.forge.runner import ForgeRunner
from blackbox.sdk import Recorder, RunSession

logger = logging.getLogger("blackbox.forge")

# ArithmeticError covers decimal.InvalidOperation: a corrupted amount crashing a calculator.
AGENT_ERRORS = (ValueError, KeyError, TypeError, IndexError, ArithmeticError)
HOPRAG_HEURISTIC_MODEL = re.compile(r"hoprag-lexical-v1-h(\d)")


class RoutingClient:
    """Send each chat request to the backend that produced its recorded model name."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fixture = None
        self._heuristic: dict[int, Any] = {}
        self._live = None

    async def chat(self, **request: Any) -> dict[str, Any]:
        model = str(request.get("model") or "")
        if model.startswith("tripcrew-fixture-"):
            if self._fixture is None:
                from agents.tripcrew.fixture_client import FixtureClient

                self._fixture = FixtureClient()
            return await self._fixture.chat(**request)
        match = HOPRAG_HEURISTIC_MODEL.fullmatch(model)
        if match:
            from agents.hoprag.heuristic import HeuristicClient

            hops = int(match.group(1))
            client = self._heuristic.setdefault(hops, HeuristicClient(hops))
            return await client.chat(**request)
        if self._live is None:
            if not self.settings.groq_api_key:
                raise RuntimeError(
                    f"run recorded with live model {model!r}; set GROQ_API_KEY to replay it"
                )
            from blackbox.llm import AsyncLLMClient

            self._live = AsyncLLMClient(
                api_key=self.settings.groq_api_key, base_url=self.settings.llm_base_url
            )
        return await self._live.chat(**request)

    async def aclose(self) -> None:
        if self._live is not None:
            await self._live.aclose()


def _scored(agent: Callable[[RunSession], Any]) -> Callable[[RunSession], Any]:
    """Score an agent crash as a failed run, matching how the suites record base runs.

    A fault that makes the agent raise is a failure the checker would also reject;
    letting it escape would count a valid POSITIVE as a forge error instead.
    """

    async def execute(run: RunSession) -> Any:
        try:
            return await agent(run)
        except AGENT_ERRORS as error:
            run.set_outcome(False, score=0, reason=str(error))
            return None

    return execute


def _chat_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ]
    }


class AgentAdapter:
    """Rebuild an agent for a recorded run, and supply oracle fixes for natural labels."""

    def __init__(self, recorder: Recorder, args: argparse.Namespace) -> None:
        self.recorder = recorder
        self.args = args

    def _run(self, run_id: str) -> dict[str, Any]:
        row = self.recorder.database.one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not row:
            raise ValueError(f"Run {run_id} not found in database")
        return row

    def _output(self, run_id: str, addr: str) -> Any:
        step = self.recorder.database.one(
            "SELECT output_hash FROM steps WHERE run_id = ? AND addr = ?", (run_id, addr)
        )
        if not step or not step.get("output_hash"):
            return None
        return self.recorder.store.load_json(step["output_hash"])

    def factory(self, run_id: str) -> Callable[[RunSession], Any]:
        raise NotImplementedError

    def oracle_fixes(self, run_id: str) -> dict[str, Any]:
        raise NotImplementedError


class TripCrewAdapter(AgentAdapter):
    SCENARIO_ID = re.compile(r"TC-(\d+)-(\d+)")

    def __init__(self, recorder: Recorder, args: argparse.Namespace) -> None:
        super().__init__(recorder, args)
        self.scenarios: dict[int, list[Any]] = {}

    def _scenario(self, run_id: str):
        from agents.tripcrew import generate_scenarios

        # Scenario IDs encode their generator seed and index, and generation is
        # prefix-stable, so any recorded scenario can be rebuilt from its ID.
        task_id = self._run(run_id)["task_id"]
        match = self.SCENARIO_ID.fullmatch(task_id)
        if not match:
            raise ValueError(f"Not a TripCrew scenario ID: {task_id}")
        seed, index = int(match.group(1)), int(match.group(2))
        cached = self.scenarios.get(seed, [])
        if len(cached) < index:
            cached = self.scenarios[seed] = generate_scenarios(max(index, 300), seed)
        return cached[index - 1]

    def factory(self, run_id: str) -> Callable[[RunSession], Any]:
        from agents.tripcrew import TravelAPI, TripCrew

        row = self._run(run_id)
        scenario = self._scenario(run_id)
        # The API must match the one that recorded the base run, or a forced-live
        # control would fetch different data than the recording and stop reproducing.
        api = TravelAPI([scenario], stale_fx=self.args.stale_fx)
        return _scored(TripCrew(scenario, api, model=row.get("model") or "tripcrew-fixture-v1"))

    def oracle_fixes(self, run_id: str) -> dict[str, Any]:
        from agents.tripcrew import TravelAPI
        from agents.tripcrew.scenarios import solve

        scenario = self._scenario(run_id)
        solution = solve(scenario)
        api = TravelAPI([scenario])
        return {
            "planner/chat#1": _chat_response(scenario.constraints()),
            "fx/tool#1": api.fx_rate(scenario.scenario_id, scenario.destination, fresh=True),
            "flight/chat#2": _chat_response({"flight_id": solution["flight_id"]}),
            "hotel/chat#2": _chat_response({"hotel_id": solution["hotel_id"]}),
        }


class HopRAGAdapter(AgentAdapter):
    def __init__(self, recorder: Recorder, args: argparse.Namespace) -> None:
        super().__init__(recorder, args)
        from agents.hoprag.data import load_examples

        if not args.dataset.exists():
            raise ValueError(f"{args.dataset} is missing; run `make hoprag-data` first")
        self.examples = {e.question.question_id: e for e in load_examples(args.dataset)}

    def _example(self, run_id: str):
        task_id = self._run(run_id)["task_id"]
        example = self.examples.get(task_id)
        if not example:
            raise ValueError(f"Question {task_id} not found in {self.args.dataset}")
        return example

    def factory(self, run_id: str) -> Callable[[RunSession], Any]:
        from agents.hoprag.agent import evaluated_agent
        from agents.hoprag.tools import RetrievalTools

        row = self._run(run_id)
        example = self._example(run_id)
        tools = RetrievalTools(example.question)
        return _scored(
            evaluated_agent(example, tools, model=row.get("model"), read_k=self.args.read_k)
        )

    def oracle_fixes(self, run_id: str) -> dict[str, Any]:
        example = self._example(run_id)
        fixes: dict[str, Any] = {
            "decompose/chat#1": _chat_response(
                {"questions": [h.question for h in example.gold_hops]}
            )
        }
        for hop, gold in enumerate(example.gold_hops, 1):
            # A grounded answer must cite a document read in that hop, so cite the
            # recorded read rather than the gold paragraph the agent may never have seen.
            read = self._output(run_id, f"hop{hop}/read#1")
            if not isinstance(read, dict) or "doc_id" not in read:
                continue
            fixes[f"hop{hop}/answer#1"] = _chat_response(
                {"text": gold.answer, "doc_ids": [read["doc_id"]]}
            )
        first_read = self._output(run_id, "hop1/read#1")
        if isinstance(first_read, dict) and "doc_id" in first_read:
            fixes["final/chat#1"] = _chat_response(
                {"text": example.answer, "doc_ids": [first_read["doc_id"]]}
            )
        return fixes


ADAPTERS: dict[str, type[AgentAdapter]] = {"tripcrew": TripCrewAdapter, "hoprag": HopRAGAdapter}


def make_adapter(recorder: Recorder, args: argparse.Namespace) -> AgentAdapter:
    try:
        return ADAPTERS[args.agent](recorder, args)
    except KeyError:
        raise ValueError(f"Unsupported agent: {args.agent}") from None


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
    parser.add_argument("--distractor-rate", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=1, help="Replay samples per fork")
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
    hoprag = parser.add_argument_group("hoprag")
    hoprag.add_argument(
        "--dataset", type=Path, default=Path("data/hoprag/musique_ans_v1.0_dev.jsonl")
    )
    hoprag.add_argument("--read-k", type=int, choices=[1, 2, 3], default=1)
    return parser


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
            labeler = NaturalLabeler(recorder, adapter.factory)
            labels = await labeler.label_all(args.agent, adapter.oracle_fixes)
            report = args.checkpoint_dir / "natural_labels.json"
            report.parent.mkdir(parents=True, exist_ok=True)
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
        if args.command == "freeze" or args.freeze:
            runner.freeze_dataset(args.freeze_dir)
    finally:
        await client.aclose()
        recorder.close()


if __name__ == "__main__":
    asyncio.run(main())
