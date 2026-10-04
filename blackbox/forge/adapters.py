"""Agent adapters: rebuild a recorded run's agent and supply oracle fixes for natural labels.

The Fault Forge replays a recorded run by calling the same agent code that recorded it, so every
supported agent needs three things: a chat backend that answers like the recorded one (the
deterministic stand-ins offline, the live model otherwise), a factory that rebuilds the agent
for one run, and the oracle's known-good outputs for the steps a natural failure could be traced to.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blackbox.config import Settings
from blackbox.sdk import Recorder, RunSession

# A fault that makes the agent crash is a failure the checker would also reject.
AGENT_ERRORS = (ValueError, KeyError, TypeError, IndexError, AttributeError, ArithmeticError)
HOPRAG_HEURISTIC_MODEL = re.compile(r"hoprag-lexical-v1-h(\d)")
HOPRAG_FIXTURE_MODEL = "hoprag-reader-fixture-v1"
HOPRAG_DATASET = Path("data/hoprag/musique_ans_v1.0_dev.jsonl")


class RoutingClient:
    """Send each chat request to the backend that produced its recorded model name."""

    def __init__(self, settings: Settings, hoprag_dataset: Path | None = None) -> None:
        self.settings = settings
        self.hoprag_dataset = hoprag_dataset or HOPRAG_DATASET
        self._fixture = None
        self._reader = None
        self._heuristic: dict[int, Any] = {}
        self._live = None

    async def chat(self, **request: Any) -> dict[str, Any]:
        model = str(request.get("model") or "")
        if model.startswith("tripcrew-fixture-"):
            if self._fixture is None:
                from agents.tripcrew.fixture_client import FixtureClient

                self._fixture = FixtureClient()
            try:
                return await self._fixture.chat(**request)
            finally:
                # The test double keeps every request it served; a long forge session would
                # otherwise hold all of them in memory.
                self._fixture.requests.clear()
        if model == HOPRAG_FIXTURE_MODEL:
            if self._reader is None:
                from agents.hoprag.data import load_examples
                from agents.hoprag.fixture_client import GoldReaderClient

                self._reader = GoldReaderClient(load_examples(self.hoprag_dataset))
            return await self._reader.chat(**request)
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
            run.set_outcome(False, score=0, reason=f"Agent raised {type(error).__name__}: {error}")
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
    """TripCrew runs; ``stale_fx_seeds`` names scenario seeds recorded against a stale FX cache."""

    SCENARIO_ID = re.compile(r"TC-(\d+)-(\d+)")

    def __init__(self, recorder: Recorder, args: argparse.Namespace) -> None:
        super().__init__(recorder, args)
        self.scenarios: dict[int, list[Any]] = {}
        self.stale_fx_seeds = frozenset(getattr(args, "stale_fx_seeds", None) or ())

    def _scenario_key(self, run_id: str) -> tuple[int, int]:
        # Scenario IDs encode their generator seed and index, and generation is
        # prefix-stable, so any recorded scenario can be rebuilt from its ID.
        task_id = self._run(run_id)["task_id"]
        match = self.SCENARIO_ID.fullmatch(task_id)
        if not match:
            raise ValueError(f"Not a TripCrew scenario ID: {task_id}")
        return int(match.group(1)), int(match.group(2))

    def _scenario(self, run_id: str):
        from agents.tripcrew import generate_scenarios
        from agents.tripcrew.scenarios import Scenario

        task_id = self._run(run_id)["task_id"]
        if task_id.startswith("PROMPT-"):
            return Scenario(**self._prompt_metadata(task_id)["scenario"])
        seed, index = self._scenario_key(run_id)
        cached = self.scenarios.get(seed, [])
        if len(cached) < index:
            cached = self.scenarios[seed] = generate_scenarios(max(index, 300), seed)
        return cached[index - 1]

    def _stale_fx(self, run_id: str) -> bool:
        task_id = self._run(run_id)["task_id"]
        if task_id.startswith("PROMPT-"):
            try:
                saved = self._prompt_metadata(task_id)
            except ValueError:
                return False
            return bool(saved.get("stale_fx"))
        return bool(self.args.stale_fx) or self._scenario_key(run_id)[0] in self.stale_fx_seeds

    def _prompt_metadata(self, task_id: str) -> dict[str, Any]:
        path = self.recorder.data_dir / "prompt-scenarios" / f"{task_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Prompt-run replay metadata is unavailable for {task_id}") from error

    def factory(self, run_id: str) -> Callable[[RunSession], Any]:
        from agents.tripcrew import TravelAPI, TripCrew

        row = self._run(run_id)
        scenario = self._scenario(run_id)
        task_prompt = (
            self._prompt_metadata(row["task_id"]).get("prompt")
            if row["task_id"].startswith("PROMPT-")
            else None
        )
        # The API must match the one that recorded the base run, or a forced-live
        # control would fetch different data than the recording and stop reproducing.
        api = TravelAPI([scenario], stale_fx=self._stale_fx(run_id))
        return _scored(
            TripCrew(
                scenario,
                api,
                model=row.get("model") or "tripcrew-fixture-v1",
                task_prompt=task_prompt,
            )
        )

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
