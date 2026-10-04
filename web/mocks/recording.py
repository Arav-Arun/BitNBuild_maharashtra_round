"""Record offline TripCrew and HopRAG runs, fault-injected forks and K-sample replays.

Everything here runs the deterministic stand-ins (`agents/tripcrew/fixture_client.py`,
`agents/hoprag/heuristic.py`). They are not language models. The wrapper below only adds the
fields a live model response would carry so the UI has something to render: a chars/4 token
estimate and a one-line, accurate description of the rule the stand-in applied.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.hoprag.agent import evaluated_agent
from agents.hoprag.data import load_examples, select_examples
from agents.hoprag.heuristic import HeuristicClient
from agents.hoprag.tools import RetrievalTools
from agents.tripcrew import TravelAPI, TripCrew, generate_scenarios
from agents.tripcrew.fixture_client import FixtureClient
from agents.tripcrew.mock_apis import calculate_budget
from agents.tripcrew.scenarios import Scenario
from blackbox.forge.inject import FaultInjector
from blackbox.forge.label import ForkResult
from blackbox.forge.natural_label import NaturalLabel, NaturalLabeler
from blackbox.forge.operators import FaultOperator, all_operators, held_out_operators
from blackbox.replay import (
    ReplayBatch,
    ReplayEngine,
    override_output,
    patch_prompt,
    patch_tool_args,
)
from blackbox.sdk import Recorder

try:
    from blackbox.forge.adapters import HopRAGAdapter, TripCrewAdapter
except ImportError:  # layout before the adapters were split out of the CLI module
    from blackbox.forge.__main__ import HopRAGAdapter, TripCrewAdapter

TRIPCREW_MODEL = "tripcrew-fixture-v1"
HOPRAG_MODEL = "hoprag-lexical-v1-h3"
SEED = 7
STALE_SCENARIOS = {1, 4, 9}
HOPRAG_POOL = 45
HOPRAG_DATASET = Path("data/hoprag/musique_ans_v1.0_dev.jsonl")

TRIPCREW_NOTES = {
    "planner": "Deterministic stand-in (no model call): parsed the trip request with a fixed "
    "pattern into the constraints object.",
    "flight_query": "Deterministic stand-in (no model call): copied the flight task fields into "
    "the query unchanged.",
    "hotel_query": "Deterministic stand-in (no model call): copied the hotel task fields into "
    "the query unchanged.",
    "flight_select": "Deterministic stand-in (no model call): kept the flights that satisfy the "
    "refundable and red-eye constraints and chose the lowest fare.",
    "hotel_select": "Deterministic stand-in (no model call): kept the hotels that satisfy the "
    "refundable and vegetarian constraints and chose the lowest nightly rate.",
    "writer": "Deterministic stand-in (no model call): assembled the plan from the selected "
    "flight and hotel and the budget total.",
    "verifier": "Deterministic stand-in (no model call): returned the draft plan unchanged.",
}
HOPRAG_NOTES = {
    "decompose": "Deterministic lexical stand-in (no model call): repeated the question once per "
    "hop, referring to the previous hop's answer.",
    "hop": "Deterministic lexical stand-in (no model call): chose the sentence with the most "
    "query-term overlap and extracted a date or capitalised span from it.",
    "final": "Deterministic lexical stand-in (no model call): chose the sentence with the most "
    "query-term overlap and extracted a date or capitalised span from it.",
}


def estimate_tokens(text: str) -> int:
    """chars/4, the usual rule of thumb. The stand-ins have no tokenizer."""
    return max(1, math.ceil(len(text) / 4))


class StandInClient:
    """Wrap a deterministic stand-in so its responses carry usage and a reasoning note."""

    def __init__(self, inner: Any, notes: Mapping[str, str]) -> None:
        self.inner = inner
        self.notes = notes

    async def chat(self, **request: Any) -> dict[str, Any]:
        response = await self.inner.chat(**request)
        role = json.loads(request["messages"][-1]["content"])["role"]
        choice = response["choices"][0]
        message = {**choice["message"], "reasoning": self.notes[role]}
        usage = {
            "prompt_tokens": estimate_tokens(json.dumps(request["messages"], sort_keys=True)),
            "completion_tokens": estimate_tokens(message["content"]),
        }
        return {**response, "choices": [{**choice, "message": message}], "usage": usage}


class NoisyWriterClient(StandInClient):
    """A stand-in whose writer step is wrong with a fixed, seeded probability.

    Used only to record a replay whose paired result is genuinely inconclusive. The
    decision depends on the request's sample seed, so it is repeatable and differs by sample.
    """

    def __init__(self, inner: Any, notes: Mapping[str, str], *, probability: float, salt: str):
        super().__init__(inner, notes)
        self.probability = probability
        self.salt = salt

    def _draw(self, seed: Any) -> float:
        digest = hashlib.sha256(f"{self.salt}|{seed}".encode()).hexdigest()
        return int(digest[:8], 16) / 2**32

    async def chat(self, **request: Any) -> dict[str, Any]:
        response = await super().chat(**request)
        role = json.loads(request["messages"][-1]["content"])["role"]
        if role != "writer" or self._draw(request.get("seed")) >= self.probability:
            return response
        choice = response["choices"][0]
        plan = json.loads(choice["message"]["content"])
        plan["total_inr"] = round(plan["total_inr"] + 1000, 2)
        message = {**choice["message"], "content": json.dumps(plan)}
        return {**response, "choices": [{**choice, "message": message}]}


def tripcrew_client() -> StandInClient:
    return StandInClient(FixtureClient(), TRIPCREW_NOTES)


def hoprag_client() -> StandInClient:
    return StandInClient(HeuristicClient(3), HOPRAG_NOTES)


@dataclass
class ForkRecord:
    """One K-sample replay with the engine events it emitted, in order."""

    fork_id: str
    batch: ReplayBatch
    branch_name: str
    hypothesis: str | None
    edits: list[Any]
    raw_events: list[dict[str, Any]]


@dataclass
class World:
    """Everything recorded for the fixtures, in temporary directories."""

    tripcrew: Recorder
    hoprag: Recorder
    scenarios: dict[str, Scenario]
    tripcrew_ids: dict[int, str] = field(default_factory=dict)
    hoprag_ids: list[str] = field(default_factory=list)
    injected: dict[str, ForkResult] = field(default_factory=dict)
    natural: dict[str, NaturalLabel] = field(default_factory=dict)
    forks: dict[str, ForkRecord] = field(default_factory=dict)
    demo: dict[str, ForkRecord] = field(default_factory=dict)
    hoprag_available: bool = False

    def recorder_for(self, run_id: str) -> Recorder:
        return self.hoprag if run_id.startswith(("hr-", "ihr")) else self.tripcrew


async def _record_tripcrew(world: World, count: int) -> None:
    scenarios = generate_scenarios(count, SEED)
    for number, scenario in enumerate(scenarios, 1):
        world.scenarios[scenario.scenario_id] = scenario
        api = TravelAPI([scenario], stale_fx=number in STALE_SCENARIOS)
        agent = TripCrew(scenario, api, model=TRIPCREW_MODEL)
        run = world.tripcrew.run(
            "tripcrew", scenario.scenario_id, SEED, model=TRIPCREW_MODEL, run_id=f"tc-{number:04d}"
        )
        with run:
            try:
                await agent(run)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                run.set_outcome(False, score=0, reason=str(error))
        world.tripcrew_ids[number] = run.run_id


async def _record_hoprag(world: World, dataset: Path) -> None:
    examples = select_examples(load_examples(dataset), HOPRAG_POOL, SEED)
    for number, example in enumerate(examples, 1):
        tools = RetrievalTools(example.question)
        agent = evaluated_agent(example, tools, model=HOPRAG_MODEL, read_k=1)
        run = world.hoprag.run(
            "hoprag",
            example.question.question_id,
            SEED,
            model=HOPRAG_MODEL,
            run_id=f"hr-{number:04d}",
        )
        with run:
            try:
                await agent(run)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                run.set_outcome(False, score=0, reason=str(error))
        world.hoprag_ids.append(run.run_id)


def _operator(code: str) -> FaultOperator:
    return {op.spec.code: op for op in [*all_operators(), *held_out_operators()]}[code]


async def _inject(
    world: World,
    injector: FaultInjector,
    rng: random.Random,
    plans: list[tuple[str, str, str | None]],
) -> None:
    for code, base_run_id, target in plans:
        plan = injector.plan_injection(base_run_id, _operator(code), rng, target_addr=target)
        if plan is None:
            continue
        result = await injector.execute(plan, samples=1, control=True)
        world.injected[f"{result.fork_id}-fix-0"] = result


TRIPCREW_INJECTIONS: list[tuple[str, str, str | None]] = [
    ("T2", "tc-0002", "fx/tool#1"),
    ("T3", "tc-0003", "flight/tool#1"),
    ("T1", "tc-0005", "hotel/tool#1"),
    ("D2", "tc-0007", "flight/chat#2"),
    ("C2", "tc-0008", "hotel/chat#1"),
    ("T4", "tc-0012", "weather/tool#1"),
]


def _first_passing_hoprag(world: World, limit: int) -> list[str]:
    rows = world.hoprag.database.query(
        "SELECT run_id FROM runs WHERE agent = 'hoprag' AND outcome = 'passed' "
        "AND fork_id IS NULL ORDER BY run_id LIMIT ?",
        (limit,),
    )
    return [row["run_id"] for row in rows]


async def _label_natural(
    world: World, recorder: Recorder, adapter: Any, run_ids: list[str]
) -> None:
    labeler = NaturalLabeler(recorder, adapter.factory)
    for run_id in run_ids:
        world.natural[run_id] = await labeler.label_run(run_id, adapter.oracle_fixes(run_id))


async def replay_fork(
    world: World,
    base_run_id: str,
    edits: list[Any],
    *,
    branch_name: str,
    hypothesis: str | None,
    samples: int = 5,
    client: Any | None = None,
) -> ForkRecord:
    """Run one paired K-sample cone replay on the TripCrew recorder and keep its events."""
    recorder = world.tripcrew
    task_id = recorder.database.one("SELECT task_id FROM runs WHERE run_id = ?", (base_run_id,))
    scenario = world.scenarios[task_id["task_id"]]
    number = int(base_run_id.split("-")[1])
    api = TravelAPI([scenario], stale_fx=number in STALE_SCENARIOS)
    agent = TripCrew(scenario, api, model=TRIPCREW_MODEL)
    raw: list[dict[str, Any]] = []
    previous = recorder.llm_client
    if client is not None:
        recorder.llm_client = client
    try:
        batch = await ReplayEngine(recorder).replay(
            base_run_id,
            agent,
            edits=edits,
            mode="cone",
            samples=samples,
            control=True,
            branch_name=branch_name,
            event_callback=raw.append,
        )
    finally:
        recorder.llm_client = previous
    record = ForkRecord(
        fork_id=batch.fork_id,
        batch=batch,
        branch_name=branch_name,
        hypothesis=hypothesis,
        edits=list(edits),
        raw_events=raw,
    )
    world.forks[batch.fork_id] = record
    return record


FRESH_FX = patch_tool_args("fx/tool#1", {"fresh": True}, known_good=True)
NOISY_BASE_NUMBER = 9


def _budget_with_fresh_rate(world: World, base_run_id: str) -> dict[str, Any]:
    """The budget the agent would compute from a fresh rate: the known-good value to splice in."""
    recorder = world.tripcrew
    task_id = recorder.database.one("SELECT task_id FROM runs WHERE run_id = ?", (base_run_id,))
    scenario = world.scenarios[task_id["task_id"]]
    fresh = TravelAPI([scenario]).fx_rate(scenario.scenario_id, scenario.destination, fresh=True)
    row = recorder.database.one(
        "SELECT state_before FROM steps WHERE run_id = ? AND addr = 'budget/tool#1'", (base_run_id,)
    )
    state = recorder.store.load_checkpoint(row["state_before"])
    return calculate_budget(
        flight=state["flight"],
        hotel=state["hotel"],
        fx=fresh,
        visa=state["visa"],
        adults=state["constraints"]["adults"],
        nights=state["hotel_catalog"]["nights"],
        rooms=state["hotel_catalog"]["rooms"],
    )


async def _record_demo_forks(world: World) -> None:
    """The demo story: stale FX on the Mumbai -> Singapore trip, repaired with a fresh rate.

    The three verification candidates are replayed on the same base run. The noisy replay uses
    a different stale-FX run, because an identical request would be served from the cassette
    the clean replay had just written, which is exactly what exact-hash caching should do.
    """
    base = world.tripcrew_ids[1]
    world.demo["fresh_fx"] = await replay_fork(
        world,
        base,
        [FRESH_FX],
        branch_name="fresh-fx",
        hypothesis="The SGD to INR rate is months old; re-fetch it with fresh=true.",
    )
    world.demo["correct_budget"] = await replay_fork(
        world,
        base,
        [override_output("budget/tool#1", _budget_with_fresh_rate(world, base), known_good=True)],
        branch_name="correct-budget",
        hypothesis="The budget total is wrong; replace it with the total from a fresh rate.",
    )
    world.demo["writer_hint"] = await replay_fork(
        world,
        base,
        [patch_prompt("writer/chat#1", "Double-check the total before answering.")],
        branch_name="writer-prompt-hint",
        hypothesis="The writer miscopies the total; ask it to double-check.",
    )
    noisy = NoisyWriterClient(
        FixtureClient(), TRIPCREW_NOTES, probability=0.4, salt="noisy-writer-v1"
    )
    world.demo["noisy"] = await replay_fork(
        world,
        world.tripcrew_ids[NOISY_BASE_NUMBER],
        [FRESH_FX],
        branch_name="fresh-fx-noisy-writer",
        hypothesis="Same fix, replayed against a writer step that is wrong 40% of the time.",
        client=noisy,
    )


async def build_world(workdir: Path, dataset: Path | None = HOPRAG_DATASET) -> World:
    """Record the base runs, injected forks, natural labels and demo forks."""
    tripcrew = Recorder(workdir / "tripcrew", mode="offline", llm_client=tripcrew_client())
    hoprag = Recorder(workdir / "hoprag", mode="offline", llm_client=hoprag_client())
    world = World(tripcrew=tripcrew, hoprag=hoprag, scenarios={})
    await _record_tripcrew(world, 12)
    # Demo forks come first so their replays start from a cold cassette. After a natural
    # label has replayed the same oracle fix, the identical requests would be cache hits.
    await _record_demo_forks(world)

    tc_adapter = TripCrewAdapter(tripcrew, Namespace(stale_fx=False))
    injector = FaultInjector(
        tripcrew, tc_adapter.factory, default_samples=1, default_control=True, ghost_hints=False
    )
    await _inject(world, injector, random.Random(5), TRIPCREW_INJECTIONS)
    stale_adapter = TripCrewAdapter(tripcrew, Namespace(stale_fx=True))
    await _label_natural(
        world, tripcrew, stale_adapter, [world.tripcrew_ids[n] for n in sorted(STALE_SCENARIOS)]
    )

    if dataset is not None and dataset.exists():
        world.hoprag_available = True
        await _record_hoprag(world, dataset)
        hr_adapter = HopRAGAdapter(hoprag, Namespace(dataset=dataset, read_k=1))
        hr_injector = FaultInjector(
            hoprag, hr_adapter.factory, default_samples=1, default_control=True, ghost_hints=False
        )
        passing = _first_passing_hoprag(world, 2)
        plans = [("R2", run_id, None) for run_id in passing[:1]]
        plans += [("T3", run_id, None) for run_id in passing[1:2]]
        await _inject(world, hr_injector, random.Random(11), plans)
        failed = [
            row["run_id"]
            for row in hoprag.database.query(
                "SELECT run_id FROM runs WHERE agent = 'hoprag' AND outcome = 'failed' "
                "AND fork_id IS NULL ORDER BY run_id LIMIT 4"
            )
        ]
        await _label_natural(world, hoprag, hr_adapter, failed)

    return world
