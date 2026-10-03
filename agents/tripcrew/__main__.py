"""Run TripCrew scenarios or demonstrate an FX repair without network by default."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from agents.tripcrew import TravelAPI, TripCrew, generate_scenarios
from agents.tripcrew.fixture_client import FixtureClient
from agents.tripcrew.scenarios import catalog
from blackbox.config import Settings
from blackbox.llm import AsyncLLMClient
from blackbox.replay import ReplayEngine, patch_tool_args
from blackbox.sdk import Recorder


async def run_suite(args):
    settings = Settings.load()
    if args.client == "groq" and not settings.groq_api_key:
        raise ValueError("Set GROQ_API_KEY locally before running --client groq")
    scenarios = generate_scenarios(args.count, args.seed)
    client = (
        FixtureClient()
        if args.client == "fixture"
        else AsyncLLMClient(api_key=settings.groq_api_key, base_url=settings.llm_base_url)
    )
    model = "tripcrew-fixture-v1" if args.client == "fixture" else settings.agent_model
    recorder = Recorder(
        args.data_dir,
        mode="live" if args.client == "groq" else "offline",
        settings=settings,
        llm_client=client,
    )
    api = TravelAPI(scenarios, stale_fx=args.stale_fx or args.command == "demo")
    results = []
    try:
        for scenario in scenarios:
            agent = TripCrew(scenario, api, model=model)
            with recorder.run("tripcrew", scenario.scenario_id, args.seed, model=model) as run:
                try:
                    await agent(run)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    run.set_outcome(False, score=0, reason=str(error))
            state = run.state.as_dict()
            result = {
                "run_id": run.run_id,
                "task_id": scenario.scenario_id,
                "passed": run.outcome == "passed",
                "reason": run.checker_reason,
                "plan": state.get("final_plan"),
                "scenario": asdict(scenario),
                "catalog": catalog(scenario),
            }
            if args.command == "demo":
                batch = await ReplayEngine(recorder).replay(
                    run.run_id,
                    agent,
                    edits=[patch_tool_args("fx/tool#1", {"fresh": True}, known_good=True)],
                )
                result["repair"] = {
                    "fork_id": batch.fork_id,
                    "run_id": batch.edited[0].run_id,
                    "outcome": batch.edited[0].outcome,
                    "reason": batch.edited[0].reason,
                    "plan": recorder.store.load_checkpoint(batch.edited[0].state_after)[
                        "final_plan"
                    ],
                    "statuses": batch.edited[0].statuses,
                    "note": "Single deterministic demonstration, not a K-sample causal verdict",
                }
            results.append(result)
            print(
                f"{scenario.scenario_id}: {'PASS' if result['passed'] else 'FAIL'} — {run.checker_reason}"
            )
            if "repair" in result:
                print(
                    f"  Fresh FX replay: {result['repair']['outcome']} — {result['repair']['reason']}"
                )
    finally:
        await recorder.aclose()
    report = {
        "client": args.client,
        "model": model,
        "seed": args.seed,
        "stale_fx": args.stale_fx or args.command == "demo",
        "count": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": sum(r["passed"] for r in results) / len(results),
        "manual_review_status": "pending human review of five runs",
        "runs": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{report['passed']}/{report['count']} passed ({report['pass_rate']:.0%}); report: {args.report}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "demo", "scenarios"], nargs="?", default="run")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--client", choices=["fixture", "groq"], default="fixture")
    parser.add_argument("--stale-fx", action="store_true", help="opt into synthetic stale FX data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tripcrew"))
    parser.add_argument("--report", type=Path, default=Path("data/tripcrew/report.json"))
    args = parser.parse_args()
    if args.count is None:
        args.count = 300 if args.command == "scenarios" else 1 if args.command == "demo" else 20
    if args.count < 1:
        parser.error("--count must be positive")
    try:
        if args.command == "scenarios":
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps([asdict(s) for s in generate_scenarios(args.count, args.seed)], indent=2)
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {args.count} scenarios to {args.report}")
        else:
            asyncio.run(run_suite(args))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"tripcrew: {error}\n")


if __name__ == "__main__":
    main()
