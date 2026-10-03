"""Download MuSiQue-Ans or record a scored HopRAG suite."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import httpx

from agents.hoprag.agent import evaluated_agent
from agents.hoprag.checker import check_answer
from agents.hoprag.data import (
    DATASET_REVISION,
    DATASET_SHA256,
    DATASET_URL,
    SOURCE_URL,
    download_dataset,
    file_hash,
    load_examples,
    select_examples,
)
from agents.hoprag.heuristic import HeuristicClient
from agents.hoprag.tools import RetrievalTools
from blackbox.config import Settings
from blackbox.llm import AsyncLLMClient
from blackbox.recorder import content_hash
from blackbox.replay import ReplayEngine
from blackbox.sdk import Recorder


def write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


async def run_suite(args):
    settings = Settings.load()
    if args.client == "groq" and not settings.groq_api_key:
        raise ValueError("Set GROQ_API_KEY locally before using --client groq")
    examples = select_examples(load_examples(args.dataset), args.count, args.seed)
    dataset_hash = file_hash(args.dataset)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    oracles = {example.question.question_id: example.oracle() for example in examples}
    oracle_payload = {"dataset_sha256": dataset_hash, "questions": oracles}
    oracle_path = args.data_dir / "oracles" / f"{content_hash(oracle_payload)}.json"
    write_report(oracle_path, oracle_payload)
    client = (
        HeuristicClient(args.hops)
        if args.client == "heuristic"
        else AsyncLLMClient(api_key=settings.groq_api_key, base_url=settings.llm_base_url)
    )
    model = (
        f"hoprag-lexical-v1-h{args.hops}" if args.client == "heuristic" else settings.agent_model
    )
    recorder = Recorder(
        args.data_dir,
        mode="offline" if args.client == "heuristic" else "live",
        settings=settings,
        llm_client=client,
    )
    report = {
        "client": args.client,
        "model": model,
        "seed": args.seed,
        "requested_count": args.count,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "dataset_source": DATASET_URL if dataset_hash == DATASET_SHA256 else "user-supplied JSONL",
        "dataset_revision": DATASET_REVISION if dataset_hash == DATASET_SHA256 else None,
        "upstream": SOURCE_URL,
        "license": "CC BY 4.0 (MuSiQue)" if dataset_hash == DATASET_SHA256 else None,
        "sample_hop_counts": dict(Counter(len(e.gold_hops) for e in examples)),
        "oracle_file": str(oracle_path),
        "read_k": args.read_k,
        "status": "running",
        "runs": [],
    }

    def save():
        rows = report["runs"]
        report.update(
            count=len(rows),
            passed=sum(row["passed"] for row in rows),
            pass_rate=sum(row["passed"] for row in rows) / len(rows) if rows else 0,
            exact_match_rate=sum(row["exact_match"] for row in rows) / len(rows) if rows else 0,
            mean_f1=sum(row["f1"] for row in rows) / len(rows) if rows else 0,
            pipeline_errors=sum(row["pipeline_error"] is not None for row in rows),
            replay_verified=sum(row.get("replay", {}).get("verified", False) for row in rows),
        )
        write_report(args.report, report)

    save()
    try:
        for example in examples:
            tools = RetrievalTools(example.question)
            agent = evaluated_agent(example, tools, model=model, read_k=args.read_k)
            pipeline_error = None
            with recorder.run(
                "hoprag", example.question.question_id, args.seed, model=model
            ) as run:
                try:
                    await agent(run)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    pipeline_error = str(error)
                    run.set_outcome(False, score=0, reason=pipeline_error)
            prediction = run.state.as_dict().get("final_answer", {}).get("text", "")
            check = check_answer(prediction, example.answer, example.aliases)
            row = {
                "run_id": run.run_id,
                "question_id": example.question.question_id,
                "question": example.question.text,
                "gold_hops": len(example.gold_hops),
                "paragraphs": len(example.question.paragraphs),
                "corpus_id": tools.corpus_id,
                "prediction": prediction,
                **asdict(check),
                "pipeline_error": pipeline_error,
            }
            if pipeline_error is not None:
                row.update(passed=False, reason=pipeline_error)
            report["runs"].append(row)
            save()
            if args.verify_replay and pipeline_error is None:
                before = dict(tools.calls)
                state_ref = run.state.checkpoint()
                batch = await ReplayEngine(recorder).replay(run.run_id, agent)
                replay = batch.edited[0]
                verified = (
                    all(status == "cached" for status in replay.statuses.values())
                    and tools.calls == before
                    and replay.state_after == state_ref
                    and replay.outcome == run.outcome
                    and replay.score == run.score
                )
                row["replay"] = {
                    "run_id": replay.run_id,
                    "fork_id": batch.fork_id,
                    "verified": verified,
                    "statuses": replay.statuses,
                }
                if not verified:
                    raise RuntimeError(
                        f"Unchanged replay failed for {example.question.question_id}"
                    )
            save()
            print(
                f"{example.question.question_id}: {'PASS' if row['passed'] else 'FAIL'} "
                f"F1={row['f1']:.3f}" + ("; replay cached" if row.get("replay") else ""),
                flush=True,
            )
        report["status"] = "complete"
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        save()
        await recorder.aclose()
    print(
        f"{report['passed']}/{report['count']} passed ({report['pass_rate']:.1%}); "
        f"mean F1={report['mean_f1']:.3f}; report: {args.report}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/hoprag/musique_ans_v1.0_dev.jsonl")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/hoprag"))
    parser.add_argument("--report", type=Path, default=Path("data/hoprag/report.json"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--client", choices=["heuristic", "groq"], default="heuristic")
    parser.add_argument(
        "--hops",
        type=int,
        choices=[2, 3, 4],
        default=3,
        help="fixed hop budget for the offline heuristic only",
    )
    parser.add_argument("--read-k", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(download_dataset(args.dataset))
        else:
            asyncio.run(run_suite(args))
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        parser.exit(1, f"hoprag: {error}\n")


if __name__ == "__main__":
    main()
