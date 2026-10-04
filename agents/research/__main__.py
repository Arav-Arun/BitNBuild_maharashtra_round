"""Download real passages, record Groq runs, replay a retrieval repair, or train a pilot ranker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx

from agents.research import ResearchAgent
from agents.research.dataset import download, load, save_json, snapshot, task
from blackbox.config import Settings
from blackbox.forge.adapters import RoutingClient
from blackbox.replay import ReplayEngine, patch_tool_args
from blackbox.sdk import Recorder


async def record(args):
    settings = Settings.load()
    if not settings.groq_api_key:
        raise ValueError("Set GROQ_API_KEY; research never substitutes a fixture for a live model")
    rows = load(args.data_dir)
    if args.command == "collect":
        rows = [
            row
            for split in ("train", "validation")
            for row in [r for r in rows if r["split"] == split][: args.count]
        ]
        if not rows or not any(r["split"] == "train" for r in rows):
            raise ValueError("Download both train and validation slices before collecting")
    else:
        rows = rows[: args.count]
    recorder = Recorder(
        args.data_dir, mode="live", settings=settings, llm_client=RoutingClient(settings)
    )
    results = []
    try:
        for row in rows:
            prompt = args.prompt or row["question"]
            variants = (False, True) if args.command == "collect" else (args.empty_retrieval,)
            for broken in variants:
                payload = task(
                    [row] if not args.prompt else load(args.data_dir),
                    prompt,
                    empty_retrieval=broken,
                )
                task_id = f"RESEARCH-{uuid4().hex[:12].upper()}"
                save_json(args.data_dir / "research-tasks" / f"{task_id}.json", payload)
                agent = ResearchAgent(payload, model=settings.agent_model)
                with recorder.run("research", task_id, 7, model=settings.agent_model) as run:
                    try:
                        async with asyncio.timeout(180):
                            await agent(run)
                    except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as error:
                        run.set_outcome(
                            False, score=0, reason=f"Execution stopped: {type(error).__name__}"
                        )
                # Label only a completed controlled retrieval failure, never a provider crash.
                completed = "check" in run.state.as_dict()
                if broken and completed and run.outcome == "failed":
                    recorder.database.execute(
                        "INSERT INTO labels(run_id, root_addr, fault_type, source, manifest_addr) "
                        "VALUES (?, 'retriever/tool#1', 'research_empty_retrieval', 'injected', 'final/state#1')",
                        (run.run_id,),
                    )
                result = {
                    "run_id": run.run_id,
                    "benchmark_id": row["id"],
                    "split": row["split"],
                    "empty_retrieval": broken,
                    "outcome": run.outcome,
                    "reason": run.checker_reason,
                    "completed": completed,
                }
                print(json.dumps(result), flush=True)
                if args.command == "demo" and completed and broken:
                    replay = await ReplayEngine(recorder).replay(
                        run.run_id,
                        agent,
                        edits=[patch_tool_args("retriever/tool#1", {"restore": True})],
                        samples=args.samples,
                        control=args.samples >= 5,
                    )
                    result["repair"] = {
                        "fork_id": replay.fork_id,
                        "outcome": replay.edited[0].outcome,
                        "verdict": replay.verdict,
                        "cached_steps": replay.cached_steps,
                        "reexecuted_steps": replay.reexecuted_steps,
                    }
                    print(json.dumps(result["repair"]), flush=True)
                results.append(result)
                save_json(
                    args.data_dir / "live-report.json",
                    {
                        "model": settings.agent_model,
                        "source": rows[0]["source"],
                        "runs": results,
                        "scope": "Real hosted model calls; empty retrieval is a controlled fault",
                    },
                )
    finally:
        await recorder.aclose()


def train(args):
    from blackbox.ml.dataset import load_corpus
    from blackbox.ml.features import Reference
    from blackbox.ml.model import Diagnoser, DiagnoserConfig

    corpus = load_corpus([args.data_dir])
    train_pairs, val_pairs, healthy_train = [], [], []
    seen = set()
    for trace in corpus.traces.values():
        metadata = snapshot(args.data_dir, trace.task_id)
        question = metadata["benchmark_id"]
        if not question:
            continue
        label = corpus.labels.get(trace.run_id)
        if label and (question, metadata["empty_retrieval"]) not in seen:
            seen.add((question, metadata["empty_retrieval"]))
            pairs = train_pairs if metadata["split"] == "train" else val_pairs
            pairs.append((trace, label))
        elif trace.outcome == "passed" and metadata["split"] == "train":
            healthy_train.append(trace)
    if len(train_pairs) < 4 or len(val_pairs) < 2 or not healthy_train:
        raise ValueError(
            "Collect at least four train and two validation questions with healthy runs"
        )
    train_ids = {snapshot(args.data_dir, t.task_id)["benchmark_id"] for t, _ in train_pairs}
    val_ids = {snapshot(args.data_dir, t.task_id)["benchmark_id"] for t, _ in val_pairs}
    if train_ids & val_ids:
        raise ValueError("Train and validation questions overlap")
    diagnoser = Diagnoser(
        Reference.fit(healthy_train),
        DiagnoserConfig(
            n_estimators=80,
            min_child_samples=2,
            calibration_folds=0,
            calibrate_on="task",
            early_stopping_rounds=10,
        ),
    ).fit(train_pairs, val_pairs)
    diagnoser.metadata.update(
        model_version="research-pilot-v1",
        scope="HotpotQA live traces; controlled empty-retrieval faults only",
        train_question_ids=sorted(train_ids),
        validation_question_ids=sorted(val_ids),
        independent_test_runs=0,
        natural_failure_labels=0,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "labels": [(t.run_id, label.root_addr) for t, label in train_pairs + val_pairs],
                "sources": [
                    snapshot(args.data_dir, t.task_id)["documents"]
                    for t, _ in train_pairs + val_pairs
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    diagnoser.save(args.model_dir, dataset_hash=fingerprint)
    print(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "train_questions": len(train_ids),
                "calibration_questions": len(val_ids),
                "independent_test_runs": 0,
                "scope": diagnoser.metadata["scope"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "run", "demo", "collect", "train"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/research"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/research-pilot-v1"))
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--prompt")
    parser.add_argument("--empty-retrieval", action="store_true")
    parser.add_argument("--samples", type=int, choices=[1, 5], default=1)
    args = parser.parse_args()
    if args.command == "collect" and args.prompt:
        parser.error("--prompt is for run/demo; collect uses distinct benchmark questions")
    if not 1 <= args.count <= 100:
        parser.error("--count must be from 1 to 100")
    try:
        if args.command == "download":
            print(download(args.data_dir, count=args.count, split=args.split))
        elif args.command == "train":
            train(args)
        else:
            asyncio.run(record(args))
    except KeyboardInterrupt:
        parser.exit(130, "research: interrupted; completed recordings were preserved\n")
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        parser.exit(1, f"research: {error}\n")


if __name__ == "__main__":
    main()
