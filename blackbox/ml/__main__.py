"""Train and evaluate the diagnoser, or diagnose one recorded run.

python -m blackbox.ml eval                 # make eval: data/eval/*.json + saved model
python -m blackbox.ml eval --no-ablations  # quick: main model, baselines, integrity
python -m blackbox.ml diagnose <run_id>    # ranked suspects for one run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from blackbox.ml.dataset import load_corpus
from blackbox.ml.evaluate import run_evaluation
from blackbox.ml.model import Diagnoser

DEFAULT_DATA_DIRS = [Path("data/tripcrew")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["eval", "diagnose"])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--data-dir", type=Path, action="append", dest="data_dirs")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/diagnoser-v2"))
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()
    data_dirs = args.data_dirs or DEFAULT_DATA_DIRS
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if args.command == "eval":
        summary = run_evaluation(
            data_dirs, args.out_dir, args.model_dir, ablations=not args.no_ablations
        )
        print(json.dumps(summary, indent=2))
        return

    if not args.run_id:
        parser.error("diagnose needs a run_id")
    diagnoser = Diagnoser.load(args.model_dir)
    corpus = load_corpus(data_dirs)
    trace = corpus.traces.get(args.run_id)
    if trace is None:
        parser.error(f"run {args.run_id} not found in {', '.join(map(str, data_dirs))}")
    print(json.dumps(diagnoser.diagnose(trace).as_dict(), indent=2))


if __name__ == "__main__":
    main()
