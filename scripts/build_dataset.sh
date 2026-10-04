#!/usr/bin/env bash
# Rebuild the offline dataset end to end: base runs, injected faults, natural failures,
# frozen labels, then train and evaluate the diagnoser. Deterministic and network-free
# once data/hoprag/musique_ans_v1.0_dev.jsonl is present (`make hoprag-data`).
set -euo pipefail

RUN=(uv run --locked --extra dev --extra ml --extra server python)
STALE_SEED=11
QUOTA_TRIPCREW=${QUOTA_TRIPCREW:-30}
QUOTA_HOPRAG=${QUOTA_HOPRAG:-20}

if [[ "${1:-}" == "--fresh" ]]; then
  rm -rf data/tripcrew data/eval data/models
  find data/hoprag -mindepth 1 -maxdepth 1 ! -name 'musique_ans_v1.0_dev.jsonl' -exec rm -rf {} +
fi

echo "== TripCrew base runs (fresh FX) and natural stale-FX failures"
"${RUN[@]}" -m agents.tripcrew run --count 120 --seed 7
"${RUN[@]}" -m agents.tripcrew run --count 40 --seed "$STALE_SEED" --stale-fx \
  --report data/tripcrew/report-stale.json

echo "== HopRAG base runs (deterministic reader stand-in)"
"${RUN[@]}" -m agents.hoprag run --client fixture --count 160 --read-k 2

echo "== Fault Forge: injected faults"
"${RUN[@]}" -m blackbox.forge inject --agent tripcrew --quota "$QUOTA_TRIPCREW" \
  --concurrency 4 --stale-fx-seeds "$STALE_SEED"
"${RUN[@]}" -m blackbox.forge inject --agent hoprag --quota "$QUOTA_HOPRAG" \
  --concurrency 4 --read-k 2

echo "== Natural failures (test-only, labelled by oracle counterfactual replay)"
"${RUN[@]}" -m blackbox.forge natural --agent tripcrew --stale-fx-seeds "$STALE_SEED"
"${RUN[@]}" -m blackbox.forge natural --agent hoprag --read-k 2

echo "== Freeze"
"${RUN[@]}" -m blackbox.forge freeze --agent tripcrew
"${RUN[@]}" -m blackbox.forge freeze --agent hoprag

echo "== Replay settings the API needs to rebuild each agent"
echo "{\"stale_fx_seeds\": [$STALE_SEED]}" > data/tripcrew/replay.json
echo '{"read_k": 2, "dataset": "data/hoprag/musique_ans_v1.0_dev.jsonl"}' > data/hoprag/replay.json

echo "== Train and evaluate"
"${RUN[@]}" -m blackbox.ml eval
