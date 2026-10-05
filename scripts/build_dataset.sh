#!/usr/bin/env bash
# Build a travel-only corpus with independent tasks, replay-labelled faults and evaluation.
set -euo pipefail

RUN=(uv run --locked --extra dev --extra ml --extra server python)
DATA_DIR=${DATA_DIR:-data/tripcrew}
EVAL_DIR=${EVAL_DIR:-data/eval}
MODEL_DIR=${MODEL_DIR:-data/models/diagnoser-v2}
TRAIN_DATA_DIRS=${TRAIN_DATA_DIRS:-$DATA_DIR}
FRESH_SEEDS=(${FRESH_SEEDS:-7 13 17 23 29})
STALE_SEEDS=(${STALE_SEEDS:-11 19 31})
FRESH_COUNT=${FRESH_COUNT:-300}
STALE_COUNT=${STALE_COUNT:-80}
FAULT_QUOTA=${FAULT_QUOTA:-100}
FAULT_SAMPLES=${FAULT_SAMPLES:-3}
TRAVEL_OPERATORS=${TRAVEL_OPERATORS:-T1,T2,T3,T4,T5,T6,D1,D2,D3,D4,C1,C2,C3,C4}
STALE_CSV=$(IFS=,; echo "${STALE_SEEDS[*]}")

if [[ "${1:-}" == "--fresh" ]]; then
  # DATA_DIR is explicit and configurable; use a new directory to preserve a prior corpus.
  rm -rf "$DATA_DIR" "$EVAL_DIR" "$MODEL_DIR"
fi

echo "== TripCrew base runs (fresh FX) -> $DATA_DIR"
for seed in "${FRESH_SEEDS[@]}"; do
  "${RUN[@]}" -m agents.tripcrew run --count "$FRESH_COUNT" --seed "$seed" \
    --data-dir "$DATA_DIR" --report "$DATA_DIR/report-$seed.json"
done

echo "== TripCrew natural stale-FX failures"
for seed in "${STALE_SEEDS[@]}"; do
  "${RUN[@]}" -m agents.tripcrew run --count "$STALE_COUNT" --seed "$seed" --stale-fx \
    --data-dir "$DATA_DIR" --report "$DATA_DIR/report-stale-$seed.json"
done

echo "== Fault Forge: travel-relevant operators only"
"${RUN[@]}" -m blackbox.forge inject --agent tripcrew --data-dir "$DATA_DIR" \
  --quota "$FAULT_QUOTA" --operators "$TRAVEL_OPERATORS" --samples "$FAULT_SAMPLES" \
  --concurrency 4 --stale-fx-seeds "$STALE_CSV"

echo "== Attribute natural failures (test-only oracle replay)"
"${RUN[@]}" -m blackbox.forge natural --agent tripcrew --data-dir "$DATA_DIR" \
  --stale-fx-seeds "$STALE_CSV"

echo "== Freeze travel corpus"
"${RUN[@]}" -m blackbox.forge freeze --agent tripcrew --data-dir "$DATA_DIR"

echo "== Replay settings for the API"
mkdir -p "$DATA_DIR"
echo "{\"stale_fx_seeds\": [$STALE_CSV]}" > "$DATA_DIR/replay.json"

echo "== Train and evaluate over: $TRAIN_DATA_DIRS"
IFS=, read -r -a TRAIN_DIR_ARRAY <<< "$TRAIN_DATA_DIRS"
DATA_ARGS=()
for directory in "${TRAIN_DIR_ARRAY[@]}"; do
  DATA_ARGS+=(--data-dir "$directory")
done
"${RUN[@]}" -m blackbox.ml eval "${DATA_ARGS[@]}" --out-dir "$EVAL_DIR" \
  --model-dir "$MODEL_DIR"
