#!/bin/bash
# Throughput benchmark on one exclusive GPU.
#
# Exclusive rather than sharded: a shared GPU makes tokens/sec depend on
# whoever else is on the card, which isn't reproducible and isn't a
# number worth reporting. The run is minutes, so it holds the card briefly.
#
#   sbatch scripts/benchmark_gpu.sbatch
#
#SBATCH --job-name=llm-inference-bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=compute
#SBATCH --output=results/bench-%j.out

set -euo pipefail

PORT=8100
MODES=(naive batched routed)
QPS_LEVELS="1 2 4 8 16"
DURATION=30

nvidia-smi

source venv/bin/activate

cd src
uvicorn server:app --port "$PORT" &
SERVER_PID=$!
cd ..
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Both models have to load and the first request warms up CUDA kernels.
until curl -sf "http://localhost:$PORT/stats" >/dev/null; do sleep 5; done
curl -sf -X POST "http://localhost:$PORT/process" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Question: warmup\n\nAnswer with as few words as possible, no explanation.","mode":"batched"}' >/dev/null

mkdir -p results
for mode in "${MODES[@]}"; do
  echo "=== $mode ==="
  # One mode at a time: the naive and batched tiers share a model, so
  # running them concurrently would have them contend for it.
  python scripts/load_test.py \
    --url "http://localhost:$PORT/process" \
    --qps $QPS_LEVELS \
    --duration "$DURATION" \
    --mode "$mode" \
    --out "results/load_${mode}.json"
  curl -s "http://localhost:$PORT/stats" > "results/stats_${mode}.json"
done

echo "done -- results/load_*.json, results/stats_*.json"
