#!/bin/bash
# Throughput benchmark on a single GPU box (no SLURM).
#
#   ./scripts/benchmark_gpu.sh
#
# Runs all three modes at increasing request rates and writes
# results/load_*.json. One mode at a time on purpose: the naive and
# batched tiers share a model object, so running them together would
# have them contend for it.

set -euo pipefail

PORT=${PORT:-8100}
DURATION=${DURATION:-30}
QPS_LEVELS=${QPS_LEVELS:-"1 2 4 8 16"}

command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: no nvidia-smi, is this a GPU box?"

# A GPU image usually ships torch system-wide; a venv on top of it would
# shadow that with a CPU build.
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- benchmark would measure CPU'; print('CUDA:', torch.cuda.get_device_name(0))"

mkdir -p results

cd src
uvicorn server:app --port "$PORT" --host 127.0.0.1 &
SERVER_PID=$!
cd ..
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for both models to load..."
until curl -sf "http://localhost:$PORT/stats" >/dev/null 2>&1; do sleep 5; done

# First request compiles CUDA kernels; excluding it keeps that one-off
# cost out of the measured latencies.
echo "warming up..."
curl -sf -X POST "http://localhost:$PORT/process" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Question: warmup\n\nAnswer with as few words as possible, no explanation.","mode":"batched"}' >/dev/null

for mode in naive batched routed; do
  echo
  echo "=== $mode ==="
  python scripts/load_test.py \
    --url "http://localhost:$PORT/process" \
    --qps $QPS_LEVELS \
    --duration "$DURATION" \
    --mode "$mode" \
    --out "results/load_${mode}.json"
  curl -s "http://localhost:$PORT/stats" > "results/stats_${mode}.json"
done

echo
echo "done. copy these back:"
ls -1 results/load_*.json results/stats_*.json
