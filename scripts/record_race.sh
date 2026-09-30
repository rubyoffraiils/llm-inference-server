#!/bin/bash
# Record the demo race on a GPU box and write static/race.json.
#
#   ./scripts/record_race.sh
#
# Each lane gets its own fresh server so neither inherits the other's
# load, the same rule the throughput benchmark follows.

set -euo pipefail

PORT=${PORT:-8100}
N=${N:-24}

if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- the race would show CPU speed'"

mkdir -p results
SERVER_PID=""

start_server() {
  cd src
  uvicorn server:app --port "$PORT" --host 127.0.0.1 --no-access-log >> ../results/race_server.log 2>&1 &
  SERVER_PID=$!
  cd ..
  until curl -sf "http://localhost:$PORT/stats" >/dev/null 2>&1; do sleep 5; done
  # Compile kernels before the timed burst.
  curl -sf -X POST "http://localhost:$PORT/process" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Question: warmup\n\nAnswer with as few words as possible, no explanation.","mode":"batched"}' >/dev/null
}

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

for mode in naive batched; do
  echo "recording $mode lane on a fresh server..."
  start_server
  python scripts/record_race.py \
    --url "http://localhost:$PORT/process" \
    --mode "$mode" --n "$N" --nonce "$mode-$(date +%s)" \
    --out "results/race_${mode}.json"
  stop_server
  sleep 30
done

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_NAME="$GPU_NAME" N="$N" python - <<'EOF'
import json, os, datetime
lanes = {m: json.load(open(f"results/race_{m}.json"))["completions_s"] for m in ("naive", "batched")}
out = {
    "recorded_at": datetime.date.today().isoformat(),
    "gpu": os.environ["GPU_NAME"],
    "requests": int(os.environ["N"]),
    "lanes": lanes,
}
json.dump(out, open("static/race.json", "w"), indent=2)
print(f"wrote static/race.json  naive {lanes['naive'][-1]:.2f}s  batched {lanes['batched'][-1]:.2f}s")
EOF
