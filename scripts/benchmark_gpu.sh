#!/bin/bash
# Throughput benchmark on a single GPU box (no SLURM).
#
#   ./scripts/benchmark_gpu.sh
#
# Runs all three modes at increasing request rates and writes
# results/load_*.json. One mode at a time on purpose: the naive and
# batched tiers share a model object, so running them together would
# have them contend for it.
#
# Every (repeat, mode) measurement is independent: its own fresh server,
# a cooldown before it, and a rotated mode order. Throughput falls the
# longer one server has been under load -- the first mode after a restart
# was consistent across repeats, later modes on the same server lost up to
# half their ceiling -- so sharing a server between measurements makes the
# result depend on position rather than on the mode being measured.

set -euo pipefail

PORT=${PORT:-8100}
DURATION=${DURATION:-30}
# Up to 64 so the batched path's own ceiling shows up, not just naive's.
QPS_LEVELS=${QPS_LEVELS:-"1 2 4 8 16 32 64"}
# Repeat each configuration so the spread is reportable rather than a
# single sample.
REPEATS=${REPEATS:-3}
COOLDOWN=${COOLDOWN:-60}

command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: no nvidia-smi, is this a GPU box?"

# A GPU image usually ships torch system-wide; a venv on top of it would
# shadow that with a CPU build.
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available -- benchmark would measure CPU'; print('CUDA:', torch.cuda.get_device_name(0))"

mkdir -p results

SERVER_PID=""
start_server() {
  cd src
  # Access logs are one line per request and bury the result tables at
  # these rates; errors still surface.
  uvicorn server:app --port "$PORT" --host 127.0.0.1 --no-access-log >> ../results/server.log 2>&1 &
  SERVER_PID=$!
  cd ..
  until curl -sf "http://localhost:$PORT/stats" >/dev/null 2>&1; do sleep 5; done
  # First request compiles CUDA kernels; excluding it keeps that one-off
  # cost out of the measured latencies.
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

# Machine telemetry runs across the whole benchmark, including cooldowns,
# so temperature and clocks can be read against each run's results.
python scripts/monitor_server.py "http://localhost:$PORT" results/monitor.csv &
MONITOR_PID=$!
trap 'stop_server; kill $MONITOR_PID 2>/dev/null || true' EXIT

MODES=(naive batched routed)
first=1
for run in $(seq 1 "$REPEATS"); do
  # Rotate so no mode always runs first on the coolest machine.
  offset=$(( (run - 1) % ${#MODES[@]} ))
  for i in 0 1 2; do
    mode=${MODES[$(( (i + offset) % ${#MODES[@]} ))]}

    # Fresh server and a cooldown for every (run, mode), not just every
    # run: a per-run restart still left modes 2 and 3 on a server that had
    # already served ~10 minutes of load, and their ceilings dropped by up
    # to 2x while the first mode after each restart stayed consistent.
    if [ "$first" -eq 0 ]; then
      echo
      echo "cooling down ${COOLDOWN}s..."
      sleep "$COOLDOWN"
    fi
    first=0

    echo
    echo "=== $mode run $run (fresh server) ==="
    start_server
    python scripts/load_test.py \
      --url "http://localhost:$PORT/process" \
      --qps $QPS_LEVELS \
      --duration "$DURATION" \
      --mode "$mode" \
      --tag "run${run}" \
      --out "results/load_${mode}_run${run}.json"
    curl -s "http://localhost:$PORT/stats" > "results/stats_${mode}_run${run}.json"
    stop_server
  done
done

kill $MONITOR_PID 2>/dev/null || true

echo
echo "=== machine and server drift over the benchmark ==="
python scripts/summarise_monitor.py results/monitor.csv

echo
echo "=== results across runs ==="
python scripts/summarise_runs.py results

echo
echo "done. copy these back:"
ls -1 results/load_*.json results/stats_*_run*.json results/monitor.csv
