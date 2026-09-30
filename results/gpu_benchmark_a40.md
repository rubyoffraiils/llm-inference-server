# GPU benchmark — A40, 3 runs per mode

Open-loop load test, 30s per level, Qwen2.5-0.5B / 1.5B, 4 decode slots
per tier, queue cap 32. Median across 3 runs, with (min-max) range.

## Throughput (achieved QPS, median of 3)

| offered | naive | batched | routed |
|---|---|---|---|
| 1 | 1.0 | 1.0 | 1.0 |
| 2 | 2.0 | 2.0 | 2.0 |
| 4 | 4.0 | 4.0 | 4.0 |
| 8 | 4.2 | **8.0** | 8.0 |
| 16 | 4.4 | **8.8** | 11.7 |
| 32 | 4.0 | **8.3** | 10.7 |
| 64 | 3.7 | **7.4** | 10.3 |

**Batching roughly doubles sustained throughput.** Naive plateaus around
4 QPS, batched around 8.

Rejections at 8 QPS: naive 114, batched 0.

## The spread is large, and that matters

| mode, level | p50 range across 3 runs |
|---|---|
| naive, 4 QPS | 0.14s – 7.07s |
| batched, 8 QPS | 0.12s – 3.29s |
| batched, 16 QPS | 0.20s – 5.30s |

Run 1 was fast across every mode; runs 2 and 3 were slower. The cleanest
view is at 1 QPS, where nothing queues: naive p50 went 0.14s → 0.25s, a
~1.8× slowdown in per-request service time. The larger swings at higher
rates are queueing amplifying that — 1.8× slower service pushes naive's
capacity below the offered load, and past capacity latency grows without
bound.

## What caused it

An instrumented re-run sampled server state every 10s for the whole run.
Ruled out, by measurement:

| candidate | result |
|---|---|
| GPU memory leak | allocated +1%, reserved +0% |
| allocator fragmentation | reserved flat while allocated flat |
| host memory leak | peak RSS never rose after startup |
| KV-cache growth | bounded; stable or shrinking per tier |
| attention-mask growth | tracks the cache, bounded |
| in-flight bookkeeping | stable at the number of live requests |
| per-call hook accumulation | transformers installs capture hooks once, only on request |

A load–idle–load experiment then separated process state from machine
state (`scripts/diagnose_degradation.py`, forward-pass time per window):

| condition | forward pass |
|---|---|
| start of sustained load | 37.5 ms |
| after 2.5 min of load | 44.7 ms |
| same process, after 60s idle | 37.3 ms — recovered |
| fresh process, machine still warm | 39.6 → 49.3 ms — not recovered |

Idling restores speed without a restart; a restart without idling does
not. That is machine state, not server state: under sustained load the
machine slows (thermal/frequency behaviour), and recovers at rest. There
is no accumulating bug in the server.

That experiment ran on a laptop CPU. The GPU-side mechanism is not yet
measured directly — the earlier "not hardware" reading was an
`nvidia-smi` snapshot taken after the run, on an idle GPU, which says
nothing about clocks under load. The benchmark now logs SM clock,
temperature, power and throttle reasons throughout.

## What was wrong with this run's method

- **One server, 40 minutes of back-to-back load, no rest.** Later runs
  measured a hot machine.
- **Fixed mode order.** Naive always ran first in each repeat, so any
  drift systematically favoured naive — the degradation did *not* hit
  all three modes equally.

`scripts/benchmark_gpu.sh` now starts a fresh server per repeat, cools
down between repeats, and rotates the mode order so each mode takes each
slot once. **These A40 numbers are superseded pending that re-run.** The
direction holds — batched beat naive in every individual run — but the
magnitudes here are not ones to quote.

## Earlier single run (RTX 3090)

A previous one-run-per-mode benchmark on a 3090 reported naive 7.8 QPS vs
batched 16 QPS. Those came from a freshly started server and are **not
comparable** to these — different GPU, and no repeats to show the spread.
Superseded by this run.

## Caveats

- Cost proxy is the parameter-count ratio (3×), not measured per-token time.
- Quality numbers (46% cheap / 60% routed / 64% expensive) come from
  offline scoring and are machine-independent.
- Three runs is enough to see the spread, not enough for tight confidence
  intervals.
