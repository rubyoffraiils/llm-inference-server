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

Run 1 was fast across every mode; runs 2 and 3 were several times slower.
**Not hardware**: `nvidia-smi` after the run showed 33°C, 0% utilization,
no other processes. No thermal throttling, no noisy neighbour.

The load generator restarts between runs but **the server process does
not**, so whatever degrades persists across runs. The prime suspect is the
KV-cache growth documented in `notes/week-2-batching.md`: leading columns
can't be reclaimed while a slot that joined an empty batch is still
running, so under ~40 minutes of continuous load the cache accumulates and
every forward pass gets slower.

This is a genuine limitation of the implementation, not a measurement
artifact. Reporting the median plus the range is the honest presentation;
quoting run 1 alone would be cherry-picking.

## What holds up regardless

The *comparison* survives the degradation because it hits all three modes
equally. In every individual run, batched sustained roughly 2× naive's
throughput and rejected far fewer requests. That ratio is the claim; the
absolute numbers depend on how long the server has been up.

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
