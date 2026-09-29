# GPU benchmark — RTX 3090

Open-loop load test, 30s per level, Qwen2.5-0.5B (cheap) / Qwen2.5-1.5B
(expensive), 4 decode slots per tier, queue cap 32.

## Naive vs batched (same model, so this isolates batching)

| offered QPS | naive p50 | naive p95 | naive rejected | batched p50 | batched p95 | batched rejected |
|---|---|---|---|---|---|---|
| 1 | 0.14s | 0.28s | 0 | 0.15s | 0.31s | 0 |
| 2 | 0.14s | 0.47s | 0 | 0.15s | 0.60s | 0 |
| 4 | 0.14s | 0.55s | 0 | 0.15s | 0.53s | 0 |
| 8 | 3.86s | 5.22s | 12 | 0.15s | 0.52s | 0 |
| 16 | 4.65s | 5.98s | 245 | 0.71s | 1.42s | 0 |

**Naive saturates at ~7.8 QPS.** At 16 offered it achieves 7.8 and rejects
245 of 480 requests.

**Batched sustains 16 QPS with zero rejections.**

- sustained throughput: **7.8 → 16 QPS (2.0×)**
- p50 latency at 16 QPS: **4.65s → 0.71s (6.5× better)**
- rejections at 16 QPS: **245 → 0**

Below 4 QPS the two are identical, which is the expected shape: batching
does nothing until there's contention to batch.

## Routed

| offered QPS | p50 | p95 | rejected |
|---|---|---|---|
| 1 | 0.15s | 0.36s | 0 |
| 2 | 0.15s | 0.43s | 0 |
| 4 | 0.17s | 0.53s | 0 |
| 8 | 0.24s | 0.76s | 0 |
| 16 | 1.77s | 5.95s | 28 |

Tracks batched up to 8 QPS, then degrades slightly. Expected: routing
sends ~half the traffic to the 1.5B model, so it trades a little
throughput for the cost and quality result.

## Quality (held-out, 200 questions, machine-independent)

| mode | accuracy | relative cost |
|---|---|---|
| always cheap | 46% | 0.33 |
| routed | 60% | 0.67 |
| always expensive | 64% | 1.00 |

93% of always-expensive quality at 67% of the cost.

## Caveats

- One run per configuration. The design doc asks for 3 repeats with the
  spread reported; this is a single pass, so treat small differences as
  noise.
- First A40 run's numbers are not comparable and aren't used here.
- Cost is the parameter-count ratio (3×), not measured per-token time.
