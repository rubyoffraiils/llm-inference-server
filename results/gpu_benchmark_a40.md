# GPU benchmark — A40

Open-loop load test, 30s per level, Qwen2.5-0.5B / 1.5B, 4 decode slots
per tier, queue cap 32, unique prompt per request (the response cache
would otherwise serve repeats without generating).

## Final result — fresh server per measurement, 3 runs each

Every (run, mode) measurement had its own freshly started server and a
60s cooldown before it; mode order rotated across runs. Median across 3
runs, (min–max):

| offered QPS | naive achieved / p50 | batched achieved / p50 | routed achieved / p50 |
|---|---|---|---|
| 1 | 1.0 / 0.12s | 1.0 / 0.14s | 1.0 / 0.14s |
| 8 | 8.0 / 0.67s (0.46–1.15) | 8.0 / 0.12s (0.11–0.12) | 8.0 / 0.78s |
| 16 | 9.8 / 3.47s (3.45–3.90), 185 dropped | **16.0 / 0.18s (0.16–0.20), 0 dropped** | 10.3 / 7.54s |
| 32 | 8.8 / 4.05s | 19.3 / 1.87s | 9.9 / 8.16s |
| 64 | 8.8 / 4.08s (4.06–4.11) | **18.6 / 1.97s** | 9.7 / 8.60s |

- **Batching raises the throughput ceiling 2.1×: ~8.8 → ~18.6 QPS.**
- **At 16 QPS: p50 3.47s → 0.18s, dropped requests 185 → 0.**
- Runs now agree closely (naive p50 at 64 QPS: 4.06–4.11s, versus
  5.44–10.72s when the three modes shared a server).

### Routing lowers throughput on one GPU

Routed tops out near 10 QPS, barely above naive and about half of
batched. Monitoring showed the 0.5B model's decode step (~37 ms) is no
faster than the 1.5B's (~34 ms): at this scale a decode step is bound by
CPU-side dispatch and kernel launches, not GPU compute, so the smaller
model is not cheaper to run here. Two tier schedulers on one GPU then
contend.

This means the parameter-count cost proxy (cheap = 1/3 the cost) does
not hold on this hardware. Measured per-step time puts the ratio near 1×.
The routing result that survives is quality: 93% of the expensive
model's accuracy while sending half the traffic to a model with 3× fewer
parameters. It does not translate into time or throughput savings at
this model size on a single GPU.

---

The sections below document the earlier run that exposed the
position effect and motivated the per-mode restart.

Three repeats. Each repeat started a fresh server, cooled down 120s
beforehand, and ran the three modes in rotated order:

| run | order |
|---|---|
| 1 | naive → batched → routed |
| 2 | batched → routed → naive |
| 3 | routed → naive → batched |

## Per run (p50 latency / achieved QPS)

| file | slot | 1 QPS | 4 QPS | 8 QPS | 16 QPS | 64 QPS |
|---|---|---|---|---|---|---|
| naive run1 | 1st | 0.14/1.0 | 0.16/4.0 | 3.78/7.6 | 4.64/7.8 | 5.44/6.9 |
| naive run3 | 2nd | 0.20/1.0 | 1.97/4.0 | 7.31/5.4 | 8.12/4.9 | 9.64/4.2 |
| naive run2 | 3rd | 0.26/1.0 | 4.12/4.0 | 9.56/4.0 | 9.37/4.3 | 10.72/3.9 |
| batched run2 | 1st | 0.15/1.0 | 0.15/4.0 | 0.14/8.0 | 0.52/16.0 | 2.49/14.9 |
| batched run1 | 2nd | 0.15/1.0 | 0.16/4.0 | 0.15/8.0 | 1.10/16.0 | 2.44/15.1 |
| batched run3 | 3rd | 0.18/1.0 | 0.20/4.0 | 0.30/8.0 | 3.75/9.5 | 4.30/9.0 |
| routed run3 | 1st | 0.14/1.0 | 0.19/4.0 | 0.41/8.0 | 0.83/16.0 | 5.84/13.5 |
| routed run2 | 2nd | 0.15/1.0 | 0.21/4.0 | 0.27/8.0 | 4.68/13.2 | 6.41/11.8 |
| routed run1 | 3rd | 0.14/1.0 | 0.22/4.0 | 0.22/8.0 | 2.10/15.1 | 6.34/12.8 |

**Slot on the server explains most of the spread.** Every mode did best
as the first mode after a restart. Later modes on the same server lost
throughput — naive most (ceiling ~7.5 → ~4), batched in the 3rd slot
(~15 → ~9). Restarting between repeats reset it; sharing a server between
the three modes of a repeat did not.

## Earlier estimate: first mode after each restart (superseded above)

| offered QPS | naive | batched | routed |
|---|---|---|---|
| 8 | 7.6 achieved, p50 3.78s, rejecting | 8.0, p50 0.14s | 8.0, p50 0.41s |
| 16 | 7.8, p50 4.64s | **16.0, p50 0.52s** | 16.0, p50 0.83s |
| 64 | 6.9 | **14.9** | 13.5 |

- **Batching roughly doubles the throughput ceiling: ~7.5 → ~15 QPS.**
- At 16 QPS naive saturates at 7.8 with p50 4.6s; batched serves all 16
  with p50 0.52s.
- Corroborated independently: a separate fresh-server run on an RTX 3090
  measured naive 7.8 vs batched 16.0.
- Routing tops out slightly below batched (~13.5 vs ~15): both tiers share
  one GPU and their schedulers contend. Routing's value is cost and
  quality (60% accuracy at 67% of always-expensive cost), not throughput.

n=1 per mode for the fresh-server numbers on this GPU, n=2 across GPUs.
The benchmark now restarts the server before every (repeat, mode) so
future runs give three fresh-server samples per mode.

## The slowdown, and what it isn't

Instrumented across this run and the one before it:

| candidate | result |
|---|---|
| GPU thermal / clock throttling | temp flat ~51°C, SM clock at rated 1740 MHz under load; only throttle flags were idle (0x1, cooldowns) and brief software power cap (0x4) |
| GPU memory leak / fragmentation | allocated +1%, reserved flat |
| host memory leak | peak RSS never rose after startup |
| KV-cache growth | bounded, stable or shrinking |
| attention-mask growth | bounded, tracks the cache |
| in-flight bookkeeping | stable |
| per-call hook accumulation | transformers installs capture hooks once, only on request |

Forward-pass time per decode step rose within a server's lifetime (naive
23 → 37 ms) while the GPU ran at full clock. For models this small a
decode step is bound by CPU-side work — Python dispatch and kernel
launches — rather than GPU compute, and naive (batch of one) pays that
per-step cost on every token instead of sharing it across four requests,
which fits naive degrading most. The container's host load average sat
around 35 on 9 vCPUs. So the likely source is the host CPU side, but the
exact mechanism is not identified; it is recorded as a known limitation.

An earlier explanation in this file — thermal throttling, inferred from a
laptop CPU experiment — is refuted for the GPU by the telemetry above.

## Caveats

- Cost proxy is the parameter-count ratio (3×), not measured per-token time.
- Quality numbers (46% cheap / 60% routed / 64% expensive) come from
  offline scoring and are machine-independent.
