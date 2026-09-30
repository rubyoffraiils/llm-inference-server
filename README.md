# llm-inference-server

a small llm inference server i built to learn and play around w batching and see how it works.

**demo:** [ruby-zhou.vercel.app/projects/llm-inference-server](https://ruby-zhou.vercel.app/projects/llm-inference-server)

## how it works

```
                        ┌──> small model (qwen2.5-0.5b) ──> batching scheduler ──┐
request ──> router ─────┤                                                        ├──> answer
                        └──> big model   (qwen2.5-1.5b) ──> batching scheduler ──┘
```

- **decode loop**: written by hand w/ kv cache reuse
- **batching**: 4 slots per model, all advanced in one forward pass. a finished request frees its slot and the next one jumps in right away
- **router**: sends a question to the big model if it doesn't come with a context passage
- **extras**: queue cap (503 when full), response cache, dedup for identical requests, `/stats` for latency + queue numbers

## results (nvidia a40)

| | one at a time | batched |
|---|---|---|
| max throughput | ~8.8 req/s | **~18.6 req/s** |
| p50 latency @ 16 req/s | 3.47s | **0.18s** |
| dropped requests @ 16 req/s | 185 | **0** |

- routing got **93%** of the big model's accuracy while sending half the traffic to the small one
- also tried jev (a hosted decision model) as the router. my one rule did at least as well, at ~0.002ms vs ~180ms per decision
- full numbers in [`results/`](results/) !
