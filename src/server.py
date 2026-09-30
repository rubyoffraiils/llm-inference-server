"""FastAPI app serving two model tiers behind a difficulty router."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import CHEAP_MODEL, EXPENSIVE_MODEL, load_model
from router import route
from scheduler import BatchingScheduler, _percentile

Mode = Literal["naive", "batched", "routed"]

# Routing decisions, so the cheap/expensive split and the router's own
# overhead are visible live rather than only in a benchmark run.
_routing_counts = {"cheap": 0, "expensive": 0}
_router_latencies: deque[float] = deque(maxlen=1000)

# Each tier gets its own queue and scheduler. Sharing one queue would let
# a burst of expensive requests block cheap ones behind it while the
# cheap model sits idle -- head-of-line blocking across tiers.
_tiers: dict[str, BatchingScheduler] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cheap_model, cheap_tokenizer = load_model(CHEAP_MODEL)
    expensive_model, expensive_tokenizer = load_model(EXPENSIVE_MODEL)

    _tiers["cheap"] = BatchingScheduler(cheap_model, cheap_tokenizer)
    # Batch size 1 turns the scheduler into a one-at-a-time path, which
    # is the naive baseline the benchmark measures batching against.
    _tiers["naive"] = BatchingScheduler(
        expensive_model, expensive_tokenizer, max_batch_size=1
    )
    _tiers["expensive"] = BatchingScheduler(expensive_model, expensive_tokenizer)

    for scheduler in _tiers.values():
        scheduler.start()
    yield
    for scheduler in _tiers.values():
        await scheduler.stop()
    _tiers.clear()


app = FastAPI(lifespan=lifespan)

_STATIC = Path(__file__).resolve().parent.parent / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/")
    async def demo_page() -> FileResponse:
        return FileResponse(_STATIC / "index.html")


class ProcessRequest(BaseModel):
    prompt: str
    # The benchmark forces each configuration on identical traffic, so
    # the batching gain and the routing gain can be attributed separately.
    mode: Mode = "routed"


class ProcessResponse(BaseModel):
    output: str
    tier: str
    router_latency_ms: float


def _process_stats() -> dict:
    """Memory held by this process, for diagnosing slowdown over time.

    Reserved climbing while allocated stays flat is allocator
    fragmentation; both climbing is a leak. Read here rather than from a
    monitoring process, which would only see its own memory.
    """
    import resource
    import sys

    import torch

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = raw / 2**20 if sys.platform == "darwin" else raw / 1024
    if not torch.cuda.is_available():
        return {"host_peak_rss_mib": round(rss_mib, 1)}
    return {
        "host_peak_rss_mib": round(rss_mib, 1),
        "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
    }


@app.get("/stats")
async def stats() -> dict:
    """Per-tier queue and latency numbers, plus a combined view.

    Reported per tier because the point of routing is that the cheap
    path stays fast while the expensive one is slow -- one blended
    number describes neither.
    """
    per_tier = {name: scheduler.stats() for name, scheduler in _tiers.items()}
    routed = _routing_counts["cheap"] + _routing_counts["expensive"]
    return {
        "tiers": per_tier,
        "process": _process_stats(),
        "routing": {
            **_routing_counts,
            "fraction_cheap": _routing_counts["cheap"] / routed if routed else 0.0,
            "router_latency_p50_ms": _percentile(_router_latencies, 0.50),
            "router_latency_p95_ms": _percentile(_router_latencies, 0.95),
        },
        "overall": {
            "requests_total": sum(t["requests_total"] for t in per_tier.values()),
            "requests_rejected": sum(t["requests_rejected"] for t in per_tier.values()),
            "queue_depth": sum(t["queue_depth"] for t in per_tier.values()),
            "tokens_per_second": sum(t["tokens_per_second"] for t in per_tier.values()),
        },
    }


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    if request.mode == "routed":
        decision = route(request.prompt)
        tier, router_latency_ms = decision.tier, decision.latency_ms
        _routing_counts[tier] += 1
        _router_latencies.append(router_latency_ms)
    else:
        # naive and batched both always use the expensive tier; they
        # differ only in whether requests are batched together.
        tier, router_latency_ms = (
            "naive" if request.mode == "naive" else "expensive",
            0.0,
        )

    try:
        result = await _tiers[tier].submit(request.prompt)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server at capacity, retry shortly")

    return ProcessResponse(
        output=result.text, tier=tier, router_latency_ms=router_latency_ms
    )
