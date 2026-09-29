"""Open-loop load test: fire requests at a fixed rate and watch latency.

Open loop, not closed: requests go out on a schedule regardless of
whether earlier ones have come back. A closed loop (keep N in flight)
throttles itself to whatever the server can handle, which hides the
overload behaviour this is meant to find.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import load_mixed_split

DEFAULT_URL = "http://localhost:8000/process"


@dataclass
class Outcome:
    latency: float
    ok: bool
    rejected: bool = False
    tier: str = ""


@dataclass
class LevelResult:
    qps: int
    duration: float
    outcomes: list[Outcome] = field(default_factory=list)

    def summary(self) -> dict:
        done = [o for o in self.outcomes if o.ok]
        rejected = sum(o.rejected for o in self.outcomes)
        errors = sum(not o.ok and not o.rejected for o in self.outcomes)
        latencies = sorted(o.latency for o in done)

        def pct(fraction: float) -> float:
            if not latencies:
                return 0.0
            return latencies[min(int(fraction * len(latencies)), len(latencies) - 1)]

        return {
            "offered_qps": self.qps,
            # Achieved diverges from offered once the server can't keep up;
            # reporting only the offered rate hides overload entirely.
            "achieved_qps": len(done) / self.duration if self.duration else 0.0,
            "sent": len(self.outcomes),
            "completed": len(done),
            "rejected": rejected,
            "errors": errors,
            "latency_p50_s": pct(0.50),
            "latency_p95_s": pct(0.95),
            "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        }


async def _fire(client: httpx.AsyncClient, url: str, prompt: str, mode: str) -> Outcome:
    start = time.perf_counter()
    try:
        response = await client.post(
            url, json={"prompt": prompt, "mode": mode}, timeout=300.0
        )
        latency = time.perf_counter() - start
        if response.status_code == 503:
            return Outcome(latency=latency, ok=False, rejected=True)
        response.raise_for_status()
        return Outcome(
            latency=latency, ok=True, tier=response.json().get("tier", "")
        )
    except Exception:
        return Outcome(latency=time.perf_counter() - start, ok=False)


async def run_level(
    url: str, prompts: list[str], qps: int, duration: float, mode: str
) -> LevelResult:
    result = LevelResult(qps=qps, duration=duration)
    interval = 1.0 / qps
    tasks: list[asyncio.Task] = []

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        sent = 0
        while time.perf_counter() - start < duration:
            prompt = prompts[sent % len(prompts)]
            tasks.append(asyncio.create_task(_fire(client, url, prompt, mode)))
            sent += 1
            # Sleep to the next scheduled send time rather than a flat
            # interval, so slow task creation doesn't drift the rate down.
            next_send = start + sent * interval
            delay = next_send - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)

        result.outcomes = list(await asyncio.gather(*tasks))

    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--qps", type=int, nargs="+", default=[1, 2, 4, 8], help="rates to test"
    )
    parser.add_argument("--duration", type=float, default=20.0, help="seconds per rate")
    parser.add_argument("--mode", default="routed", choices=["naive", "batched", "routed"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dev, _ = load_mixed_split(200, 200)
    prompts = [ex.as_prompt() for ex in dev]
    random.Random(0).shuffle(prompts)

    print(f"mode={args.mode}  {args.duration:.0f}s per level  {args.url}\n")
    header = f"{'offered':>8}{'achieved':>10}{'p50 s':>9}{'p95 s':>9}{'done':>7}{'rej':>6}{'err':>6}"
    print(header)
    print("-" * len(header))

    summaries = []
    for qps in args.qps:
        result = await run_level(args.url, prompts, qps, args.duration, args.mode)
        s = result.summary()
        summaries.append(s)
        print(
            f"{s['offered_qps']:>8}{s['achieved_qps']:>10.2f}"
            f"{s['latency_p50_s']:>9.2f}{s['latency_p95_s']:>9.2f}"
            f"{s['completed']:>7}{s['rejected']:>6}{s['errors']:>6}"
        )
        # Let queues drain so the next level starts from a clean slate.
        await asyncio.sleep(3)

    _report_knee(summaries)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"mode": args.mode, "levels": summaries}, indent=2))
        print(f"\nwrote {args.out}")


def _report_knee(summaries: list[dict]) -> None:
    """Find where the server stops keeping up with offered load.

    Latency growth is the signal, not achieved QPS: the client waits as
    long as it takes, so everything eventually completes and achieved
    tracks offered even when requests are queued for a minute. A server
    past capacity shows it by latency climbing with each level.
    """
    print()
    if not summaries:
        return

    healthy = []
    for previous, current in zip(summaries, summaries[1:]):
        grew = current["latency_p50_s"] > 1.5 * previous["latency_p50_s"]
        if grew or current["rejected"]:
            break
        healthy.append(current)

    first_level = summaries[0]
    if not healthy and len(summaries) > 1:
        print(
            f"saturated from the start: p50 climbs "
            f"{first_level['latency_p50_s']:.1f}s -> {summaries[-1]['latency_p50_s']:.1f}s "
            f"across {first_level['offered_qps']}-{summaries[-1]['offered_qps']} QPS. "
            "Queue is growing; capacity is below the lowest rate tested."
        )
        return

    best = healthy[-1] if healthy else first_level
    print(
        f"holds up to ~{best['offered_qps']} QPS "
        f"(p50 {best['latency_p50_s']:.1f}s, p95 {best['latency_p95_s']:.1f}s)"
    )
    beyond = [s for s in summaries if s["offered_qps"] > best["offered_qps"]]
    if beyond:
        worst = beyond[-1]
        print(
            f"past that, latency grows with load: "
            f"{worst['offered_qps']} QPS -> p50 {worst['latency_p50_s']:.1f}s, "
            f"{worst['rejected']} rejected"
        )


if __name__ == "__main__":
    asyncio.run(main())
