"""Separate process-state slowdown from machine-state slowdown.

The A40 benchmark showed unloaded latency rising ~1.8x over a run while
GPU memory, host RSS and KV-cache length all stayed flat. Two families
of cause fit that: something in the server process getting slower
(a code bug), or the machine running hot or contended (environment).

This runs load, then idles, then loads again, reporting per-window
forward-pass time. If idling in the same process restores speed, the
cause is the machine. If only a fresh process does, it is the process.
Prompts are fixed-length so later requests don't cost more by accident.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import load_model
from scheduler import BatchingScheduler

# Fixed-width so every prompt tokenizes to the same length: the previous
# version's growing counter made later prompts longer and slower.
PROMPT = "Question: What is {n:08d} plus {m:08d}?\n\nAnswer with as few words as possible, no explanation."
WINDOW_S = 15.0


async def load_phase(scheduler, seconds: float, counter: list[int], label: str) -> None:
    started = time.perf_counter()
    next_report = started + WINDOW_S
    inflight: set[asyncio.Task] = set()
    while time.perf_counter() - started < seconds:
        while len(inflight) < 6:
            counter[0] += 1
            n = counter[0]
            task = asyncio.create_task(scheduler.submit(PROMPT.format(n=n, m=n * 7)))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
        await asyncio.sleep(0.05)
        if time.perf_counter() >= next_report:
            next_report += WINDOW_S
            report(scheduler, label)
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    report(scheduler, label)


def report(scheduler, label: str) -> None:
    # Drain the samples each window so every line is that window alone,
    # not a rolling mix with older, faster steps.
    forward = list(scheduler._forward_times)
    scheduler._forward_times.clear()
    if not forward:
        return
    print(
        f"{label:<14}{statistics.median(forward) * 1000:>10.1f}"
        f"{len(forward):>9}{scheduler.stats()['kv_cache_columns']:>6}",
        flush=True,
    )


async def main() -> None:
    load_s = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    idle_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4, max_queue_size=64)
    scheduler.start()
    counter = [0]

    print(f"{'phase':<14}{'fwd ms':>10}{'steps':>9}{'kv':>6}", flush=True)
    try:
        await load_phase(scheduler, load_s, counter, "load")
        print(f"{'idle ' + str(int(idle_s)) + 's':<14}", flush=True)
        await asyncio.sleep(idle_s)
        await load_phase(scheduler, 60.0, counter, "after idle")
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
