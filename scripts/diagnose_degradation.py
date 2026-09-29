"""Watch whether the KV-cache grows under sustained overlapping load.

The A40 benchmark degraded across runs while the GPU stayed idle and
cool, which points at the server rather than the hardware. This keeps
the batch continuously occupied -- the condition where padding columns
accumulate -- and samples the cache while requests are in flight, since
a drained batch resets it to zero and hides the problem.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import load_model
from scheduler import BatchingScheduler

SHORT = "Question: What is 2 + 2?\n\nAnswer with as few words as possible, no explanation."
LONG = "Question: Write a detailed history of the Roman Empire.\n\nAnswer as fully as you can."


async def sampler(scheduler: BatchingScheduler, samples: list, stop: asyncio.Event):
    """Record cache length while the batch is busy, not after it drains."""
    while not stop.is_set():
        stats = scheduler.stats()
        if stats["active_slots"] > 0:
            samples.append((time.perf_counter(), stats["kv_cache_columns"]))
        await asyncio.sleep(0.25)


async def main() -> None:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4, max_queue_size=64)
    scheduler.start()

    samples: list[tuple[float, int]] = []
    stop = asyncio.Event()
    watcher = asyncio.create_task(sampler(scheduler, samples, stop))

    started = time.perf_counter()
    counter = 0
    inflight: set[asyncio.Task] = set()

    try:
        # Keep the batch permanently occupied: top up as soon as anything
        # finishes, so it never fully drains and never resets.
        while time.perf_counter() - started < duration:
            while len(inflight) < 6:
                counter += 1
                prompt = LONG if counter % 4 == 0 else SHORT
                task = asyncio.create_task(scheduler.submit(f"{prompt} #{counter}"))
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            await asyncio.sleep(0.1)

        stop.set()
        await watcher
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
    finally:
        await scheduler.stop()

    if not samples:
        print("no samples taken while slots were busy")
        return

    base = samples[0][0]
    print(f"{'t (s)':>7}{'kv_cols':>9}")
    for timestamp, columns in samples[:: max(1, len(samples) // 20)]:
        print(f"{timestamp - base:>7.1f}{columns:>9}")

    columns = [c for _, c in samples]
    print()
    print(f"samples {len(columns)}  min {min(columns)}  max {max(columns)}")
    first, last = columns[: len(columns) // 4], columns[-len(columns) // 4 :]
    print(f"first quarter mean {sum(first) / len(first):.0f}")
    print(f"last  quarter mean {sum(last) / len(last):.0f}")
    print(f"requests completed: {scheduler.stats()['completed_in_window']}")


if __name__ == "__main__":
    asyncio.run(main())
