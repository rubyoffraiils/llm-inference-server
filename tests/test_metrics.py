import asyncio

from model import load_model
from scheduler import BatchingScheduler, _percentile


def test_percentile_picks_the_tail_not_the_mean():
    """The case percentiles exist for: one slow request among fast ones."""
    values = [0.1] * 9 + [5.0]
    assert _percentile(values, 0.50) == 0.1
    assert _percentile(values, 0.95) == 5.0
    # The mean would be 0.59 -- describing none of the ten requests.


def test_percentile_handles_empty_and_single():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.5], 0.95) == 1.5


def test_stats_are_reported_before_any_traffic():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4)
    s = scheduler.stats()
    assert s["queue_depth"] == 0
    assert s["active_slots"] == 0
    assert s["max_slots"] == 4
    assert s["requests_total"] == 0
    assert s["latency_p50_ms"] == 0.0


async def test_stats_record_completed_requests():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4)
    scheduler.start()
    try:
        await asyncio.gather(
            scheduler.submit("What is the capital of Japan?"),
            scheduler.submit("2 + 2 ="),
        )
    finally:
        await scheduler.stop()

    s = scheduler.stats()
    assert s["requests_total"] == 2
    assert s["completed_in_window"] == 2
    assert s["latency_p50_ms"] > 0
    assert s["tokens_per_second"] > 0
    # Everything drained, so nothing should still be held.
    assert s["active_slots"] == 0
    assert s["queue_depth"] == 0


async def test_queue_wait_is_recorded_separately_from_generation():
    """A request that waits for a slot must show it in queue_wait, or
    there's no way to tell capacity problems from slow generation.
    """
    model, tokenizer = load_model()
    # One slot, so the second request has to wait for the first.
    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=1)
    scheduler.start()
    try:
        await asyncio.gather(
            scheduler.submit("Write a detailed history of Rome."),
            scheduler.submit("2 + 2 ="),
        )
    finally:
        await scheduler.stop()

    s = scheduler.stats()
    assert s["queue_wait_p95_ms"] > 0, "second request waited but queue_wait is zero"


async def test_rejected_requests_are_counted():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(
        model, tokenizer, max_batch_size=1, max_queue_size=1
    )
    scheduler.start()
    try:
        first = asyncio.create_task(scheduler.submit("Write a long history of Rome."))
        await asyncio.sleep(0.3)
        second = asyncio.create_task(scheduler.submit("Write a long history of Rome."))
        await asyncio.sleep(0.1)
        try:
            await scheduler.submit("2 + 2 =")
        except asyncio.QueueFull:
            pass
        first.cancel()
        second.cancel()
    finally:
        await scheduler.stop()

    assert scheduler.stats()["requests_rejected"] == 1
