import asyncio

from model import load_model
from scheduler import BatchingScheduler

PROMPT = "What is the capital of Japan?"
OTHER = "2 + 2 ="


async def test_repeat_prompt_is_served_from_cache():
    """Greedy decoding is deterministic, so a cached answer is exactly
    what regenerating would produce -- not an approximation.
    """
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer)
    scheduler.start()
    try:
        first = await scheduler.submit(PROMPT)
        second = await scheduler.submit(PROMPT)
    finally:
        await scheduler.stop()

    assert first.text == second.text
    stats = scheduler.stats()
    assert stats["cache_hits"] == 1
    assert stats["requests_total"] == 2
    # Only the first request reached a decode slot.
    assert stats["completed_in_window"] == 1


async def test_concurrent_duplicates_share_one_slot():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer)
    scheduler.start()
    try:
        results = await asyncio.gather(
            scheduler.submit(PROMPT),
            scheduler.submit(PROMPT),
            scheduler.submit(PROMPT),
        )
    finally:
        await scheduler.stop()

    assert results[0].text == results[1].text == results[2].text
    stats = scheduler.stats()
    assert stats["dedup_hits"] == 2, "duplicates should attach, not queue"
    assert stats["completed_in_window"] == 1, "only one generation should run"


async def test_distinct_prompts_are_not_deduplicated():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer)
    scheduler.start()
    try:
        first, second = await asyncio.gather(
            scheduler.submit(PROMPT), scheduler.submit(OTHER)
        )
    finally:
        await scheduler.stop()

    assert first.text != second.text
    stats = scheduler.stats()
    assert stats["dedup_hits"] == 0
    assert stats["cache_hits"] == 0
    assert stats["completed_in_window"] == 2


async def test_cache_is_bounded():
    """Unbounded caching is the same leak the queue used to have."""
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, response_cache_size=2)
    scheduler.start()
    try:
        for prompt in ("2 + 2 =", "3 + 3 =", "4 + 4 ="):
            await scheduler.submit(prompt)
    finally:
        await scheduler.stop()

    assert scheduler.stats()["cache_entries"] == 2


async def test_evicted_entry_regenerates_rather_than_erroring():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer, response_cache_size=1)
    scheduler.start()
    try:
        first = await scheduler.submit(PROMPT)
        await scheduler.submit(OTHER)  # evicts PROMPT
        again = await scheduler.submit(PROMPT)
    finally:
        await scheduler.stop()

    assert again.text == first.text
    assert scheduler.stats()["cache_hits"] == 0, "PROMPT should have been evicted"
