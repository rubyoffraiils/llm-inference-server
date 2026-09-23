import asyncio

import pytest

from model import generate, load_model
from scheduler import BatchingScheduler

LONG_PROMPT = "Write a long detailed poem about the sea."
SHORT_PROMPT = "What is the capital of Japan?"


async def test_queue_full_rejects_instead_of_hanging():
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(
        model, tokenizer, max_batch_size=1, max_queue_size=1
    )
    scheduler.start()
    try:
        first = asyncio.create_task(scheduler.submit(LONG_PROMPT))
        await asyncio.sleep(0.5)  # let it occupy the slot
        second = asyncio.create_task(scheduler.submit(LONG_PROMPT))
        await asyncio.sleep(0.1)  # let it take the one queue place

        with pytest.raises(asyncio.QueueFull):
            await scheduler.submit(SHORT_PROMPT)

        first.cancel()
        second.cancel()
    finally:
        await scheduler.stop()


async def test_slot_churn_does_not_corrupt_a_long_running_request():
    """Short requests joining and leaving around a long one must not
    change its output, however the cache is reshaped underneath it.
    """
    model, tokenizer = load_model()
    expected_long = generate(LONG_PROMPT, model, tokenizer, max_new_tokens=50).text

    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4)
    scheduler.start()
    try:
        long_task = asyncio.create_task(scheduler.submit(LONG_PROMPT))
        # Several short requests cycle through the other slots while the
        # long one runs, so slots free repeatedly and trimming kicks in.
        for _ in range(3):
            await asyncio.sleep(0.4)
            await scheduler.submit(SHORT_PROMPT)
        long_result = await long_task
    finally:
        await scheduler.stop()

    assert long_result.text == expected_long, (
        "a slot that stayed live across trims produced different output:\n"
        f"  got:  {long_result.text!r}\n"
        f"  want: {expected_long!r}"
    )
