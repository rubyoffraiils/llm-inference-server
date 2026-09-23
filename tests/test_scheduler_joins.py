import asyncio

import pytest

from model import generate, load_model
from scheduler import BatchingScheduler

EARLY_PROMPT = "Tell me a short story about a dragon."
LATE_PROMPT = "What is the capital of Japan?"


@pytest.mark.asyncio
async def test_late_request_joins_running_batch():
    """A request submitted after the batch is already generating must get
    the same answer it would get on its own.
    """
    model, tokenizer = load_model()
    expected_late = generate(LATE_PROMPT, model, tokenizer, max_new_tokens=50).text

    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4)
    scheduler.start()
    try:
        early = asyncio.create_task(scheduler.submit(EARLY_PROMPT))
        # Long enough that the early request is several decode steps in,
        # so the late one genuinely joins mid-flight.
        await asyncio.sleep(2.0)
        late = asyncio.create_task(scheduler.submit(LATE_PROMPT))

        early_result, late_result = await asyncio.gather(early, late)
    finally:
        await scheduler.stop()

    assert late_result.text == expected_late, (
        f"joined mid-batch: {late_result.text!r} vs alone: {expected_late!r}"
    )
    assert early_result.tokens_generated > 0


@pytest.mark.asyncio
async def test_concurrent_requests_each_get_own_answer():
    """Requests submitted together must not cross-contaminate."""
    model, tokenizer = load_model()
    prompts = [
        "What is the capital of Japan?",
        "2 + 2 =",
        "The opposite of hot is",
    ]
    expected = [generate(p, model, tokenizer, max_new_tokens=50).text for p in prompts]

    scheduler = BatchingScheduler(model, tokenizer, max_batch_size=4)
    scheduler.start()
    try:
        results = await asyncio.gather(*(scheduler.submit(p) for p in prompts))
    finally:
        await scheduler.stop()

    for prompt, result, want in zip(prompts, results, expected):
        assert result.text == want, f"{prompt!r}: got {result.text!r}, want {want!r}"
