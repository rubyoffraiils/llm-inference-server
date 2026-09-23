"""Batching scheduler: run several prompts through one model together.

A fixed group of prompts is decoded in lockstep, one shared forward
pass per step instead of one pass per prompt. Requests still in the
batch when another slot finishes ride along until the whole batch is
done -- refilling a freed slot mid-batch is a separate, later piece.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from model import GenerationResult

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 50,
) -> list[GenerationResult]:
    """Generate continuations for all `prompts` together, as one batch.

    Left-padding keeps every sequence's last real token in the same
    column, so a single next-token prediction per row stays valid
    across the whole batch even though prompts differ in length.
    """
    start_time = time.perf_counter()
    n = len(prompts)

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    tokenizer.padding_side = "left"
    encoded = tokenizer(chat_prompts, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids.to(_device)
    attention_mask = encoded.attention_mask.to(_device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids
    past_key_values = None
    next_input = input_ids
    next_mask = attention_mask

    finished = [False] * n
    tokens_generated = [0] * n
    time_to_first_token: list[float | None] = [None] * n
    pad_id = tokenizer.pad_token_id

    for _ in range(max_new_tokens):
        # position_ids must come from the mask, not the step index --
        # left padding means real tokens don't start at column 0. Only the
        # last column is needed: it lines up with next_input's one token
        # (or all columns, on the first step, when next_input is the
        # full prompt).
        position_ids = next_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(next_mask == 0, 1)
        position_ids = position_ids[:, -next_input.shape[1] :]

        outputs = model(
            input_ids=next_input,
            attention_mask=next_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Slots already finished contribute a masked-out pad token instead
        # of a real prediction, so they don't drift once we stop reading them.
        for i in range(n):
            if finished[i]:
                next_token[i, 0] = pad_id

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for i in range(n):
            if finished[i]:
                continue
            tokens_generated[i] += 1
            if time_to_first_token[i] is None:
                time_to_first_token[i] = time.perf_counter() - start_time
            if next_token[i, 0].item() == tokenizer.eos_token_id:
                finished[i] = True

        if all(finished):
            break

        next_input = next_token
        # Every real column so far, plus one more real column for this
        # step's new token -- finished slots stay masked out from here on.
        step_mask = torch.tensor(
            [[0 if finished[i] else 1] for i in range(n)], device=_device
        )
        next_mask = torch.cat([next_mask, step_mask], dim=-1)

    total_time = time.perf_counter() - start_time

    results = []
    for i in range(n):
        reply_ids = generated_ids[i, prompt_len : prompt_len + tokens_generated[i]]
        text = tokenizer.decode(reply_ids, skip_special_tokens=True)
        results.append(
            GenerationResult(
                text=text,
                time_to_first_token=time_to_first_token[i],
                total_time=total_time,
                tokens_generated=tokens_generated[i],
            )
        )
    return results


def _prefill_and_pad(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    target_len: int,
):
    """Run `prompt` alone through the model, then left-pad its cache to `target_len`.

    Returns (cache, attention_mask, first_token, prompt_len) for this
    one request, shaped so its batch dimension can be spliced into an
    existing batch whose cache is already `target_len` columns long.
    The returned mask already covers `first_token` -- it's the input
    for the caller's very next step, so its position must count.
    """
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(_device)
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    cache = outputs.past_key_values
    first_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

    pad_amount = target_len - prompt_len
    if pad_amount > 0:
        for layer in cache.layers:
            # keys/values are [batch, heads, seq, head_dim] -- pad the seq
            # dim (index -2) on the left with zeros; values are never read
            # for these columns once the mask marks them as padding.
            pad_shape = list(layer.keys.shape)
            pad_shape[-2] = pad_amount
            zeros = torch.zeros(pad_shape, dtype=layer.keys.dtype, device=layer.keys.device)
            layer.keys = torch.cat([zeros, layer.keys], dim=-2)
            layer.values = torch.cat([zeros, layer.values], dim=-2)
        attention_mask = torch.cat(
            [torch.zeros((1, pad_amount), dtype=attention_mask.dtype, device=_device), attention_mask],
            dim=-1,
        )
    # pad_amount < 0 means this prompt is longer than the batch's cache;
    # the caller grows the batch with _pad_cache_left instead.

    # first_token is already "generated" -- extend the mask by one real
    # column for it now, so callers can feed it straight into their loop.
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=_device)],
        dim=-1,
    )

    return cache, attention_mask, first_token, prompt_len


def _pad_cache_left(cache, pad_amount: int) -> None:
    """Left-pad every slot in `cache` by `pad_amount` columns, in place.

    Used when a joining request's prompt is longer than the running
    batch's cache: the batch grows to meet it instead of the reverse.
    """
    if pad_amount <= 0:
        return
    for layer in cache.layers:
        pad_shape = list(layer.keys.shape)
        pad_shape[-2] = pad_amount
        zeros = torch.zeros(pad_shape, dtype=layer.keys.dtype, device=layer.keys.device)
        layer.keys = torch.cat([zeros, layer.keys], dim=-2)
        layer.values = torch.cat([zeros, layer.values], dim=-2)


def _splice_slot(batch_cache, slot_index: int, new_cache) -> None:
    """Overwrite `slot_index`'s row in `batch_cache` with `new_cache`'s data, in place.

    Both caches must already have the same sequence length (pad the
    new request to the batch's length with _prefill_and_pad first).
    Only `slot_index`'s row changes -- every other slot's keys/values
    are untouched.
    """
    for batch_layer, new_layer in zip(batch_cache.layers, new_cache.layers):
        batch_layer.keys[slot_index : slot_index + 1] = new_layer.keys
        batch_layer.values[slot_index : slot_index + 1] = new_layer.values


@dataclass
class _Slot:
    """One decode slot's bookkeeping. `future` is None when the slot is free."""

    future: asyncio.Future | None = None
    token_ids: list[int] = field(default_factory=list)
    start_time: float = 0.0
    time_to_first_token: float | None = None

    @property
    def is_free(self) -> bool:
        return self.future is None


class BatchingScheduler:
    """Runs a continuous decode loop, swapping requests in and out of slots.

    A slot that finishes is refilled from the queue on the very next
    step, while the other slots keep generating -- a request never
    waits for unrelated requests to drain first. Each caller awaits its
    own future, which the loop resolves as soon as that slot hits EOS
    or the token cap.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_batch_size: int = 4,
        max_new_tokens: int = 50,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._max_batch_size = max_batch_size
        self._max_new_tokens = max_new_tokens
        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self._loop_task: asyncio.Task | None = None

        self._slots = [_Slot() for _ in range(max_batch_size)]
        self._cache = None
        self._mask: torch.Tensor | None = None
        self._next_input: torch.Tensor | None = None

    def start(self) -> None:
        self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()

    async def submit(self, prompt: str) -> GenerationResult:
        """Queue `prompt` and wait for its own slot to finish generating."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((prompt, future))
        return await future

    async def _run(self) -> None:
        while True:
            await self._admit_waiting_requests()

            if all(slot.is_free for slot in self._slots):
                # Nothing in flight: block rather than spin, then loop
                # back around to admit whatever just arrived.
                prompt, future = await self._queue.get()
                self._queue.put_nowait((prompt, future))
                continue

            await asyncio.to_thread(self._decode_step)
            self._reap_finished_slots()

    async def _admit_waiting_requests(self) -> None:
        """Fill every free slot with a queued request, if any are waiting."""
        for index, slot in enumerate(self._slots):
            if not slot.is_free or self._queue.empty():
                continue
            prompt, future = self._queue.get_nowait()
            await asyncio.to_thread(self._join_slot, index, prompt, future)

    def _join_slot(self, index: int, prompt: str, future: asyncio.Future) -> None:
        """Prefill `prompt` alone and splice it into slot `index`."""
        target_len = self._cache.get_seq_length() if self._cache is not None else 0
        cache, mask_row, first_token, _ = _prefill_and_pad(
            prompt, self._model, self._tokenizer, target_len
        )

        if self._cache is None:
            # First request in an empty batch: this slot's cache becomes the
            # batch's cache, widened to hold every slot.
            for layer in cache.layers:
                layer.keys = layer.keys.repeat(self._max_batch_size, 1, 1, 1)
                layer.values = layer.values.repeat(self._max_batch_size, 1, 1, 1)
            self._cache = cache
            self._mask = mask_row.repeat(self._max_batch_size, 1)
            self._next_input = first_token.repeat(self._max_batch_size, 1)
        else:
            # A longer prompt grows the whole batch to meet it, rather
            # than being truncated to fit.
            overflow = mask_row.shape[1] - self._mask.shape[1]
            if overflow > 0:
                _pad_cache_left(self._cache, overflow)
                self._mask = torch.cat(
                    [
                        torch.zeros(
                            (self._max_batch_size, overflow),
                            dtype=self._mask.dtype,
                            device=_device,
                        ),
                        self._mask,
                    ],
                    dim=-1,
                )
            elif overflow < 0:
                _pad_cache_left(cache, -overflow)
                mask_row = torch.cat(
                    [
                        torch.zeros((1, -overflow), dtype=mask_row.dtype, device=_device),
                        mask_row,
                    ],
                    dim=-1,
                )
            _splice_slot(self._cache, index, cache)

        self._mask[index : index + 1] = mask_row
        self._next_input[index : index + 1] = first_token

        slot = self._slots[index]
        slot.future = future
        slot.token_ids = [first_token.item()]
        slot.start_time = time.perf_counter()
        slot.time_to_first_token = 0.0

    def _decode_step(self) -> None:
        """Advance every occupied slot by one token, in one shared forward pass."""
        position_ids = self._mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(self._mask == 0, 1)
        position_ids = position_ids[:, -self._next_input.shape[1] :]

        with torch.no_grad():
            outputs = self._model(
                input_ids=self._next_input,
                attention_mask=self._mask,
                position_ids=position_ids,
                past_key_values=self._cache,
                use_cache=True,
            )
        self._cache = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        pad_id = self._tokenizer.pad_token_id
        for index, slot in enumerate(self._slots):
            if slot.is_free:
                next_token[index, 0] = pad_id
            else:
                slot.token_ids.append(next_token[index, 0].item())

        self._next_input = next_token
        step_mask = torch.tensor(
            [[0 if slot.is_free else 1] for slot in self._slots], device=_device
        )
        self._mask = torch.cat([self._mask, step_mask], dim=-1)

    def _reap_finished_slots(self) -> None:
        """Resolve and free any slot that just hit EOS or the token cap."""
        for slot in self._slots:
            if slot.is_free:
                continue
            hit_eos = slot.token_ids[-1] == self._tokenizer.eos_token_id
            if not hit_eos and len(slot.token_ids) < self._max_new_tokens:
                continue

            text = self._tokenizer.decode(slot.token_ids, skip_special_tokens=True)
            slot.future.set_result(
                GenerationResult(
                    text=text,
                    time_to_first_token=slot.time_to_first_token,
                    total_time=time.perf_counter() - slot.start_time,
                    tokens_generated=len(slot.token_ids),
                )
            )
            slot.future = None
            slot.token_ids = []
