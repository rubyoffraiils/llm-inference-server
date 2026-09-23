import torch

from model import generate, load_model
from scheduler import _prefill_and_pad, _splice_slot


def _step(model, next_input, mask, cache):
    """One shared decode step for an arbitrary-size batch, mirroring
    generate_batch's per-step math, for use directly in these tests."""
    position_ids = mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    position_ids = position_ids[:, -next_input.shape[1] :]
    with torch.no_grad():
        outputs = model(
            input_ids=next_input,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
    cache = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    new_mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)], dim=-1)
    return next_token, new_mask, cache


def test_prefill_and_pad_matches_single_request():
    """A request prefilled and left-padded to a longer target length
    must still generate identically to running it with no padding at all.
    """
    model, tokenizer = load_model()
    prompt = "2 + 2 ="
    target_len = 40  # simulates joining a batch whose cache is already this long

    expected = generate(prompt, model, tokenizer, max_new_tokens=10).text

    cache, mask, next_token, _ = _prefill_and_pad(prompt, model, tokenizer, target_len)

    generated_tokens = [next_token.item()]
    for _ in range(9):
        position_ids = mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        position_ids = position_ids[:, -1:]

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype)], dim=-1)

    actual = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    assert actual == expected, f"padded: {actual!r} vs single-request: {expected!r}"


def test_splice_does_not_disturb_other_slots():
    """Splicing a new request into slot 0 mid-batch must not change slot 1's
    continuation at all, and the new slot 0 must match standalone generation.
    """
    model, tokenizer = load_model()
    tokenizer.padding_side = "left"

    prompts = [
        "Write a long poem about the sea.",
        "Tell me a short story about a dragon.",
    ]
    new_prompt = "What is the capital of Japan?"

    expected_new = generate(new_prompt, model, tokenizer, max_new_tokens=10).text

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    encoded = tokenizer(chat_prompts, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask

    # Ground truth for slot 1, run straight through with no splice at all.
    # 3 steps before the splice + 10 after, matching the post-splice run below.
    next_input, mask, cache = input_ids, attention_mask, None
    slot1_ground_truth = []
    for _ in range(13):
        next_token, mask, cache = _step(model, next_input, mask, cache)
        slot1_ground_truth.append(next_token[1, 0].item())
        next_input = next_token

    # Replay the same 3 steps, then splice a new request into slot 0.
    next_input, mask, cache = input_ids, attention_mask, None
    for _ in range(3):
        next_token, mask, cache = _step(model, next_input, mask, cache)
        next_input = next_token

    target_len = cache.get_seq_length()
    new_cache, new_mask_row, new_first_token, _ = _prefill_and_pad(
        new_prompt, model, tokenizer, target_len
    )
    _splice_slot(cache, 0, new_cache)
    mask[0:1] = new_mask_row
    next_input[0:1] = new_first_token

    # Run 10 more steps -- enough for slot 0 to finish -- comparing slot 1
    # against the matching window of the no-splice ground truth throughout.
    slot0_tokens = [new_first_token.item()]
    slot1_after_splice = []
    slot0_finished = False
    for _ in range(10):
        next_token, mask, cache = _step(model, next_input, mask, cache)
        if not slot0_finished:
            slot0_tokens.append(next_token[0, 0].item())
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                slot0_finished = True
        slot1_after_splice.append(next_token[1, 0].item())
        next_input = next_token

    assert slot1_after_splice == slot1_ground_truth[3:13], (
        "splicing slot 0 changed slot 1's continuation -- the splice leaked "
        "across the batch dimension"
    )

    actual_new = tokenizer.decode(slot0_tokens, skip_special_tokens=True)
    assert actual_new == expected_new, f"spliced: {actual_new!r} vs standalone: {expected_new!r}"
