"""Model loading and a manual token-by-token decode loop with KV-cache reuse."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 50

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GenerationResult:
    text: str
    time_to_first_token: float
    total_time: float
    tokens_generated: int


def load_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the tokenizer and model once at startup, kept resident in memory."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(_device)
    model.eval()  # disables dropout etc. -- we're doing inference, not training
    return model, tokenizer


@torch.no_grad()  # no backprop during inference -- saves memory and time
def generate(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> GenerationResult:
    """Generate a continuation for `prompt` via manual token-by-token decoding.

    Each step feeds only the newest token back in, along with the
    cached past_key_values, instead of re-running the whole sequence.
    """
    start_time = time.perf_counter()
    time_to_first_token = None

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to(_device)

    generated_ids = input_ids
    past_key_values = None
    # First step feeds the full prompt; every step after feeds one token.
    next_input = input_ids
    tokens_generated = 0

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # Logits for the last position predict the next token.
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        tokens_generated += 1

        if time_to_first_token is None:
            time_to_first_token = time.perf_counter() - start_time

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Next iteration only needs this new token -- the cache covers the rest.
        next_input = next_token

    total_time = time.perf_counter() - start_time

    # Only the newly generated tokens are the reply -- the rest is the
    # chat-template-wrapped prompt we fed in.
    reply_ids = generated_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(reply_ids, skip_special_tokens=True)

    return GenerationResult(
        text=text,
        time_to_first_token=time_to_first_token,
        total_time=total_time,
        tokens_generated=tokens_generated,
    )
