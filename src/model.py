"""Model loading and a manual token-by-token decode loop with KV-cache reuse."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_NAME = "distilgpt2"
MAX_NEW_TOKENS = 50

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
) -> str:
    """Generate a continuation for `prompt` via manual token-by-token decoding.

    Each step feeds only the newest token back in, along with the
    cached past_key_values, instead of re-running the whole sequence.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(_device)

    generated_ids = input_ids
    past_key_values = None
    # First step feeds the full prompt; every step after feeds one token.
    next_input = input_ids

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

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Next iteration only needs this new token -- the cache covers the rest.
        next_input = next_token

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
