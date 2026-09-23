import torch

from model import generate, load_model

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time,",
]


def test_manual_decode_matches_generate():
    model, tokenizer = load_model()

    for prompt in PROMPTS:
        manual_output = generate(prompt, model, tokenizer, max_new_tokens=20)

        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids
        reference_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            repetition_penalty=1.0,
        )
        reference_output = tokenizer.decode(
            reference_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        assert manual_output == reference_output, (
            f"Mismatch for prompt {prompt!r}:\n"
            f"  manual:    {manual_output!r}\n"
            f"  reference: {reference_output!r}"
        )
