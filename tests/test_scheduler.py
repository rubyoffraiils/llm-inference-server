from model import generate, load_model
from scheduler import generate_batch

PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time,",
]


def test_batched_matches_single_request():
    model, tokenizer = load_model()

    single_outputs = [
        generate(p, model, tokenizer, max_new_tokens=20).text for p in PROMPTS
    ]
    batched_results = generate_batch(PROMPTS, model, tokenizer, max_new_tokens=20)
    batched_outputs = [r.text for r in batched_results]

    for prompt, single, batched in zip(PROMPTS, single_outputs, batched_outputs):
        assert single == batched, (
            f"Mismatch for prompt {prompt!r}:\n"
            f"  single:  {single!r}\n"
            f"  batched: {batched!r}"
        )
