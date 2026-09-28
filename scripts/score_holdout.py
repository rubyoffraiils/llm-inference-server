"""Score the cheap model on the held-out half of the SQuAD split.

Kept separate from the dev set so router heuristics tuned on dev can be
reported on questions they were never fitted to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    contains_answer,
    exact_match,
    is_failure,
    load_squad_split,
    token_f1,
)
from model import load_model
from scheduler import generate_batch

BATCH_SIZE = 8
OUT = Path("results/cheap_tier_holdout.json")


def main() -> None:
    _, examples = load_squad_split(200, 200)
    model, tokenizer = load_model()

    records = []
    for start in range(0, len(examples), BATCH_SIZE):
        chunk = examples[start : start + BATCH_SIZE]
        results = generate_batch(
            [ex.as_prompt() for ex in chunk], model, tokenizer, max_new_tokens=32
        )
        for example, result in zip(chunk, results):
            records.append(
                {
                    "question": example.question,
                    "golds": list(example.answers),
                    "prediction": result.text,
                    "exact_match": exact_match(result.text, example.answers),
                    "contains": contains_answer(result.text, example.answers),
                    "f1": token_f1(result.text, example.answers),
                    "failed": is_failure(result.text, example.answers),
                    "tokens": result.tokens_generated,
                }
            )
        print(f"scored {len(records)}/{len(examples)}", flush=True)

    failures = sum(r["failed"] for r in records)
    print()
    print(f"n           {len(records)}")
    print(f"exact match {sum(r['exact_match'] for r in records) / len(records):.1%}")
    print(f"token F1    {sum(r['f1'] for r in records) / len(records):.3f}")
    print(f"failures    {failures} ({failures / len(records):.1%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
