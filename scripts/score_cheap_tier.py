"""Score the cheap model on the fixed SQuAD sample.

Establishes the quality baseline the router is measured against, and
records which questions the cheap tier gets wrong -- those are the
"needed the expensive model" labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import (
    contains_answer,
    exact_match,
    is_failure,
    load_squad_sample,
    token_f1,
)
from model import load_model
from scheduler import generate_batch

BATCH_SIZE = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=200, help="questions to score")
    parser.add_argument("--seed", type=int, default=0, help="sample seed")
    parser.add_argument("--out", type=Path, default=Path("results/cheap_tier.json"))
    args = parser.parse_args()

    examples = load_squad_sample(n=args.n, seed=args.seed)
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

    em = sum(r["exact_match"] for r in records) / len(records)
    contains = sum(r["contains"] for r in records) / len(records)
    f1 = sum(r["f1"] for r in records) / len(records)
    tokens = sum(r["tokens"] for r in records) / len(records)
    failures = sum(r["failed"] for r in records)

    print()
    print(f"n              {len(records)}")
    print(f"exact match    {em:.1%}")
    print(f"contains gold  {contains:.1%}")
    print(f"token F1       {f1:.3f}")
    print(f"mean tokens    {tokens:.1f}")
    print(f"failures       {failures} ({failures / len(records):.1%}) -- router targets these")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n": len(records),
                "exact_match": em,
                "contains": contains,
                "token_f1": f1,
                "mean_tokens": tokens,
                "failure_rate": failures / len(records),
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
