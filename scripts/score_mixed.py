"""Score the cheap model on the mixed dev and held-out sets.

Produces the failure labels the router is measured against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import exact_match, is_failure, load_mixed_split, token_f1
from model import load_model
from scheduler import generate_batch

BATCH_SIZE = 8


def score(examples, model, tokenizer, label):
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
                    "prompt": example.as_prompt(),
                    "source": example.source,
                    "golds": list(example.answers),
                    "prediction": result.text,
                    "exact_match": exact_match(result.text, example.answers),
                    "f1": token_f1(result.text, example.answers),
                    "failed": is_failure(result.text, example.answers),
                }
            )
        print(f"{label}: {len(records)}/{len(examples)}", flush=True)
    return records


def summarise(records, label):
    n = len(records)
    fails = sum(r["failed"] for r in records)
    print(f"\n{label}  n={n}  failures={fails} ({fails / n:.0%})")
    for source in ("squad", "trivia"):
        rows = [r for r in records if r["source"] == source]
        if rows:
            f = sum(r["failed"] for r in rows)
            print(f"  {source:7s} {len(rows):3d} questions, {f:3d} failed ({f / len(rows):.0%})")


def main() -> None:
    dev, test = load_mixed_split(200, 200)
    model, tokenizer = load_model()

    dev_records = score(dev, model, tokenizer, "dev")
    test_records = score(test, model, tokenizer, "test")

    summarise(dev_records, "DEV")
    summarise(test_records, "HELD-OUT")

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "mixed_dev.json").write_text(json.dumps({"records": dev_records}, indent=2))
    (out / "mixed_holdout.json").write_text(
        json.dumps({"records": test_records}, indent=2)
    )
    print("\nwrote results/mixed_dev.json, results/mixed_holdout.json")


if __name__ == "__main__":
    main()
