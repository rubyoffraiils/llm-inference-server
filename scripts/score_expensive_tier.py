"""Score the expensive model on the same questions as the cheap one.

Same prompts, same scoring, same seed -- only the model changes, so the
comparison is clean. Joined per question, because an average gap hides
the case that matters: questions the bigger model gets *wrong* that the
small one got right, which routing would actively make worse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import exact_match, is_failure, load_mixed_split, token_f1
from model import EXPENSIVE_MODEL, load_model
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


def compare(cheap_path: Path, expensive_records, label):
    cheap = {r["question"]: r for r in json.load(cheap_path.open())["records"]}
    both = [(cheap[r["question"]], r) for r in expensive_records if r["question"] in cheap]

    cheap_fail = sum(c["failed"] for c, _ in both)
    exp_fail = sum(e["failed"] for _, e in both)
    fixed = sum(1 for c, e in both if c["failed"] and not e["failed"])
    broken = sum(1 for c, e in both if not c["failed"] and e["failed"])

    n = len(both)
    print(f"\n{label}  n={n}")
    print(f"  cheap accuracy      {1 - cheap_fail / n:.0%}")
    print(f"  expensive accuracy  {1 - exp_fail / n:.0%}")
    print(f"  fixed by expensive  {fixed}  (cheap wrong -> expensive right)")
    print(f"  broken by expensive {broken}  (cheap right -> expensive wrong)")
    for source in ("squad", "trivia"):
        rows = [(c, e) for c, e in both if e["source"] == source]
        if rows:
            cf = sum(c["failed"] for c, _ in rows) / len(rows)
            ef = sum(e["failed"] for _, e in rows) / len(rows)
            print(f"  {source:7s} cheap {1 - cf:.0%} -> expensive {1 - ef:.0%}")


def main() -> None:
    dev, test = load_mixed_split(200, 200)
    model, tokenizer = load_model(EXPENSIVE_MODEL)

    dev_records = score(dev, model, tokenizer, "dev")
    test_records = score(test, model, tokenizer, "test")

    compare(Path("results/mixed_dev.json"), dev_records, "DEV")
    compare(Path("results/mixed_holdout.json"), test_records, "HELD-OUT")

    out = Path("results")
    (out / "expensive_dev.json").write_text(
        json.dumps({"records": dev_records}, indent=2)
    )
    (out / "expensive_holdout.json").write_text(
        json.dumps({"records": test_records}, indent=2)
    )
    print("\nwrote results/expensive_dev.json, results/expensive_holdout.json")


if __name__ == "__main__":
    main()
