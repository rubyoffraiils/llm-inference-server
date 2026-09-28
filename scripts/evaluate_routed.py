"""Measure the routed pipeline against always-cheap and always-expensive.

Every answer here is a real generation from whichever tier the router
picked -- no assuming the expensive tier is right about what it was
handed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from router import route

# Relative compute cost per token. Parameter-count ratio stands in for
# measured decode time until the benchmark runs on a GPU.
CHEAP_COST = 1.0
EXPENSIVE_COST = 3.0


def evaluate(cheap_path: Path, expensive_path: Path, label: str) -> None:
    cheap = json.load(cheap_path.open())["records"]
    expensive = json.load(expensive_path.open())["records"]
    # Both files come from the same ordered split, so pair by position:
    # TriviaQA repeats a few questions verbatim, and keying on text
    # silently drops them.
    assert len(cheap) == len(expensive), "record counts differ"
    pairs = list(zip(cheap, expensive))
    n = len(pairs)

    cheap_right = sum(not c["failed"] for c, _ in pairs)
    exp_right = sum(not e["failed"] for _, e in pairs)

    routed_right = 0
    to_expensive = 0
    for c, e in pairs:
        assert c["question"] == e["question"], "records misaligned"
        tier = route(c["prompt"]).tier
        record = e if tier == "expensive" else c
        to_expensive += tier == "expensive"
        routed_right += not record["failed"]

    # Cost is dominated by which tier ran, not by output length here,
    # so count requests rather than tokens for the proxy.
    cost_cheap = n * CHEAP_COST
    cost_expensive = n * EXPENSIVE_COST
    cost_routed = (n - to_expensive) * CHEAP_COST + to_expensive * EXPENSIVE_COST

    print(f"{label}  n={n}")
    print(f"{'mode':<18}{'accuracy':>10}{'rel. cost':>12}")
    print(f"{'always cheap':<18}{cheap_right / n:>9.0%}{cost_cheap / cost_expensive:>12.2f}")
    print(f"{'routed':<18}{routed_right / n:>9.0%}{cost_routed / cost_expensive:>12.2f}")
    print(f"{'always expensive':<18}{exp_right / n:>9.0%}{1.00:>12.2f}")

    quality_gap = (routed_right - exp_right) / n
    saving = 1 - cost_routed / cost_expensive
    print(
        f"\n  routed keeps {routed_right / n:.0%} accuracy "
        f"({quality_gap:+.0%} vs always-expensive) at {saving:.0%} lower cost"
    )
    print(f"  that's {routed_right / exp_right:.0%} of always-expensive quality")
    print(f"  {to_expensive}/{n} requests ({to_expensive / n:.0%}) went to the expensive tier\n")


def main() -> None:
    evaluate(
        Path("results/mixed_dev.json"),
        Path("results/expensive_dev.json"),
        "DEV",
    )
    evaluate(
        Path("results/mixed_holdout.json"),
        Path("results/expensive_holdout.json"),
        "HELD-OUT",
    )


if __name__ == "__main__":
    main()
