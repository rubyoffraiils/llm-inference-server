"""Measure the router against the cheap tier's actual failures.

Reports dev and held-out side by side: a heuristic fitted to noise
scores well on dev and collapses on held-out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from router import route


def evaluate(path: Path, label: str) -> None:
    records = json.load(path.open())["records"]
    total = len(records)
    total_failures = sum(r["failed"] for r in records)

    routed = [r for r in records if route(r["prompt"]).tier == "expensive"]
    caught = sum(r["failed"] for r in routed)

    volume = len(routed) / total
    recall = caught / total_failures if total_failures else 0.0
    precision = caught / len(routed) if routed else 0.0
    base_rate = total_failures / total

    print(f"{label}")
    print(f"  traffic to expensive  {len(routed):3d}/{total} = {volume:.0%}")
    print(f"  failures caught       {caught:3d}/{total_failures} = {recall:.0%}")
    print(f"  lift over chance      {recall / volume:.2f}x" if volume else "  n/a")
    print(f"  precision             {precision:.0%}  (base rate {base_rate:.0%})")

    # Quality if every routed question were answered correctly by the
    # expensive tier -- the ceiling this routing buys.
    remaining = total_failures - caught
    print(f"  accuracy if routed    {1 - remaining / total:.0%}  (cheap alone {1 - base_rate:.0%})")
    print()


def main() -> None:
    for name, label in (
        ("results/mixed_dev.json", "DEV (tuned here)"),
        ("results/mixed_holdout.json", "HELD-OUT (never looked at)"),
    ):
        path = Path(name)
        if path.exists():
            evaluate(path, label)
        else:
            print(f"missing {name} -- run scripts/score_mixed.py first\n")


if __name__ == "__main__":
    main()
