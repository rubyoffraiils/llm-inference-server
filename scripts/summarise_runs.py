"""Collapse repeated benchmark runs into one table with spread.

A single run can't distinguish a real effect from noise. Reporting the
median and the min-max range across repeats makes it clear which
differences are worth believing.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def main() -> None:
    by_mode: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(RESULTS.glob("load_*.json")):
        payload = json.loads(path.read_text())
        for level in payload["levels"]:
            by_mode[payload["mode"]][level["offered_qps"]].append(level)

    if not by_mode:
        print(f"no load_*.json in {RESULTS}")
        raise SystemExit(1)

    for mode in ("naive", "batched", "routed"):
        if mode not in by_mode:
            continue
        levels = by_mode[mode]
        runs = max(len(v) for v in levels.values())
        print(f"\n=== {mode} ({runs} run{'s' if runs > 1 else ''}) ===")
        print(
            f"{'QPS':>5}{'achieved':>10}{'p50 s':>18}{'p95 s':>18}{'rejected':>12}"
        )
        for qps in sorted(levels):
            samples = levels[qps]
            achieved = statistics.median(s["achieved_qps"] for s in samples)
            p50s = [s["latency_p50_s"] for s in samples]
            p95s = [s["latency_p95_s"] for s in samples]
            rejected = [s["rejected"] for s in samples]
            print(
                f"{qps:>5}{achieved:>10.1f}"
                f"{_spread(p50s):>18}{_spread(p95s):>18}"
                f"{_spread_int(rejected):>12}"
            )


def _spread(values: list[float]) -> str:
    median = statistics.median(values)
    if len(values) == 1:
        return f"{median:.2f}"
    return f"{median:.2f} ({min(values):.2f}-{max(values):.2f})"


def _spread_int(values: list[int]) -> str:
    median = int(statistics.median(values))
    if len(values) == 1:
        return str(median)
    return f"{median} ({min(values)}-{max(values)})"


if __name__ == "__main__":
    main()
