"""Print how server state drifted over a benchmark run.

Answers the question the A40 run raised: what changed between the start
and the end, given the GPU was idle and cool afterwards.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

WATCH = [
    "kv_cache_columns",
    "latency_p50_ms",
    "latency_p95_ms",
    "cache_entries",
    "gpu_allocated_mib",
    "gpu_reserved_mib",
    "host_rss_mib",
]


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/monitor.csv")
    if not path.exists():
        print(f"missing {path}")
        return

    rows_by_tier: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows_by_tier[row["tier"]].append(row)

    for tier, rows in rows_by_tier.items():
        if len(rows) < 2:
            continue
        busy = [r for r in rows if int(r["active_slots"]) > 0] or rows
        # Compare quarters rather than single samples: one reading at
        # either end is noise.
        quarter = max(1, len(busy) // 4)
        head, tail = busy[:quarter], busy[-quarter:]

        print(f"\n=== {tier} ({len(rows)} samples, {len(busy)} while busy) ===")
        print(f"{'metric':<20}{'first 25%':>12}{'last 25%':>12}{'change':>10}")
        for field in WATCH:
            try:
                start = sum(float(r[field]) for r in head) / len(head)
                end = sum(float(r[field]) for r in tail) / len(tail)
            except (KeyError, ValueError):
                continue
            if start == 0 and end == 0:
                continue
            change = f"{(end - start) / start * 100:+.0f}%" if start else "n/a"
            print(f"{field:<20}{start:>12.1f}{end:>12.1f}{change:>10}")


if __name__ == "__main__":
    main()
