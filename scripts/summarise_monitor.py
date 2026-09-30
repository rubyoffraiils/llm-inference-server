"""Print how server and machine state drifted over a benchmark run."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

TIER_FIELDS = [
    "kv_cache_columns",
    "forward_ms_p50",
    "latency_p50_ms",
    "gpu_allocated_mib",
    "gpu_reserved_mib",
    "host_peak_rss_mib",
]
MACHINE_FIELDS = ["gpu_sm_clock_mhz", "gpu_temp_c", "gpu_power_w", "host_load_1m"]

# nvidia-smi reports this bitmask when nothing is limiting clocks.
NOT_THROTTLED = {"", "0x0000000000000000", "Not Active", "[N/A]"}


def _mean(rows: list[dict], field: str) -> float | None:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def _drift_table(rows: list[dict], fields: list[str]) -> None:
    # Compare quarters rather than single samples: one reading at either
    # end is noise.
    quarter = max(1, len(rows) // 4)
    head, tail = rows[:quarter], rows[-quarter:]
    print(f"{'metric':<22}{'first 25%':>12}{'last 25%':>12}{'change':>10}")
    for field in fields:
        start, end = _mean(head, field), _mean(tail, field)
        if start is None or end is None or (start == 0 and end == 0):
            continue
        change = f"{(end - start) / start * 100:+.0f}%" if start else "n/a"
        print(f"{field:<22}{start:>12.1f}{end:>12.1f}{change:>10}")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/monitor.csv")
    if not path.exists():
        print(f"missing {path}")
        return

    rows_by_tier: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows_by_tier[row["tier"]].append(row)

    machine = rows_by_tier.pop("machine", [])
    if machine:
        print(f"=== machine ({len(machine)} samples) ===")
        _drift_table(machine, MACHINE_FIELDS)
        clocks = [float(r["gpu_sm_clock_mhz"]) for r in machine if r.get("gpu_sm_clock_mhz")]
        max_clock = _mean(machine, "gpu_max_sm_clock_mhz")
        if clocks and max_clock:
            print(
                f"SM clock min {min(clocks):.0f} / max {max(clocks):.0f} MHz "
                f"(rated {max_clock:.0f})"
            )
        throttled = [r for r in machine if r.get("gpu_throttle_reasons", "") not in NOT_THROTTLED]
        print(f"samples with an active throttle reason: {len(throttled)}/{len(machine)}")
        if throttled:
            reasons = sorted({r["gpu_throttle_reasons"] for r in throttled})
            print(f"reasons seen: {', '.join(reasons)}")

    for tier, rows in rows_by_tier.items():
        busy = [r for r in rows if r.get("active_slots") and int(r["active_slots"]) > 0] or rows
        if len(busy) < 2:
            continue
        print(f"\n=== {tier} ({len(rows)} samples, {len(busy)} while busy) ===")
        _drift_table(busy, TIER_FIELDS)


if __name__ == "__main__":
    main()
