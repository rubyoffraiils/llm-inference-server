"""Sample server and GPU state during a benchmark run.

The A40 benchmark degraded ~5x across 40 minutes while the GPU sat idle
and cool afterwards. KV-cache growth was measured locally and ruled out
(bounded, 52-100 columns), so this records several candidate causes at
once rather than testing one guess at a time.

    python scripts/monitor_server.py http://localhost:8100 monitor.csv
"""

from __future__ import annotations

import csv
import sys
import time
import urllib.request
from pathlib import Path

INTERVAL_S = 10.0


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/monitor.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "elapsed_s",
        "tier",
        "kv_cache_columns",
        "active_slots",
        "queue_depth",
        "requests_total",
        "latency_p50_ms",
        "latency_p95_ms",
        "cache_entries",
        "gpu_allocated_mib",
        "gpu_reserved_mib",
        "host_rss_mib",
    ]

    started = time.perf_counter()
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while True:
            try:
                with urllib.request.urlopen(f"{base}/stats", timeout=5) as response:
                    import json

                    stats = json.loads(response.read())
            except Exception:
                time.sleep(INTERVAL_S)
                continue

            # Memory comes from the server's own /stats: sampling it here
            # would only measure this monitoring process.
            process = stats.get("process", {})
            elapsed = time.perf_counter() - started
            for tier, values in stats["tiers"].items():
                writer.writerow(
                    {
                        "elapsed_s": round(elapsed, 1),
                        "tier": tier,
                        "kv_cache_columns": values.get("kv_cache_columns", 0),
                        "active_slots": values["active_slots"],
                        "queue_depth": values["queue_depth"],
                        "requests_total": values["requests_total"],
                        "latency_p50_ms": round(values["latency_p50_ms"], 1),
                        "latency_p95_ms": round(values["latency_p95_ms"], 1),
                        "cache_entries": values["cache_entries"],
                        "gpu_allocated_mib": process.get("gpu_allocated_mib", 0),
                        "gpu_reserved_mib": process.get("gpu_reserved_mib", 0),
                        "host_rss_mib": process.get("host_rss_mib", 0),
                    }
                )
            handle.flush()
            time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
