"""Sample server, GPU and host state throughout a benchmark run.

A load-idle-load experiment showed the earlier slowdown was machine state,
not server state: the same process recovered after idling, a fresh process
on a warm machine did not. So beyond the server's own counters, this
records GPU clocks, temperature, power, throttle reasons and host load
*during* the run -- a single nvidia-smi reading after the run ends shows
an idle, cooled GPU and says nothing about clocks under load.

    python scripts/monitor_server.py http://localhost:8100 results/monitor.csv
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

INTERVAL_S = 10.0

FIELDS = [
    "elapsed_s",
    "tier",
    "kv_cache_columns",
    "active_slots",
    "queue_depth",
    "requests_total",
    "latency_p50_ms",
    "latency_p95_ms",
    "forward_ms_p50",
    "gpu_allocated_mib",
    "gpu_reserved_mib",
    "host_peak_rss_mib",
    "gpu_sm_clock_mhz",
    "gpu_max_sm_clock_mhz",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_throttle_reasons",
    "host_load_1m",
]


def gpu_telemetry() -> dict:
    """GPU-wide clocks and thermals, so throttling under load is visible."""
    query = "clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().splitlines()[0]
        clock, max_clock, temp, power = (v.strip() for v in out.split(","))
    except Exception:
        return {}

    reasons = ""
    # Renamed across driver versions; try the current name, then the old one.
    for field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                reasons = result.stdout.strip().splitlines()[0]
                break
        except Exception:
            continue

    return {
        "gpu_sm_clock_mhz": clock,
        "gpu_max_sm_clock_mhz": max_clock,
        "gpu_temp_c": temp,
        "gpu_power_w": power,
        "gpu_throttle_reasons": reasons,
    }


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/monitor.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        while True:
            elapsed = round(time.perf_counter() - started, 1)

            # Machine row every interval, even while the server restarts,
            # so cooldowns between runs show up as falling temperature.
            writer.writerow(
                {
                    "elapsed_s": elapsed,
                    "tier": "machine",
                    "host_load_1m": round(os.getloadavg()[0], 2),
                    **gpu_telemetry(),
                }
            )

            try:
                with urllib.request.urlopen(f"{base}/stats", timeout=5) as response:
                    stats = json.loads(response.read())
            except Exception:
                stats = None

            if stats is not None:
                # Memory comes from the server's own /stats: sampling it here
                # would only measure this monitoring process.
                process = stats.get("process", {})
                for tier, values in stats["tiers"].items():
                    writer.writerow(
                        {
                            "elapsed_s": elapsed,
                            "tier": tier,
                            "kv_cache_columns": values.get("kv_cache_columns", 0),
                            "active_slots": values["active_slots"],
                            "queue_depth": values["queue_depth"],
                            "requests_total": values["requests_total"],
                            "latency_p50_ms": round(values["latency_p50_ms"], 1),
                            "latency_p95_ms": round(values["latency_p95_ms"], 1),
                            "forward_ms_p50": round(values.get("forward_ms_p50", 0), 2),
                            "gpu_allocated_mib": process.get("gpu_allocated_mib", 0),
                            "gpu_reserved_mib": process.get("gpu_reserved_mib", 0),
                            "host_peak_rss_mib": process.get("host_peak_rss_mib", 0),
                        }
                    )

            handle.flush()
            time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
