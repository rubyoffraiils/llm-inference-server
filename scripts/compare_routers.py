"""Compare the hand-written router against Jev on identical traffic.

Same held-out questions, same failure labels, so the two are judged on
accuracy, latency and cost side by side rather than in isolation.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

import httpx

from router import route, route_with_jev

HOLDOUT = ROOT / "results/mixed_holdout.json"
EXPENSIVE = ROOT / "results/expensive_holdout.json"
OUT = ROOT / "results/router_comparison.json"


def summarise(name: str, decisions: list, records: list, expensive: list) -> dict:
    total = len(records)
    failures = sum(r["failed"] for r in records)

    routed_expensive = [d.tier == "expensive" for d in decisions]
    caught = sum(
        r["failed"] and to_exp for r, to_exp in zip(records, routed_expensive)
    )
    volume = sum(routed_expensive) / total

    # Accuracy of the pipeline: whichever tier was picked actually answered.
    correct = sum(
        not (e if to_exp else c)["failed"]
        for c, e, to_exp in zip(records, expensive, routed_expensive)
    )

    latencies = sorted(d.latency_ms for d in decisions)
    cost = sum(d.cost_usd for d in decisions)
    confidences = [d.confidence for d in decisions if d.confidence is not None]

    def pct(f: float) -> float:
        return latencies[min(int(f * len(latencies)), len(latencies) - 1)]

    return {
        "router": name,
        "routed_to_expensive": volume,
        "failures_caught": caught,
        "failures_total": failures,
        "recall": caught / failures if failures else 0.0,
        "lift_over_chance": (caught / failures) / volume if failures and volume else 0.0,
        "pipeline_accuracy": correct / total,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "cost_usd_total": cost,
        "cost_usd_per_1k_requests": cost / total * 1000,
        "mean_confidence": statistics.mean(confidences) if confidences else None,
        "fell_back": sum(d.source == "heuristic" for d in decisions)
        if name == "jev"
        else 0,
    }


def main() -> None:
    records = json.loads(HOLDOUT.read_text())["records"]
    expensive = json.loads(EXPENSIVE.read_text())["records"]
    assert len(records) == len(expensive)

    print(f"routing {len(records)} held-out questions through both routers\n")

    heuristic_decisions = [route(r["prompt"]) for r in records]

    jev_decisions = []
    with httpx.Client(timeout=10.0) as client:
        for index, record in enumerate(records, 1):
            jev_decisions.append(route_with_jev(record["prompt"], client=client))
            if index % 25 == 0:
                print(f"  jev: {index}/{len(records)}", flush=True)

    summaries = [
        summarise("heuristic", heuristic_decisions, records, expensive),
        summarise("jev", jev_decisions, records, expensive),
    ]

    # Jev returns a probability per option, not just a pick. Its own
    # threshold sends almost nothing to the expensive tier, so sweep the
    # probability directly to separate "Jev can't tell these apart" from
    # "its default cut-off is too conservative for this task".
    print()
    print("jev, routing on P(expensive) instead of its chosen label:")
    print(f"{'threshold':>10}{'routed':>8}{'recall':>8}{'lift':>7}{'acc':>7}")
    failures = sum(r["failed"] for r in records)
    for threshold in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        to_expensive = [d.score >= threshold for d in jev_decisions]
        volume = sum(to_expensive) / len(records)
        caught = sum(r["failed"] and e for r, e in zip(records, to_expensive))
        correct = sum(
            not (e if to_exp else c)["failed"]
            for c, e, to_exp in zip(records, expensive, to_expensive)
        )
        recall = caught / failures if failures else 0.0
        lift = recall / volume if volume else 0.0
        print(
            f"{threshold:>10.2f}{volume:>8.0%}{recall:>8.0%}{lift:>7.2f}"
            f"{correct / len(records):>7.0%}"
        )

    print()
    header = f"{'':<12}{'routed':>8}{'recall':>8}{'lift':>7}{'acc':>7}{'p50 ms':>9}{'$/1k req':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['router']:<12}{s['routed_to_expensive']:>7.0%}{s['recall']:>8.0%}"
            f"{s['lift_over_chance']:>7.2f}{s['pipeline_accuracy']:>7.0%}"
            f"{s['latency_p50_ms']:>9.1f}{s['cost_usd_per_1k_requests']:>10.4f}"
        )

    jev = summaries[1]
    if jev["fell_back"]:
        print(f"\n{jev['fell_back']} jev calls failed and fell back to the heuristic")
    if jev["mean_confidence"] is not None:
        print(f"jev mean confidence: {jev['mean_confidence']:.2f}")

    OUT.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
