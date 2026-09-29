"""One Jev call, to check the key and the request shape work.

Run before the full comparison so a bad key or a changed API surfaces
on one request rather than two hundred.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Minimal .env reader -- avoids a dependency for two lines of parsing.
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

from dataset import QAExample
from router import route, route_with_jev

WITH_CONTEXT = QAExample(
    question="Who won Super Bowl 50?",
    context="The Denver Broncos defeated the Carolina Panthers 24-10.",
    answers=("Denver Broncos",),
).as_prompt()

BARE_RECALL = QAExample(
    question="Which Lloyd Webber musical premiered in the US in 1993?",
    answers=("Sunset Boulevard",),
).as_prompt()


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("no OPENROUTER_API_KEY -- put it in .env first")
        raise SystemExit(1)

    for label, prompt in (("with context", WITH_CONTEXT), ("bare recall", BARE_RECALL)):
        heuristic = route(prompt)
        jev = route_with_jev(prompt)
        print(f"\n{label}")
        print(f"  heuristic: {heuristic.tier:9s} {heuristic.latency_ms:8.3f}ms")
        confidence = "n/a" if jev.confidence is None else f"{jev.confidence:.2f}"
        print(
            f"  jev:       {jev.tier:9s} {jev.latency_ms:8.1f}ms  "
            f"confidence={confidence}  cost=${jev.cost_usd:.6f}  ({jev.source})"
        )
        if jev.source == "heuristic":
            print("  ^ jev call failed and fell back -- see the error below")

    # Surface the real error rather than the silent fallback.
    import httpx

    response = httpx.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": "typesafe/jev-1.13",
            "state": {"request": BARE_RECALL},
            "questions": {
                "tier": {
                    "type": "choice",
                    "instructions": "Which model tier is needed?",
                    "criteria": {"cheap": "small model suffices", "expensive": "needs the larger model"},
                }
            },
        },
        timeout=10.0,
    )
    print(f"\nraw call: HTTP {response.status_code}")
    print(response.text[:600])


if __name__ == "__main__":
    main()
