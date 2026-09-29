"""Difficulty routing: pick a model tier per request.

Two implementations behind one interface. The hand-written one is the
default and costs nothing to run; the Jev one asks a hosted decision
model and is there to be benchmarked against it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

Tier = Literal["cheap", "expensive"]

# Questions arriving with a passage to read are answerable from that text;
# bare recall questions need the model to already know the answer, which
# is where a small model falls down.
HAS_CONTEXT_WEIGHT = -1.0

ROUTE_THRESHOLD = 0.5

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
JEV_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RoutingDecision:
    tier: Tier
    score: float
    latency_ms: float
    # Jev reports how sure it is; the heuristic has no such notion, so it
    # reports None rather than a fabricated 1.0.
    confidence: float | None = None
    cost_usd: float = 0.0
    source: str = "heuristic"


def score_difficulty(prompt: str) -> float:
    """Higher means more likely to need the expensive tier.

    Deliberately one signal. Proper-noun counts and year mentions were
    measured too and made routing worse -- they sent questions the cheap
    tier handles fine to the expensive one, at a 30% hit rate against a
    54% base rate.
    """
    score = 1.0
    if "Context:" in prompt:
        score += HAS_CONTEXT_WEIGHT
    return score


def route(prompt: str) -> RoutingDecision:
    """Pick a tier, timing the decision so its cost can be reported."""
    start = time.perf_counter()
    score = score_difficulty(prompt)
    tier: Tier = "expensive" if score >= ROUTE_THRESHOLD else "cheap"
    return RoutingDecision(
        tier=tier,
        score=score,
        latency_ms=(time.perf_counter() - start) * 1000,
        source="heuristic",
    )


JEV_QUESTION = {
    "tier": {
        "type": "choice",
        "instructions": (
            "A small 0.5B language model and a larger 1.5B one can answer "
            "this question. Which is needed?"
        ),
        "criteria": {
            "cheap": (
                "The small model can answer it: the question supplies a "
                "passage containing the answer, or asks something simple "
                "and widely known."
            ),
            "expensive": (
                "The small model will likely fail: the question needs "
                "specific recalled knowledge with no passage supplied, or "
                "asks for interpretation rather than a fact."
            ),
        },
    }
}


def route_with_jev(prompt: str, client=None) -> RoutingDecision:
    """Ask Jev which tier to use, falling back to the heuristic.

    Never a hard dependency: a missing key, a timeout or an API change
    degrades to the hand-written router rather than failing the request.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return route(prompt)

    import httpx

    start = time.perf_counter()
    try:
        owns_client = client is None
        client = client or httpx.Client(timeout=JEV_TIMEOUT_S)
        try:
            response = client.post(
                JEV_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": JEV_MODEL,
                    "state": {"request": prompt},
                    "questions": JEV_QUESTION,
                },
            )
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                client.close()

        answer = payload["answers"]["tier"]
        tier: Tier = answer["choice"]
        return RoutingDecision(
            tier=tier,
            score=answer["probabilities"].get("expensive", 0.0),
            latency_ms=(time.perf_counter() - start) * 1000,
            confidence=answer.get("confidence"),
            cost_usd=payload.get("usage", {}).get("cost", 0.0),
            source="jev",
        )
    except Exception:
        return route(prompt)
