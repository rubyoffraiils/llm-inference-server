"""Difficulty routing: pick a model tier per request.

A hand-written score, not a model of its own -- it has to be cheap
enough that it doesn't eat the latency the routing saves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

Tier = Literal["cheap", "expensive"]

# Questions arriving with a passage to read are answerable from that text;
# bare recall questions need the model to already know the answer, which
# is where a small model falls down.
HAS_CONTEXT_WEIGHT = -1.0

ROUTE_THRESHOLD = 0.5


@dataclass(frozen=True)
class RoutingDecision:
    tier: Tier
    score: float
    latency_ms: float


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
    )
