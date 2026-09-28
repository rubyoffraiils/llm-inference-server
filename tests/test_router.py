from dataset import QAExample
from router import ROUTE_THRESHOLD, route, score_difficulty

WITH_CONTEXT = QAExample(
    question="Who won Super Bowl 50?",
    context="The Denver Broncos defeated the Carolina Panthers 24-10.",
    answers=("Denver Broncos",),
    source="squad",
).as_prompt()

BARE_RECALL = QAExample(
    question="Which Lloyd Webber musical premiered in the US in 1993?",
    answers=("Sunset Boulevard",),
    source="trivia",
).as_prompt()


def test_supplied_context_routes_cheap():
    assert route(WITH_CONTEXT).tier == "cheap"


def test_bare_recall_routes_expensive():
    assert route(BARE_RECALL).tier == "expensive"


def test_context_lowers_the_score_below_threshold():
    assert score_difficulty(WITH_CONTEXT) < ROUTE_THRESHOLD
    assert score_difficulty(BARE_RECALL) >= ROUTE_THRESHOLD


def test_decision_records_its_own_latency():
    decision = route(BARE_RECALL)
    assert decision.latency_ms >= 0
    # The routing decision has to be negligible next to generation, or
    # it eats the saving it exists to create.
    assert decision.latency_ms < 5


def test_routing_ignores_proper_nouns_and_years():
    """Both were measured and made routing worse, so the score must not
    react to them -- reintroducing either should fail here.
    """
    plain = QAExample(question="What is the capital?", answers=("x",)).as_prompt()
    loaded = QAExample(
        question="What did Albert Einstein publish in 1905?", answers=("x",)
    ).as_prompt()
    assert score_difficulty(loaded) == score_difficulty(plain)
