import pytest

from dataset import (
    QAExample,
    contains_answer,
    exact_match,
    is_failure,
    normalize_answer,
    token_f1,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("The Denver Broncos", "denver broncos"),
        ("Golden Anniversary.", "golden anniversary"),
        ("  extra   space  ", "extra space"),
        ("24–10", "2410"),  # en-dash, the case a naive comparison misses
        ("24-10", "2410"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_endash_and_hyphen_score_as_a_match():
    assert exact_match("24-10", ("24–10",))


def test_exact_match_ignores_articles_and_case():
    assert exact_match("the Denver Broncos", ("Denver Broncos",))
    assert not exact_match("Carolina Panthers", ("Denver Broncos",))


def test_exact_match_accepts_any_gold():
    assert exact_match("Santa Clara", ("Santa Clara, California", "Santa Clara"))


def test_contains_answer_is_looser_than_exact_match():
    prediction = "The highest mountain in Africa is Mount Kilimanjaro."
    golds = ("Kilimanjaro",)
    assert contains_answer(prediction, golds)
    assert not exact_match(prediction, golds)


def test_token_f1_partial_overlap():
    # 1 of 3 predicted tokens correct, 1 of 3 gold tokens found.
    score = token_f1("san francisco california", ("santa clara california",))
    assert 0.3 < score < 0.4


def test_token_f1_penalises_padding():
    """A correct answer buried in words scores well below a terse one."""
    terse = token_f1("Kilimanjaro", ("Kilimanjaro",))
    padded = token_f1(
        "The highest mountain in Africa is Mount Kilimanjaro", ("Kilimanjaro",)
    )
    assert terse == 1.0
    assert padded < 0.3


def test_is_failure_forgives_rephrasing_but_catches_wrong_answers():
    """Cases taken from a real cheap-tier run, where exact match
    labelled correct answers as failures.
    """
    assert not is_failure("June to September", ("between June and September",))
    assert not is_failure("Dioxygen (O2)", ("dioxygen",))
    assert not is_failure("Kill him.", ("kill Luther",))
    assert is_failure("They feel he was unfairly biased.", ("his brutality",))


def test_prompt_includes_context_and_question():
    example = QAExample(
        question="Who won?", context="The Broncos won.", answers=("Broncos",)
    )
    prompt = example.as_prompt()
    assert "The Broncos won." in prompt
    assert "Who won?" in prompt
