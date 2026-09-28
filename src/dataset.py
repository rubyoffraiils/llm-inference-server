"""QA sample loading and answer scoring.

Normalization and the exact-match / token-F1 definitions follow the
official SQuAD evaluation script, so scores here are comparable to
published numbers rather than to a homemade metric.

Traffic mixes two task types on purpose. Reading comprehension with a
supplied passage and bare recall questions differ enough in difficulty
for routing to have something real to decide; on SQuAD alone every
cheap signal tested scored at chance.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from dataclasses import dataclass

from datasets import load_dataset

SQUAD_NAME = "rajpurkar/squad"
TRIVIA_NAME = "mandarjoshi/trivia_qa"
TRIVIA_CONFIG = "rc.nocontext"
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 0

# Below this F1, an answer is treated as a genuine miss rather than a
# rephrasing. Exact match is too strict to label failures with: it calls
# "June to September" wrong against "between June and September", and a
# router trained on those labels would learn phrasing, not difficulty.
# Calibrated against a cheap-tier run where correct-but-reworded answers
# scored 0.50-0.67 and genuinely wrong ones scored 0.00.
FAILURE_F1_THRESHOLD = 0.4


@dataclass(frozen=True)
class QAExample:
    question: str
    answers: tuple[str, ...]
    context: str = ""
    source: str = "squad"

    def as_prompt(self) -> str:
        """Format for an instruction model, asking for a short answer.

        Brevity matters for scoring, not for the task: a correct answer
        buried in a sentence scores near zero on exact match.
        """
        instruction = "Answer with as few words as possible, no explanation."
        if self.context:
            return (
                f"Context: {self.context}\n\n"
                f"Question: {self.question}\n\n{instruction}"
            )
        return f"Question: {self.question}\n\n{instruction}"


def load_squad_sample(
    n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[QAExample]:
    """Draw a fixed random sample of SQuAD reading-comprehension questions.

    Sampling across the split rather than taking a slice matters:
    SQuAD groups several questions per context, so consecutive
    examples come from the same paragraph and aren't representative.
    """
    split = load_dataset(SQUAD_NAME, split="validation")
    indices = random.Random(seed).sample(range(len(split)), n)
    return [
        QAExample(
            question=split[i]["question"],
            context=split[i]["context"],
            answers=tuple(split[i]["answers"]["text"]),
            source="squad",
        )
        for i in indices
    ]


def load_trivia_sample(
    n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[QAExample]:
    """Draw a fixed random sample of bare recall questions, no context.

    Aliases come from Wikipedia redirects and some are unrelated to the
    intended sense ("Exile" the band picks up "Exile (politics)"), so
    only the canonical answer is kept for scoring.
    """
    split = load_dataset(TRIVIA_NAME, TRIVIA_CONFIG, split="validation")
    indices = random.Random(seed).sample(range(len(split)), n)
    return [
        QAExample(
            question=split[i]["question"],
            answers=(split[i]["answer"]["value"],),
            source="trivia",
        )
        for i in indices
    ]


def load_mixed_split(
    n_dev: int = DEFAULT_SAMPLE_SIZE,
    n_test: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[QAExample], list[QAExample]]:
    """Build disjoint dev and held-out sets, half each task type.

    Both halves are shuffled together so the two sources interleave the
    way real traffic would, rather than arriving in blocks.
    """
    total = n_dev + n_test
    half = total // 2
    examples = load_squad_sample(half, seed) + load_trivia_sample(total - half, seed)
    random.Random(seed).shuffle(examples)
    return examples[:n_dev], examples[n_dev:]


def normalize_answer(text: str) -> str:
    """Lowercase, drop articles and punctuation, collapse whitespace.

    Without this, "24-10" and "24–10" score as a miss, as do trailing
    periods and a leading "the".
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Dashes aren't in string.punctuation once unicode variants appear.
    text = re.sub(r"[‐-―]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: tuple[str, ...]) -> bool:
    """True if the prediction equals any gold answer after normalization."""
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(g) for g in golds)


def contains_answer(prediction: str, golds: tuple[str, ...]) -> bool:
    """True if a gold answer appears anywhere in the prediction.

    Forgiving of a model that answers in a full sentence. Reported
    next to exact match, never instead of it -- a long enough answer
    hits this by accident.
    """
    pred = normalize_answer(prediction)
    return any(normalize_answer(g) in pred for g in golds)


def token_f1(prediction: str, golds: tuple[str, ...]) -> float:
    """Best token-overlap F1 across the gold answers.

    Harmonic mean rather than an average so a rambling answer that
    happens to contain the gold tokens still scores badly on precision.
    """
    return max(_f1(prediction, g) for g in golds)


def is_failure(prediction: str, golds: tuple[str, ...]) -> bool:
    """True if the model genuinely missed, rather than just rephrasing.

    This is the label the router is judged against, so it has to track
    "would a person call this wrong" more closely than exact match does.
    """
    return token_f1(prediction, golds) < FAILURE_F1_THRESHOLD


def _f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    shared = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
