"""Record one lane of the demo race: N requests fired at once.

Saves when each request completed, relative to the moment they were all
sent, so the demo page can replay the race at real speed from real data
rather than from summary statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

QUESTIONS = [
    "What is the capital of Japan?",
    "Who wrote Hamlet?",
    "What is the largest ocean on Earth?",
    "How many days are in a leap year?",
    "What gas do plants absorb from the air?",
    "What language is spoken in Brazil?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water in Celsius?",
    "What planet is known as the Red Planet?",
    "What is the square root of 81?",
    "Which country gifted the Statue of Liberty to the United States?",
    "What is the chemical symbol for gold?",
    "Who developed the theory of general relativity?",
    "What is the longest river in Africa?",
    "How many continents are there?",
    "What is the smallest prime number?",
    "What currency is used in the United Kingdom?",
    "Which organ pumps blood through the body?",
    "What is the freezing point of water in Fahrenheit?",
    "Who was the first person to walk on the Moon?",
    "What is the hardest natural substance?",
    "In which city is the Eiffel Tower?",
    "What is the main language of Mexico?",
    "How many sides does a hexagon have?",
]


def prompt_for(index: int, nonce: str) -> str:
    # The nonce keeps prompts unique across lanes and runs: the server
    # caches responses and deduplicates identical in-flight prompts, either
    # of which would let a lane finish without generating.
    question = QUESTIONS[index % len(QUESTIONS)]
    return (
        f"Question: {question} (ref {nonce}-{index})\n\n"
        "Answer with as few words as possible, no explanation."
    )


async def record(url: str, mode: str, count: int, nonce: str) -> list[float]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        start = time.perf_counter()

        async def one(index: int) -> float:
            response = await client.post(
                url, json={"prompt": prompt_for(index, nonce), "mode": mode}
            )
            response.raise_for_status()
            return time.perf_counter() - start

        completions = await asyncio.gather(*(one(i) for i in range(count)))
    return sorted(round(t, 3) for t in completions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100/process")
    parser.add_argument("--mode", required=True, choices=["naive", "batched"])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--nonce", default=str(int(time.time())))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    completions = asyncio.run(record(args.url, args.mode, args.n, args.nonce))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mode": args.mode, "completions_s": completions}))
    print(f"{args.mode}: {len(completions)} requests, last finished at {completions[-1]:.2f}s")


if __name__ == "__main__":
    main()
