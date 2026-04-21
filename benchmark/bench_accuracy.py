"""
Accuracy benchmark — MMLU and GSM8K.

Sends evaluation questions to a running MiniEngine server and scores
the responses.  Useful for verifying that optimizations (batching,
paging) don't degrade model quality.

Usage:
    # Start the server first, then:
    python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200
    python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100

Requires: datasets, aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp
from datasets import load_dataset


# ── MMLU ────────────────────────────────────────────────────────────────

MMLU_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(row: dict) -> str:
    """Format an MMLU question as a multiple-choice prompt."""
    q = row["question"]
    choices = row["choices"]
    lines = [q, ""]
    for i, c in enumerate(choices):
        lines.append(f"{MMLU_LABELS[i]}. {c}")
    lines.append("")
    lines.append("Answer with the letter only (A, B, C, or D).")
    return "\n".join(lines)


def parse_mmlu_answer(text: str) -> int | None:
    """Extract the answer index (0-3) from model output."""
    text = text.strip()
    # Try to find a standalone letter A-D
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return MMLU_LABELS.index(m.group(1))
    # Fallback: first character
    if text and text[0].upper() in MMLU_LABELS:
        return MMLU_LABELS.index(text[0].upper())
    return None


def load_mmlu(num_samples: int, subjects: list[str] | None = None):
    """Load MMLU samples. If subjects is None, sample across all."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    if subjects:
        ds = ds.filter(lambda x: x["subject"] in subjects)
    ds = ds.shuffle(seed=42)
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    samples = []
    for row in ds:
        samples.append(
            {
                "prompt": format_mmlu_prompt(row),
                "gold": row["answer"],  # int 0-3
                "subject": row["subject"],
            }
        )
    return samples


# ── GSM8K ───────────────────────────────────────────────────────────────


def format_gsm8k_prompt(row: dict) -> str:
    """Format a GSM8K question."""
    return (
        f"{row['question']}\n\n"
        "Solve step by step. Put your final numeric answer after ####."
    )


def parse_gsm8k_gold(answer: str) -> str:
    """Extract the gold numeric answer from the GSM8K answer field."""
    # GSM8K answers end with #### <number>
    m = re.search(r"####\s*(.+)", answer)
    if m:
        return m.group(1).strip().replace(",", "")
    return ""


def parse_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric answer from model output."""
    # Look for #### pattern first
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def load_gsm8k(num_samples: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42)
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    samples = []
    for row in ds:
        samples.append(
            {
                "prompt": format_gsm8k_prompt(row),
                "gold": parse_gsm8k_gold(row["answer"]),
            }
        )
    return samples


# ── HTTP client ─────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    prompt: str
    gold: str | int
    predicted: str | int | None
    correct: bool
    model_output: str
    latency: float


async def send_eval_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, float]:
    """Send a non-streaming request and return (text, latency)."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    t0 = time.perf_counter()
    async with session.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
    ) as resp:
        data = await resp.json()
    latency = time.perf_counter() - t0

    text = data["choices"][0]["message"]["content"]
    return text, latency


async def run_eval(
    base_url: str,
    samples: list[dict],
    dataset: str,
    max_tokens: int,
    concurrency: int,
) -> list[EvalResult]:
    """Run all eval samples with bounded concurrency."""
    results: list[EvalResult] = []
    queue: asyncio.Queue = asyncio.Queue()
    for s in samples:
        await queue.put(s)

    completed = 0
    total = len(samples)

    async def worker(session: aiohttp.ClientSession):
        nonlocal completed
        while True:
            try:
                sample = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                text, latency = await send_eval_request(
                    session,
                    base_url,
                    sample["prompt"],
                    max_tokens,
                )
            except Exception as e:
                text = f"ERROR: {e}"
                latency = 0.0

            # Score
            if dataset == "mmlu":
                pred = parse_mmlu_answer(text)
                correct = pred == sample["gold"]
            else:  # gsm8k
                pred = parse_gsm8k_answer(text)
                correct = pred is not None and pred == sample["gold"]

            results.append(
                EvalResult(
                    prompt=sample["prompt"][:80],
                    gold=sample["gold"],
                    predicted=pred,
                    correct=correct,
                    model_output=text[:200],
                    latency=latency,
                )
            )

            completed += 1
            if completed % 10 == 0 or completed == total:
                acc_so_far = sum(r.correct for r in results) / len(results)
                print(
                    f"  [{completed}/{total}] "
                    f"accuracy={acc_so_far:.1%}  "
                    f"latency={latency:.1f}s",
                    flush=True,
                )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
    ) as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    return results


# ── Reporting ───────────────────────────────────────────────────────────


def print_report(results: list[EvalResult], dataset: str):
    total = len(results)
    correct = sum(r.correct for r in results)
    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(r.latency for r in results) / total if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {dataset.upper()} Accuracy Report")
    print(f"  Samples    : {total}")
    print(f"  Correct    : {correct}")
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"{'=' * 60}")

    if dataset == "mmlu":
        # Per-subject breakdown if we have subject info
        from collections import defaultdict

        # Not available from results directly, just show overall
        pass

    # Show some wrong examples
    wrong = [r for r in results if not r.correct]
    if wrong:
        print(
            f"\n  Sample incorrect predictions ({min(5, len(wrong))} of {len(wrong)}):"
        )
        for r in wrong[:5]:
            print(
                f"    Gold={r.gold}  Pred={r.predicted}  Output: {r.model_output[:80]}..."
            )


# ── Main ────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Accuracy benchmark (MMLU / GSM8K)")
    p.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "gsm8k"],
    )
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (default: 32 for MMLU, 512 for GSM8K)",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--base-url", type=str, default="http://localhost:8000")
    p.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated MMLU subjects to filter (e.g. 'abstract_algebra,anatomy')",
    )
    args = p.parse_args()

    if args.max_tokens is None:
        args.max_tokens = 32 if args.dataset == "mmlu" else 512

    print(f"\n{'=' * 60}")
    print(f"  Accuracy Benchmark")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Samples     : {args.num_samples}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Server      : {args.base_url}")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    if args.dataset == "mmlu":
        subjects = args.subjects.split(",") if args.subjects else None
        samples = load_mmlu(args.num_samples, subjects=subjects)
    else:
        samples = load_gsm8k(args.num_samples)
    print(f"  Loaded {len(samples)} samples\n")

    # Run evaluation
    print("Running evaluation...")
    results = asyncio.run(
        run_eval(
            base_url=args.base_url,
            samples=samples,
            dataset=args.dataset,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )

    print_report(results, args.dataset)


if __name__ == "__main__":
    main()
