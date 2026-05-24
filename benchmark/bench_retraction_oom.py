"""
Decode-time OOM workload for Milestone 3 Step 8.

Run this against a MiniEngine server on a GPU VM.  It first warms the radix
cache with a shared prompt, then launches concurrent long generations with the
same prefix.  Use a deliberately small KV pool on the server to force decode
page growth to exhaust the pool.

Example:
    python -m benchmark.bench_retraction_oom \\
        --base-url http://localhost:8000 \\
        --num-requests 24 --concurrency 24 --max-tokens 4096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class Result:
    request_id: int
    ok: bool
    output_tokens: int
    elapsed_s: float
    error: str | None = None


def _shared_prompt(repeat: int, suffix: str = "") -> str:
    base = (
        "You are stress-testing a paged KV cache. "
        "Continue with concise numbered facts about memory scheduling. "
    )
    filler = " ".join(f"prefix{i % 97}" for i in range(repeat))
    return f"{base}{filler}\n{suffix}"


async def _model_id(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(f"{base_url}/v1/models") as resp:
        resp.raise_for_status()
        payload = await resp.json()
        return payload["data"][0]["id"]


async def _cache_stats(session: aiohttp.ClientSession, base_url: str) -> dict:
    async with session.get(f"{base_url}/cache_stats") as resp:
        resp.raise_for_status()
        return await resp.json()


async def _stream_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    request_id: int,
    prompt: str,
    max_tokens: int,
) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    output_tokens = 0
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                return Result(
                    request_id=request_id,
                    ok=False,
                    output_tokens=0,
                    elapsed_s=time.perf_counter() - start,
                    error=f"HTTP {resp.status}: {await resp.text()}",
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data = decoded[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = payload["choices"][0].get("delta", {})
                if delta.get("content"):
                    output_tokens += 1
    except Exception as exc:
        return Result(
            request_id=request_id,
            ok=False,
            output_tokens=output_tokens,
            elapsed_s=time.perf_counter() - start,
            error=repr(exc),
        )

    return Result(
        request_id=request_id,
        ok=True,
        output_tokens=output_tokens,
        elapsed_s=time.perf_counter() - start,
    )


async def _run(args: argparse.Namespace) -> int:
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model or await _model_id(session, args.base_url)
        print(f"model={model}")

        warm_prompt = _shared_prompt(args.prompt_repeat, "Warm the prefix cache.")
        print("warming cache...")
        warm = await _stream_one(
            session,
            args.base_url,
            model,
            request_id=-1,
            prompt=warm_prompt,
            max_tokens=args.warmup_tokens,
        )
        print(f"warmup ok={warm.ok} out={warm.output_tokens} error={warm.error}")
        print(f"cache_after_warmup={await _cache_stats(session, args.base_url)}")

        semaphore = asyncio.Semaphore(args.concurrency)

        async def one(i: int) -> Result:
            async with semaphore:
                prompt = _shared_prompt(args.prompt_repeat, f"Request {i}.")
                result = await _stream_one(
                    session,
                    args.base_url,
                    model,
                    request_id=i,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                )
                status = "ok" if result.ok else "ERR"
                print(
                    f"[{i:03d}] {status} out={result.output_tokens} "
                    f"elapsed={result.elapsed_s:.1f}s error={result.error}",
                    flush=True,
                )
                return result

        started = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(args.num_requests)))
        elapsed = time.perf_counter() - started
        ok = [r for r in results if r.ok]
        errors = [r for r in results if not r.ok]
        total_out = sum(r.output_tokens for r in ok)
        print(f"cache_after_stress={await _cache_stats(session, args.base_url)}")
        print(
            "summary "
            f"ok={len(ok)}/{len(results)} errors={len(errors)} "
            f"output_tokens={total_out} elapsed={elapsed:.1f}s "
            f"tok_s={(total_out / elapsed) if elapsed else 0:.1f}"
        )
        return 0 if not errors or args.allow_errors else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode OOM/retraction workload")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--prompt-repeat", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit 0 even if requests fail; useful for capturing baseline OOM logs.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
