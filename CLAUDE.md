# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MiniEngine — a minimal LLM serving engine for Stanford CS349D (ML Systems). The **scheduler** (`miniengine/scheduler.py`) is the primary optimization target across milestones.

Upstream course repo: `stanford-mast/CS349D-miniengine`

## Commands

```bash
# Install (editable)
pip install -e .

# Install with benchmark deps
pip install -e ".[bench]"

# Launch server (downloads model on first run)
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16

# Serving benchmark (server must be running)
python -m benchmark.bench_serving --input-len 512 --output-len 256 --concurrencies 1,4,8,16

# Accuracy benchmark (server must be running)
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200
python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100
```

## Architecture

```
Client → FastAPI Server (server.py) → Scheduler (background thread) → Engine (HF model wrapper)
```

- **core.py** — Data structures: `Request`, `SamplingParams`, `TokenOutput`, `RequestStatus`
- **engine.py** — HuggingFace model wrapper: `prefill()`, `decode()`, `batched_decode()`
- **model.py** — Pure PyTorch Qwen3 transformer (no HF model classes)
- **sampler.py** — Token sampling: greedy, temperature, top-k, top-p, repetition penalty
- **scheduler.py** — Request scheduling loop with strategies: `step()` (naive), `step_batch()`, `step_iterative()` (continuous batching)
- **server.py** — OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming
- **__main__.py** — CLI entry point

Responses stream back to clients via per-request token queues. The scheduler runs in a background thread, pulling from a waiting queue and driving the engine.

## Milestones

Assignment milestones live in `milestones/`. Each milestone specifies which files to modify and what metrics to hit. Benchmarks in `benchmark/` are used for validation.
