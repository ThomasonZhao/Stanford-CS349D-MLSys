# Milestone 4 HiCache Benchmark Run

Created: 2026-05-31T01:49:33
Model: `Qwen/Qwen3-8B`
Base URL: `http://127.0.0.1:8000`
Suite: `accuracy`

Server settings:

- `mem_fraction_static`: `0.75`
- `page_size`: `32`
- `prefill_chunk_size`: `512`
- `max_running`: `16`
- HiCache CPU tier: `12.0` GiB
- Smoke HiCache CPU tier: `1.0` GiB

Client workload:

- `num_sessions`: `64`
- `turns_per_session`: `8`
- `concurrency`: `16`
- `max_tokens`: `32`
- `multiturn_schedule`: `round_robin`
- MMLU accuracy samples: `200`
- MMLU accuracy concurrency: `8`
- MMLU accuracy max tokens: `32`

Each experiment directory contains:

- `server.log`: MiniEngine server stdout/stderr
- `<client>.log`: benchmark client stdout/stderr
- `cache_stats_before.json`: cache counters before client run
- `cache_stats_after_<client>.json`: cache counters after each client
- `metadata.json`: commands, return codes, and cache stats file paths

Report mapping:

- GPU-only cliff: `gpu_only_multiturn_cliff`
- HiCache recovery: `hicache_multiturn_recovery`
- Optional overlap: `hicache_overlap_multiturn`
- MMLU correctness: `gpu_only_mmlu_accuracy`, `hicache_mmlu_accuracy`, `hicache_overlap_mmlu_accuracy`

Experiments:

- `gpu_only_mmlu_accuracy`: Correctness baseline: MMLU accuracy with the Milestone 3 GPU-only radix cache.
- `hicache_mmlu_accuracy`: Correctness check: MMLU accuracy with HiCache enabled and blocking CPU-tier copies.
- `hicache_overlap_mmlu_accuracy`: Correctness check: MMLU accuracy with HiCache async overlap enabled.
