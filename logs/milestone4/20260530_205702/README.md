# Milestone 4 HiCache Benchmark Run

Created: 2026-05-30T20:57:02
Model: `Qwen/Qwen3-8B`
Base URL: `http://127.0.0.1:8000`
Suite: `required`

Server settings:

- `mem_fraction_static`: `0.75`
- `page_size`: `32`
- `prefill_chunk_size`: `512`
- `max_running`: `16`
- HiCache CPU tier: `8.0` GiB
- Smoke HiCache CPU tier: `1.0` GiB

Client workload:

- `num_sessions`: `32`
- `turns_per_session`: `8`
- `concurrency`: `16`
- `max_tokens`: `16`
- `multiturn_schedule`: `round_robin`

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

Experiments:

- `gpu_only_multiturn_cliff`: Milestone 4 Track 1 baseline: GPU-only radix cache on the multiturn workload. Tune sessions/turns until late-turn hit rate collapses.
- `hicache_multiturn_recovery`: Same multiturn workload with HiCache enabled. The CPU tier should preserve evicted prefixes and restore late-turn hits.
