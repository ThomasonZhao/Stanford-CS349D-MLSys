# M3 Step 8 Retraction Validation

## Retraction Policy

Decode-time OOM is caught at the scheduler boundary after `pool.allocate`
has already asked the radix cache to evict unlocked pages.  Retraction is
therefore the fallback when cache eviction cannot satisfy decode page growth.

Victim selection:

1. Prefer the request whose decode page growth failed.
2. Otherwise choose the request holding the most KV pages.
3. Break ties by largest remaining output budget.
4. Break final ties by newest `arrival_time`.

The newest-arrival final tie-break is intentional: it preserves FCFS behavior
for older requests when all other retraction costs are equal.

When a victim is retracted, MiniEngine frees only pool-owned pages, unlocks
but does not free borrowed radix-cache pages, resets the request's `_PagedState`,
and pushes the request back to the front of `scheduler.waiting`.  If the request
has already streamed output tokens, its next prefill replays
`input_ids + output_ids` so the client-visible prefix is preserved while KV is
rebuilt.

## Expected Artifacts

This directory is reserved for the GPU-only validation artifacts:

- `server_oom_baseline.log`
- `bench_oom_baseline.log`
- `server_with_retraction.log`
- `bench_with_retraction.log`

The local macOS run cannot produce these files because the workload requires a
CUDA KV pool.  The current local status is code-complete plus CPU control-flow
coverage; end-to-end evidence still needs the L4 VM run.

## L4 Flow

Baseline capture, using the parent commit before retraction:

```bash
mkdir -p report_screenshots/M3_step8
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 \
  --mode paged --attention-backend flashinfer --page-size 32 \
  --kv-pool-gb 1.0 --max-running 24 --prefill-chunk-size 512 \
  > report_screenshots/M3_step8/server_oom_baseline.log 2>&1
```

In another shell:

```bash
python -m benchmark.bench_retraction_oom \
  --num-requests 24 --concurrency 24 --max-tokens 4096 --allow-errors \
  > report_screenshots/M3_step8/bench_oom_baseline.log 2>&1
```

Retraction capture, using this commit:

```bash
mkdir -p report_screenshots/M3_step8
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 \
  --mode paged --attention-backend flashinfer --page-size 32 \
  --kv-pool-gb 1.0 --max-running 24 --prefill-chunk-size 512 \
  > report_screenshots/M3_step8/server_with_retraction.log 2>&1
```

In another shell:

```bash
python -m benchmark.bench_retraction_oom \
  --num-requests 24 --concurrency 24 --max-tokens 4096 \
  > report_screenshots/M3_step8/bench_with_retraction.log 2>&1
```

Success conditions for the retraction run:

- `bench_with_retraction.log` reports `errors=0`.
- `server_with_retraction.log` contains at least one `[retract] victim=...`
  line.
- The final `cache_after_stress=...` line reports `num_cached_pages > 0`,
  showing locked or resident cache pages survived retraction.
