# M3 Step 8 Retraction Validation

This directory is reserved for the GPU-only validation artifacts:

- `server_oom_baseline.log`
- `bench_oom_baseline.log`
- `server_with_retraction.log`
- `bench_with_retraction.log`

Suggested L4 flow:

```bash
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 \
  --mode paged --attention-backend flashinfer --page-size 32 \
  --kv-pool-gb 1.0 --max-running 24 --prefill-chunk-size 512
```

In another shell:

```bash
python -m benchmark.bench_retraction_oom \
  --num-requests 24 --concurrency 24 --max-tokens 4096
```

For the baseline capture, run the workload with `--allow-errors`.  For the
retraction capture, the same workload should finish with no client errors and
the server log should contain at least one `[retract]` line.
