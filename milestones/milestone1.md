# Milestone 1: Batching & Continuous Batching

## Objective

Transform the naive scheduler from processing **one request at a time** to
**batching multiple requests** in a single GPU forward pass, then further
improve it with **iteration-level (continuous) batching** so new requests
can join the running batch without waiting for the entire batch to drain.

All memory allocation remains on-the-fly — no memory pool or paging needed.
KV caches are still contiguous per-request tensors managed by PyTorch's
default CUDA allocator.

## Background

The baseline scheduler is maximally naive: it picks one request, prefills
it, decodes it to completion, then moves to the next.  The GPU sits mostly
idle because each forward pass is a single-sequence matrix-vector product.

Modern GPUs are designed for large matrix-matrix multiplications.  Batching
multiple requests into a single forward pass converts those thin mat-vec
ops into fat mat-mat ops, dramatically improving utilization.

## Part A: Batched Forward Pass

### What to change

1. **Scheduler**: instead of running one request to completion, admit up
   to `max_running` requests from the waiting queue, prefill each one
   (still individually), then **decode all running requests together** in
   a single batched forward pass.

2. **Engine**: add a `batched_decode(requests)` method that:
   - Stacks the last generated token from each request → `(batch, 1)`
   - Pads per-request KV caches to the max cache length in the batch
   - Builds an attention mask to ignore padding positions
   - Runs one batched model forward
   - Extracts per-request KV (actual portion + new token) and samples

3. **Model**: thread an optional `attention_mask` parameter through
   `Attention → TransformerBlock → TransformerModel → CausalLM`.  When
   provided, pass it to `F.scaled_dot_product_attention(attn_mask=...)`.
   When `None`, fall back to the existing `is_causal` logic.

### Key details

- **Padding**: for a batch with cache lengths `[100, 200, 150]`, pad all
  KV caches to 200 on the sequence dimension.  After the model concatenates
  the new K/V token, total KV length is 201.  The attention mask shape is
  `(batch, 1, 1, 201)` — mask padding positions between each request's
  actual cache end and position 200.

- **RoPE correctness**: each request's `position_ids` is its own cache
  length (not the padded length).  The old cached K/V already have correct
  RoPE from when they were computed.  Padding positions are masked out so
  their RoPE values don't matter.

- **KV extraction after forward**: the returned KV has shape
  `(batch, kv_heads, max_cache_len+1, head_dim)` with padding in the
  middle.  For request `i` with actual cache length `L`, keep
  `[:, :, :L, :]` (old) and `[:, :, -1:, :]` (new token), concatenate.

- **Prefill stays unbatched**: different prompt lengths make batched
  prefill complex (variable-length padding + causal mask).  The decode
  phase is where the throughput gain lives — prefill is a one-time cost.

### Expected improvement

At concurrency 16, batched decode should be **5-7x** faster than baseline
because the linear layers (QKV projection, MLP) now process a `(16, ...)`
batch instead of 16 separate `(1, ...)` calls.

## Part B: Continuous Batching (Iteration-Level Batching)

### What to change

Modify the scheduler's `step()` so that:

1. **Phase 1**: prefill any newly admitted requests (one by one)
2. **Phase 2**: batched decode **all** running requests (including those
   just prefilled in Phase 1) in a single forward pass
3. Requests that finish are removed; their slots are immediately available
   for the next `step()`

This means the batch composition changes **every iteration** — a request
that just finished prefill joins the decode batch in the same step, and a
finished request's slot opens up for the next step.  No request has to
wait for the entire batch to drain before being admitted.

### Why this matters

With static batching, if you have a batch of 8 requests and one finishes
early, its slot sits empty until all 8 finish.  With iteration-level
batching, a new request from the waiting queue fills that slot on the very
next step.  This reduces:

- **Time-to-first-token (TTFT)**: new requests start faster
- **GPU idle time**: the batch stays full as long as there's work
- **Tail latency**: short requests aren't held hostage by long ones

## Validation

Use the serving benchmark to measure improvement:

```bash
# Start the server
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16

# Run benchmark
python -m benchmark.bench_serving --input-len 512 --output-len 256 --concurrencies 1,4,8,16
```

**Metrics to compare against baseline:**

| Metric | Where to look |
|--------|--------------|
| Generation throughput (tok/s) | Should scale with concurrency (5-7x at conc=16) |
| TTFT p50 | Should be similar or slightly better than baseline |
| Completion latency p50 | Should decrease significantly at higher concurrency |

## Files to modify

| File | Changes |
|------|---------|
| `miniengine/model.py` | Add `attention_mask` parameter to forward methods |
| `miniengine/engine.py` | Add `batched_decode()` method |
| `miniengine/scheduler.py` | Rewrite `step()` for batched + continuous batching |
