# CS349D Writeup - Milestone 1: Batching & Continuous Batching

Thomason Zhao (changyuz)
Joey Ji (junyiji3)

## Hardware Setup

All experiments were run on an AWS `g4dn.2xlarge` instance:

- **CPU**: 8-core Intel Xeon Platinum 8259CL @ 2.50GHz
- **Memory**: 32 GB DDR4
- **GPU**: NVIDIA T4 (16 GB GDDR6)
- **Model**: Qwen/Qwen3-4B-Instruct-2507, float16

We chose the `g4dn.2xlarge` for its balance of cost and capability. The T4's 16 GB VRAM is sufficient for the Qwen3-4B model in float16 (model is around 8 GB), but later we figure out it actually not quite sufficient for larger batches. The 8 CPU cores provide enough headroom for request handling and tokenization without bottlenecking the GPU.

**Note on dtype: float16 vs bfloat16.** The default engine configuration uses bfloat16, but NVIDIA GPUs only natively support bfloat16 starting from the Ampere architecture (A100 and later). On older GPUs such as the T4 (Turing) and V100 (Volta), bfloat16 operations are emulated by converting to float32 and back, which significantly degrades inference throughput. We confirmed this empirically -- running our continuous batching implementation in bfloat16 on the T4 yielded substantially worse results:

**bfloat16 on T4 (continuous batching, input 512, output 256, 100 requests):**

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 554          | 568          | 9,235          | 12,494         | 34.1          | 46.8          | 27       |
| 4    | 1,120        | 2,376        | 50,172         | 50,966         | 192.8         | 330.2         | 20       |
| 8    | 2,059        | 4,573        | 59,161         | 59,529         | 223.1         | 603.7         | 33       |
| 16   | 4,177        | 8,728        | 77,576         | 78,064         | 286.6         | 1,147.8       | 49       |

Compared to float16 (80 tok/s at conc=16), bfloat16 only reaches 49 tok/s -- a **39% throughput drop** -- with dramatically worse tail latencies (TPOT p99 of 1,148 ms vs 195 ms). All subsequent experiments therefore use **float16**.

## Overview

The baseline MiniEngine scheduler processes requests **one at a time**: it picks a single request, prefills it, decodes it to completion, and only then moves to the next. This means the GPU executes thin matrix-vector products for each decode step, vastly underutilizing the hardware's matrix-multiplication throughput.

We implemented two optimizations in sequence:

1. **Part A** - Batched decode: admit multiple requests and decode them together in a single GPU forward pass.
2. **Part B** - Continuous (iteration-level) batching: allow the batch composition to change every decode step, so new requests join immediately and finished requests free their slots without waiting for the entire batch to drain.

## Design and Implementation

### Changes to the Model Layer (`model.py`)

We threaded an optional `attn_mask` parameter through the entire model stack: `Attention` -> `TransformerBlock` -> `TransformerModel` -> `CausalLM`. When a batched decode provides an attention mask, it is passed directly to `F.scaled_dot_product_attention(attn_mask=...)`. When `None` (the single-request path), the existing `is_causal` logic is preserved. This change is minimal and purely additive -- it does not alter the behavior of single-request prefill or decode.

### Batched Decode in the Engine (`engine.py`)

We added a `batched_decode(requests)` method to the `Engine` class. The key steps are:

1. **Input stacking**: The last generated token from each request is stacked into a `(batch, 1)` tensor. Position IDs are set to each request's individual cache length (not the padded length), ensuring correct RoPE encoding.

2. **KV cache padding**: Per-request KV caches have different sequence lengths. We pad all caches to the maximum length in the batch by allocating zero-filled tensors of shape `(batch, kv_heads, max_cache_len, head_dim)` and copying each request's actual cache into the appropriate slice.

3. **Attention mask construction**: We build a float-valued attention mask of shape `(batch, 1, 1, total_cache_len)` initialized to `-inf`. For each request `i` with cache length `L`, positions `[0, L)` and the final position (the new token) are unmasked (set to `0.0`). Using a float mask avoids SDPA's internal bool-to-float conversion overhead.

4. **KV extraction**: After the forward pass, the returned KV tensors have shape `(batch, kv_heads, max_cache_len+1, head_dim)` with padding in the middle. For each request with actual cache length `L`, we extract positions `[:L]` (existing cache) and `[-1]` (newly generated) and concatenate them into a clean `(1, kv_heads, L+1, head_dim)` cache.

5. **Per-request sampling**: Each request's logits are sampled independently according to its own `SamplingParams`.

### Scheduler Strategies (`scheduler.py`)

We implemented two new scheduling methods alongside the original naive `step()`:

**`step_batch()`** (Part A): Admits up to `max_running` requests from the waiting queue, prefills them sequentially, then enters a decode loop that calls `batched_decode()` on all running requests until every request in the batch finishes. No new requests are admitted mid-batch.

**`step_iterative()`** (Part B): Each call performs two phases:
- *Phase 1 (Prefill)*: Admit as many new requests as there are open slots (up to `max_running - len(running)`), and prefill each one individually.
- *Phase 2 (Decode)*: Execute a single `batched_decode()` step on all running requests (including those just prefilled).
- Finished requests are removed immediately, and control returns to the caller. On the next call, vacated slots are filled by newly admitted requests.

This means the batch composition changes **every iteration** -- a newly prefilled request joins the decode batch in the same step, and a finished request's slot opens up on the very next step.

We also added a helper `_clear_finished()` to factor out the shared logic of scanning for completed requests and moving them to the finished list.

### Other Changes

- **Idle sleep reduction** (`scheduler.py`): Reduced the idle-loop sleep from 5 ms to 1 ms, cutting worst-case TTFT jitter for the first request in a burst.
- **`torch.compile` support** (`engine.py`, `__main__.py`): Added a `--compile` flag that wraps the model with `torch.compile()` for potential kernel fusion benefits. This was not used in the benchmark results below but is available for future exploration.

## Results

We benchmarked with 100 requests, input length 512, output length 256, at concurrencies 1, 4, 8, and 16.

### Baseline (Naive Sequential Scheduling)

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 240          | 249          | 11,654         | 11,685         | 44.8          | 44.9          | 22       |
| 4    | 35,211       | 35,244       | 46,629         | 46,667         | 44.8          | 44.9          | 22       |
| 8    | 81,771       | 81,828       | 93,183         | 93,247         | 44.8          | 44.8          | 22       |
| 16   | 151,484      | 175,004      | 174,634        | 232,984        | 136.1         | 227.5         | 22       |

The baseline processes requests serially. Generation throughput is fixed at 22 tok/s regardless of concurrency. TTFT grows linearly because each request must wait for all preceding requests to fully complete before it can begin.

### Part A: Batched Decode (`step_batch`)

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 238          | 245          | 12,562         | 12,591         | 48.3          | 48.4          | 20       |
| 4    | 13,077       | 18,062       | 30,378         | 30,418         | 67.4          | 69.0          | 34       |
| 8    | 13,565       | 27,314       | 39,615         | 39,676         | 101.5         | 105.2         | 51       |
| 16   | 19,145       | 22,735       | 60,120         | 61,103         | 159.6         | 181.4         | 67       |

Batched decode shows clear throughput scaling: **34 tok/s at concurrency 4** (1.5x baseline), **51 tok/s at concurrency 8** (2.3x), and **67 tok/s at concurrency 16** (3.0x). TTFT improves dramatically at higher concurrencies (e.g., 151s -> 19s at conc=16) because requests no longer wait for the entire preceding queue to drain -- they are admitted in batches.

At concurrency 1, per-token latency increases slightly (44.8 -> 48.3 ms) due to the overhead of KV padding and attention masking in the batched path, even though the batch size is 1. This is an expected tradeoff.

### Part B: Continuous Batching (`step_iterative`)

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 241          | 246          | 12,596         | 12,629         | 48.5          | 48.6          | 20       |
| 4    | 631          | 1,019        | 20,069         | 20,097         | 76.1          | 77.6          | 51       |
| 8    | 1,005        | 1,965        | 29,752         | 29,862         | 112.1         | 115.6         | 68       |
| 16   | 1,985        | 3,942        | 49,780         | 49,966         | 186.6         | 194.6         | 80       |

Continuous batching provides the most significant improvement. The key gains over Part A:

- **TTFT is dramatically reduced**: At concurrency 16, TTFT p50 drops from 19,145 ms (Part A) to **1,985 ms** -- a **9.6x improvement**. This is because new requests no longer wait for the entire batch to finish decoding; they are admitted as soon as a slot opens.
- **Throughput reaches 80 tok/s at concurrency 16**, a **3.6x improvement** over baseline. The batch stays full as long as there is work in the queue, minimizing GPU idle time.
- **Completion latency improves at all concurrency levels**: At concurrency 4, p50 drops from 30,378 ms (Part A) to 20,069 ms; at concurrency 16, from 60,120 ms to 49,780 ms.

### Summary Comparison

| Metric (conc=16) | Baseline | Part A (Batched) | Part B (Continuous) | Improvement (B vs Baseline) |
|-------------------|----------|------------------|---------------------|-----------------------------|
| TTFT p50 (ms)     | 151,484  | 19,145           | 1,985               | **76x**                     |
| Compl p50 (ms)    | 174,634  | 60,120           | 49,780              | **3.5x**                    |
| GenTok/s          | 22       | 67               | 80                  | **3.6x**                    |

## Discussion

### Memory Saturation and Padding Waste

The throughput improvement at concurrency 16 (3.6x) is below the theoretical 5-7x suggested in the milestone spec. We hypothesize this is primarily due to **GPU memory saturation from KV cache padding**. With input length 512 and output length 256, each request's KV cache can grow up to 768 tokens across 36 layers. When batching 16 requests, the padded KV tensors must accommodate the longest cache in the batch for all 16 requests, leading to significant memory waste from padding shorter sequences to the maximum length. This pressure saturates the T4's 16 GB VRAM, forcing the GPU to spend cycles on memory management rather than compute.

To validate this hypothesis, we ran additional experiments on the same T4 instance with shorter sequence lengths using continuous batching (`step_iterative`):

**Input 256, Output 256:**

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 67           | 253          | 9,690          | 11,085         | 37.7          | 42.5          | 28       |
| 4    | 8,143        | 9,661        | 17,446         | 17,742         | 36.5          | 37.4          | 57       |
| 16   | 8,745        | 18,654       | 26,652         | 27,054         | 70.0          | 72.6          | 144      |

**Input 128, Output 128:**

| Conc | TTFT p50 (ms) | TTFT p99 (ms) | Compl p50 (ms) | Compl p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | GenTok/s |
|------|--------------|--------------|----------------|----------------|---------------|---------------|----------|
| 1    | 66           | 267          | 8,066          | 9,326          | 31.4          | 35.5          | 32       |
| 4    | 156          | 267          | 10,139         | 10,438         | 39.0          | 40.4          | 98       |
| 16   | 491          | 952          | 19,322         | 19,363         | 73.4          | 75.2          | 193      |

The trend is clear: as sequence lengths decrease, throughput scales much more favorably:

| Configuration         | GenTok/s (conc=16) | vs. Baseline (22 tok/s) |
|-----------------------|--------------------|------------------------|
| Input 512, Output 256 | 80                 | 3.6x                   |
| Input 256, Output 256 | 144                | 6.5x                   |
| Input 128, Output 128 | 193                | 8.8x                   |

At input 256 / output 256, we already reach 144 tok/s (6.5x), well within the expected 5-7x range. At input 128 / output 128, throughput reaches 193 tok/s (8.8x), exceeding the theoretical target. This confirms that the throughput gap in our primary benchmark is not a fundamental implementation issue but rather a consequence of **memory pressure from long sequences on the T4's limited 16 GB VRAM**.

### Hardware Impact

The T4 GPU's relatively low memory bandwidth (320 GB/s) and 16 GB VRAM further constrain scaling with longer sequences. We briefly tested the same workload (input 256, output 256) on a V100 GPU (16 GB HBM2, 900 GB/s bandwidth) and observed noticeably higher throughput, confirming that the bottleneck is hardware-bound rather than algorithmic. The V100 also offers roughly double the float16 TFLOPS compared to the T4 (31.4 vs 65.1 TFLOPS), meaning it can sustain higher throughput even as batch sizes grow and the workload shifts from memory-bound to compute-bound.

### Takeaways

Our padding-based batched decode implementation works correctly and achieves the expected throughput gains when memory is not the bottleneck. The primary limitation is that naive KV cache padding wastes memory proportional to `batch_size * (max_cache_len - actual_cache_len) * num_layers * 2 * kv_heads * head_dim`. With 16 concurrent requests and sequences up to 768 tokens, this overhead is substantial on a 16 GB GPU. Techniques such as **paged attention** (vLLM-style), which eliminates padding by storing KV cache in non-contiguous pages, or **in-place KV cache management** with pre-allocated buffers, could dramatically reduce this overhead and are natural targets for future milestones.
