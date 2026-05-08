# Milestone 2 — Paged KV Cache + Compile/Graph Acceleration

**Author:** Joey Ji (junyiji3@stanford.edu) | Thomason Zhao (changyuz@stanford.edu)
**Hardware:** NVIDIA L4 (24 GB), CUDA 13.2, torch 2.6.0+cu124, flash-attn 2.7.4.post1
**Model:** `Qwen/Qwen3-8B`, bf16

---

## Summary

Milestone 1 left the engine bottlenecked on the way it stored its KV cache: every decode step grew per-request tensors with `torch.cat`, fragmented the allocator, and capped concurrency. This milestone replaces that with a pre-allocated **paged KV pool** and rebuilds prefill and decode on top of it using `flash-attn`'s paged kernels. With paging in place, we layered on `torch.compile` (max-autotune, decode-block scope) and manual CUDA-graph capture.

The headline result on L4 with Qwen3-8B at 32-way concurrency:

| | M1 batched | + paging (Part B) | + compile (Part C) | + cuda-graph (extra) |
|---|---|---|---|---|
| Generation throughput | 71 tok/s | 147 tok/s | 153 tok/s | **157 tok/s** |
| TPOT p50 | 215.4 ms | 95.5 ms | 92.4 ms | **89.9 ms** |
| Speedup vs M1 | 1.00× | **2.07×** | 2.15× | **2.21×** |

The rest of this report walks through what we built, what we measured, and what we'd do next.

---

## Design and implementation

### Part A — The KV memory pool

The pool lives in `miniengine/kv_memory_pool.py`. At startup we pre-allocate one K tensor and one V tensor per layer, each shaped `(num_pages, page_size, num_kv_heads, head_dim)`. K and V are kept as separate tensors rather than fused into a single KV tensor because that's the layout flash-attn's paged kernels expect — so attention can read straight from the pool without a transpose at every call.

For tracking which pages are free, we use two cooperating data structures: a deque of free-page indices for O(1) allocate and release, plus a set of those same indices for safety. The set lets us detect double-free at the point it happens (releasing a page that's already in the set raises immediately) rather than silently handing the same page to two different requests later and discovering the corruption when one of them reads garbage KV.

Pool capacity is derived at startup from `--mem-fraction-static × total_GPU_memory`, minus the model parameter bytes, minus a 512 MiB safety margin. At `--mem-fraction-static 0.85` on a 24 GB L4 with a 16.4 GB Qwen3-8B in bf16, that leaves room for 84 pages of 256 tokens each — 21,504 tokens of total KV budget across all in-flight requests.

Each request keeps two pieces of paging state: a **page table** (a small Python list holding the indices of the pages it has been allocated, in order) and a **`cache_len`** counter that records how many KV tokens have actually been written into those pages so far. When a decode step pushes `cache_len` past the next page boundary, the scheduler grabs another page from the free list and appends its index to the page table. Admission control is per-step rather than worst-case: at each scheduler step we reserve enough free pages to cover the immediate decode growth of currently-running requests plus the next-page boundary of any prompts about to be admitted, and if there isn't room, we don't admit them this step. We do **not** pre-reserve every page that a request might need across all of `max_new_tokens`, so a long-running decode can still hit a pool-exhaustion error mid-generation if too many other requests grow at once; that case is surfaced as a request failure rather than a silent corruption.

### Part B — Paged attention

Paging only pays off if both prefill and decode can index into the pool without per-step copies. Two kernel paths handle this:

**Packed batched prefill** uses `flash_attn_varlen_func`. Instead of running each newly admitted prompt through prefill one at a time, the scheduler concatenates all N admitted prompts into one long flat sequence (so total length = sum of prompt lengths) and runs a single forward pass over it. The per-prompt boundaries are passed to flash-attn as a `cu_seqlens` offset array, which it uses to compute block-diagonal causal attention — every prompt attends only to itself, and there is no padding between prompts. To write the resulting K and V values into the pool, we precompute a `slot_mapping` tensor that maps each packed token to the (page, slot-within-page) it should land in, and do a single scatter: `flat_k[slot_mapping] = k_packed`. For the LM head, prefill only needs logits at the last token of each prompt, so we pass an explicit `logit_positions = cu_seqlens[1:] - 1` and the head matmul runs on `num_requests × vocab` rows instead of `total_packed_tokens × vocab` — for an 8-request packed prefill at ~480 tokens each, that's roughly a 480× FLOP cut on the heaviest single op of prefill.

**Paged decode** uses `flash_attn_with_kvcache`. Each decode step builds two small int32 tensors and hands them to the kernel: `block_table` has one row per running request listing the page indices that request currently owns, and `cache_seqlens` has one entry per request giving its current KV length. The kernel uses these to write the new token's K and V into the right slot of the right page in place, and to read each request's full KV history through its page table — all in one fused call per layer. There's no padding across requests, and no per-step `torch.cat` to grow tensors.

A small `PagedContext` dataclass threads the pool tensors and per-request state (block tables, cache lengths, slot mappings) through the model — `Attention → TransformerBlock → TransformerModel → CausalLM` — so the layer code itself stays free of pool bookkeeping.

One sharp edge worth flagging here: `flash_attn_with_kvcache` requires the page size to be a multiple of 256. With `--page-size 32` (the milestone doc's default) the kernel raises on first call; our code catches this once, sets a module-level flag that disables the flash decode path for the rest of the process, and falls back to a manual SDPA gather. The fallback is numerically correct but materially slower — see the page-size sweep below for the cost.

### Part C — `torch.compile` and CUDA graphs

**`--torch-compile`** wraps two regions of the model. The first is `TransformerBlock._paged_decode_block`, which covers the entire body of one decode step inside a single transformer block: input layernorm, Q/K/V projection, Q/K-Norm, RoPE, paged attention, o_proj, post-attention layernorm, MLP, and the two residual adds. The second is `CausalLM._project_logits`, which is the standalone LM head matmul. Both are compiled with `dynamic=False`, which means we get a separately-compiled specialization for each batch size we run at — and we lean on that specialization deliberately, because shape-stable code lets inductor pick the best triton config for that shape.

Two engine-level details make compile actually pay off rather than thrashing on every step. First, dynamo's compilation cache has a default size limit of 8 — far too small for our case, because `_paged_decode_block` is a method on `TransformerBlock` and dynamo keys cached compilations on `(layer_id, batch_size)`. With 36 transformer blocks and up to 16 concurrent requests, we can need up to 576 slots, so we raise the limit to `num_layers * max_running + 64` (= 640). In practice this is enough for the decode block itself to stay cached across the whole bench, though we did observe at least one warning in the compile-mode server log indicating a related compiled function (`_paged_decode_attention_flash`) hit the 640 cap once during steady-state — the run completes correctly but a small amount of recompilation overhead is included in the measured numbers. Second, engine init runs a `_warmup_compile_buckets` routine that does a few decode iterations at every batch size from 1 to `max_running` against a reserved scratch page, so the bulk of the compile cost lands at startup instead of in the first user request.

The compile mode is `"max-autotune-no-cudagraphs"`. The first half of the name (`max-autotune`) lets inductor benchmark a variety of triton kernel configurations and pick the fastest for each shape; the second half (`-no-cudagraphs`) tells it not to wrap the result in its own CUDA-graph capture, because we want our manual capture path below to own that. The flash-attn paged-decode call is the only graph break left in the compile region — it's a pybind C++ extension and opaque to dynamo — so everything around it (the norms, RoPE, projections, MLP, residuals) ends up as one fused inductor kernel chain per layer.

**`--cuda-graph`** (extra credit) captures one CUDA graph for each possible batch size from 1 up to `max_running` — i.e. with `--max-running 16` we capture 16 graphs at startup (sizes 1, 2, 3, ..., 16). At runtime, when the scheduler builds a decode batch with B running requests, it picks the graph that was captured at exactly batch size B and replays it. Because we capture every batch size the scheduler can produce, normal replay never has padded rows; the padded-row / scratch-page code below is a safety / generalization path that would only kick in if we later switched to a sparser bucket set. All graphs replay against the same pre-allocated input buffers (`input_ids`, `positions`, `cache_seqlens`, `block_table`), which the rest of the engine writes into via `copy_(non_blocking=True)` rather than reallocating.

The capture path has three correctness invariants that took some care to get right:

```python
# 1. (Safety / generalization path.) If a captured bucket ever runs with
#    a smaller live batch, padded rows get their own reserved scratch
#    pages so flash-attn's in-place K/V writes from those padded rows
#    can't corrupt or race a real running request's KV. Our current CLI
#    captures one graph per exact batch size, so this never fires at
#    runtime — but it's the right invariant for any future bucket policy.
KVMemoryPool.reserve_scratch_pages = max_running

# 2. Before every replay, we assert the captured input-buffer pointers are
#    still the same tensors we captured against. If anything has been
#    reallocated, replay would silently read stale memory.
self._check_buffer_identity()

# 3. Once any graph has been captured, the decode input buffers are
#    frozen — any code path that wants to grow them raises immediately
#    rather than silently breaking the captured pointers.
if self._graph_active and need_grow: raise RuntimeError(...)
```

Sampling deliberately stays outside the captured region, because per-request top-k/top-p sampling with `multinomial` and `.item()` would break capture (CUDA-graphs forbid CPU↔GPU sync inside the captured region). The engine also fails fast at startup if you ask for cuda-graph but aren't running paged mode on CUDA with flash-attn available and the page size divisible by 256 — those constraints together are what make the capture safe.

---

## Performance

All benchmarks use the same command and prompt distribution:

```bash
python -m benchmark.bench_serving --input-len 512 --output-len 256 \
    --randomness 0.5 --concurrencies 1,2,4,8,16,32
```

Server `--max-running 16`, `--mem-fraction-static 0.85`, `--page-size 256`. Prompts sampled from WildChat. Each concurrency runs `max(2*conc, 8)` requests.

### M1 batched baseline

This is the milestone-1 batched mode: dynamic per-request KV tensors grown with `torch.cat` each decode step, padded SDPA decode, sequential prefill in the scheduler.

![M1 batched baseline benchmark](../screenshots/Screenshot1.png)

The numbers tell two stories. At conc=32, throughput tops out at 71 tok/s and TPOT p50 climbs to 215 ms — every step is paying for the padded KV layout and the per-step cat overhead, both of which scale with batch and context. Worse, TTFT p50 at conc=32 is **33.8 seconds**: the scheduler runs prompts through prefill one at a time, and the queue backs up immediately. This is the ceiling we need to beat.

### Paged

Same workload, but with the paged pool, packed batched prefill, and `flash_attn_with_kvcache` for decode.

![Paged benchmark](../screenshots/Screenshot2.png)

At conc=32, throughput doubles to 147 tok/s — that's **2.07× over the M1 baseline**. TPOT p50 drops 56% (215 → 96 ms) and stays nearly flat across the concurrency sweep (66 → 96 ms from conc=1 to conc=32) because per-request attention now scales with that request's actual KV length, not with the batch's max-padded length. TTFT p50 at conc=32 collapses to 16.3 s — half the batched number — entirely because the scheduler now packs all admitted prompts into one `flash_attn_varlen_func` call instead of running them sequentially.

The intuition for why this works is straightforward. Batched decode pays three compounding costs that paged eliminates. First, every decode step grows each request's K and V tensors with `torch.cat` along the sequence dimension; at conc=32 that is 36 layers × 32 requests × 2 (one for K, one for V) = 2,304 fresh allocations and memory copies *per decode iteration*, which fragments the CUDA allocator and caps how many requests we can keep live at once. Second, batched SDPA pads every request's KV to the longest one in the batch and masks out the padding, so attention compute is wasted in proportion to how much the requests' KV lengths differ. Third, prefill runs one prompt at a time inside the scheduler. The paged pool replaces all three: KV append is an O(1) push to the page table (no reallocation), attention reads exactly the valid positions for each request through that page table (no padding waste), and prefill is one packed kernel call (no serialization).

### Paged + `torch.compile`

Same paged engine, with `--torch-compile`. The compile region is the entire decode block plus the LM head.

![Paged + torch.compile benchmark](../screenshots/Screenshot3.png)

At conc=1 this is the cleanest place to read the Part C win. TPOT p50 drops from 66.0 ms to 59.6 ms — a 9.7% latency reduction, just under the 10% latency bar. Throughput at conc=1 is just `1000 / TPOT` tokens/sec, so the same TPOT change works out to `15.15 → 16.78` tok/s, a 10.7% throughput gain that does meet the 10% bar. The bench's `GenTok/s` column prints integers and rounds both runs to 15, which obscures the win at a glance; the improvement is real and shows up directly in the TPOT field. Whether one calls this "the bar cleared" depends on which of throughput-vs-latency one privileges; we treat the asymmetry as honest and report both.

Above conc=1 the gain plateaus at +3–5% on throughput and TPOT both, and the reason is structural rather than something we can fix with more compile work. An 8B model in bf16 has roughly 16 GB of weights, and L4's spec sheet quotes around 300 GB/s of memory bandwidth — so even in the best case the GPU has to spend at least `16 / 300 ≈ 53 ms` per decode iteration just streaming weights through the multipliers. Our measured per-iter wall time at conc=32 (which is just TPOT p50 at that concurrency, since each iter produces one token per running request) is around 95 ms for paged and 92 ms for paged+compile, so the GPU is already operating at well over half of its theoretical bandwidth ceiling. What's left for `torch.compile` to attack is the small launch-overhead chunk between the bandwidth-bound matmuls; saving a handful of µs per layer × 36 layers comes out to a few percent of a 95 ms iteration, which is exactly what we measure. Launch fusion really only pays off at conc=1, where the GPU is *not* bandwidth-saturated yet and the small projections in decode are GEMV-shaped (M=1) — precisely the kind of kernel triton autotune is good at speeding up.

### Paged + `torch.compile` + CUDA graphs (extra credit)

Same engine, with manual CUDA-graph capture stacked on top.

![Paged + compile + cuda-graph benchmark](../screenshots/Screenshot4.png)

At conc=32 this gets to 157 tok/s and TPOT p50 89.9 ms — **+6.8% throughput and −5.9% TPOT over paged-only**, and **+2.7% throughput and −2.7% TPOT over compile**. The extra credit asked for ≥20% over compile and we're well short of that, but the wins are clean and consistent across the sweep, with TPOT improvements ranging from −7.0% to −8.3% over paged at conc=1–8.

The reason cuda-graph adds so little on top of compile is the same reason compile added little on top of paged: once compile fusion has already merged the small kernels into a few big ones, there isn't much launch overhead left for cuda-graph to eliminate. And once we're bandwidth-bound, dispatching the kernels even faster doesn't help — the GPU spends its time waiting for weights from HBM, not waiting for the CPU to issue the next launch.

### Accuracy

All four configurations score **62.0% (124/200)** on MMLU 200 with greedy sampling — same final accuracy, same number correct (124), same number wrong (76). The intermediate running-accuracy lines printed during each run differ slightly between configurations because samples complete in non-deterministic order under concurrent serving, but the final per-sample correctness sets converge to the same numbers across all four configs. This is the strongest signal we have that max-autotune compile and cuda-graph capture preserve numerical behavior end-to-end on this benchmark; a per-sample diff across configs would be a stronger guarantee than this report contains, and would be a worthwhile next step.

| Configuration | MMLU 200 
|---|---|
| `batched` (M1) | 62.0% (124/200) | 
| `paged` | 62.0% (124/200) | Screenshot 6 |
| `paged + compile` | 62.0% (124/200) | Screenshot 7 |
| `paged + compile + cuda-graph` | 62.0% (124/200) | Screenshot 8 |

![MMLU — batched](../screenshots/Screenshot5.png)
![MMLU — paged](../screenshots/Screenshot6.png)
![MMLU — paged + compile](../screenshots/Screenshot7.png)
![MMLU — paged + compile + cuda-graph](../screenshots/Screenshot8.png)

### Page-size comparison

The milestone asks for a sweep across two `--page-size` values. The interesting break-point on flash-attn is at 256: the paged-decode kernel rejects any block size that isn't divisible by 256.

At `--page-size 32` (the milestone doc's example default), the flash-attn paged-decode kernel raises `RuntimeError: Paged KV cache block size must be divisible by 256` on its very first call. Our guard catches that first failure and disables the flash decode path for the rest of the process; from then on, paged decode falls back to a manual SDPA path that gathers each request's full KV history out of the pool into a dense `(batch, heads, max_kv_len, head_dim)` buffer and runs SDPA with a length mask. The math is identical to the flash path, so accuracy is preserved — but the gather plus dense compute is materially slower:

![Paged with --page-size 32 (SDPA fallback)](../screenshots/Screenshot9.png)

| Metric @ conc=32 | `--page-size 32` (SDPA fallback) | `--page-size 256` (flash-attn fast path) |
|---|---|---|
| Generation throughput | 82 tok/s | 147 tok/s |
| TPOT p50 | 181.7 ms | 95.5 ms |
| MMLU 200 | 62.0% | 62.0% |

Same accuracy (62.0% MMLU on both — the math is identical, only the kernel and layout differ), but **1.79× lower throughput at conc=32** because the SDPA path materializes the full per-request KV history each step instead of letting flash-attn read the page table directly. All other throughput numbers in this report run at `--page-size 256` to keep the fast path active. (`flash_attn_varlen_func` for prefill has no such alignment requirement, so packed prefill stays fast at any page size.)

The takeaway is that the choice of page size is dominated by kernel constraints in our setting, not by the usual "small pages waste less tail KV, large pages keep page tables small" tradeoff. The 1.79× gap is entirely the difference between the flash-attn paged decode kernel and a fallback that doesn't have a paged equivalent.

---

## Next steps

The clearest bottleneck we identified during this milestone is one we can name precisely: **L4 + Qwen3-8B in bf16 is memory-bandwidth-bound at decode**. With ~16 GB of weights streaming through ~300 GB/s of bandwidth, every decode iteration has a hard floor of ~53 ms that no amount of launch-overhead reduction can shrink. This is why compile gets a clean +10% only at conc=1 (where launch dispatch is the bottleneck and bandwidth is not) and why cuda-graph adds only ~3% on top of compile (because compile fusion already consolidated most of the small kernels). On a smaller model in the 1–3 B parameter range running on the same L4, or on the same Qwen3-8B running on an H100, our code would have much more launch overhead left to remove and would land much closer to the ≥20% cuda-graph target. The optimizations themselves are correct — the workload just doesn't have enough launch overhead left to fight over.

We also tried two more aggressive approaches to push compile and cuda-graph further. Neither shipped, but both are worth recording:

- **Whole-decode-forward compile with flash-attn as a `torch.library.custom_op`.** This would have let dynamo trace through the flash-attn pybind call without graph-breaking, so all 36 transformer layers would compile into one continuous graph (which would in turn let `mode="reduce-overhead"` capture the entire decode forward as one cuda-graph). The custom op registered fine, but AOTAutograd's functionalization pass crashed on the post-attention `attn_out.reshape(...)` call with `RuntimeError: <weakref to UntypedStorage>`. This looks like a torch 2.6 bug, and our `enable_torch_compile` already exposes the path so it can be re-enabled on a future torch version with a single flag.

- **`mode="reduce-overhead"` on per-layer compile chains.** This would have let inductor wrap each compiled block with its own internal cuda-graph. It trips inductor's `cudagraph_trees` aliasing guard at the residual add: layer N's output buffer is the same memory as layer N+1's input buffer at capture time, so the runtime refuses with "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run". The `cudagraph_mark_step_begin()` API doesn't help here because it only resets state between user-level forward calls, not between chained compiled functions inside one forward.

Beyond standard request batching, several smaller optimizations shipped in this milestone — some for raw speed, some as prerequisites that the compile and cuda-graph paths needed in order to be safe:

- **Pre-allocated decode-step input buffers** (`input_ids`, `positions`, `cache_seqlens`, `block_table`) live on the GPU and are written via `copy_(non_blocking=True)` from pinned host tensors, replacing per-step `torch.tensor(list, device=cuda)` calls that would each trigger a CPU↔GPU sync. This is also what makes cuda-graph capture safe — capture requires the input tensor identities (storage pointers) to be stable across replays.
- **Bounded RoPE cache** with admission-time rejection of requests beyond the cap. The previous RoPE forward used `.item()` to read the max position, which is forbidden inside a captured graph; bounding the cache lets us drop that sync entirely.
- **Greedy sampling fast path.** When all requests in the batch have `temperature=0` (the bench's default), we replace per-request top-k/top-p sampling with a single batched `argmax(dim=-1).tolist()`. We fall back to the per-request path when needed.
- **Selective LM head at prefill.** Only computing logits at the last token of each prompt cuts the prefill LM-head matmul by roughly the average prompt length — about 480× on our bench workload.
- **Per-bucket scratch pages for cuda-graph capture.** When the live batch is smaller than the captured bucket, the padded rows still execute the in-place K/V write inside flash-attn. Giving each padded row its own reserved scratch page eliminates the race risk that an earlier single-shared-scratch-page design would have created.
- **Buffer-pointer identity checks and frozen decode buffers** once any cuda-graph has been captured, so any code path that would silently invalidate the capture raises immediately instead.

If we were to keep going from here, the three highest-leverage next steps would be:

1. **Chunked / continuous batching** that interleaves prefill chunks into decode batches, instead of running prefill and decode as alternating phases. This would directly attack the 22-second TTFT p99 we still see at conc=32, where new arrivals have to wait for the current decode batch to make progress.
2. **Speculative decoding.** Bandwidth-bound decode is the ideal target for speculation: running a small draft model and verifying its tokens with the 8B in a single batched forward effectively re-uses the same 16 GB-per-iter bandwidth budget to accept 2-4× more output tokens.

---

## CLI flags shipped

| Flag | Default | Description |
|---|---|---|
| `--mode {baseline,batched,paged}` | `batched` | Selects the engine path. |
| `--mem-fraction-static` | 0.85 | Fraction of GPU memory for weights + KV pool. |
| `--page-size` | 32¹ | Tokens per KV page. |
| `--torch-compile` | off | Compile decode block + LM head per bucket (max-autotune-no-cudagraphs). |
| `--cuda-graph` | off | Manual cuda-graph capture for paged decode (extra credit). |
| `--cuda-graph-batch-sizes` | (deprecated) | Ignored — captures exact buckets 1..max-running. |

¹ The doc-suggested default of 32 produces a working but ~1.79× slower SDPA fallback (see page-size comparison). All benchmark numbers in this report run at `--page-size 256` for the flash-attn fast path.
