# Milestone 4 Presentation — HiRadixCache (Track 1)

**Format:** 5 minutes, 6 slides. Aim ~45–70s per slide. Clarity over completeness.

**The one thing to remember (build everything around this):**
> A pinned DRAM tier underneath the GPU prefix cache turns the multi-turn
> hit-rate *cliff* (73% → 4.6%) into a *rising line* (73% → 86%) — and because
> the copies overlap with compute, you also get a real end-to-end win
> (+25% throughput, −35% TTFT), not just more capacity.

**Secondary point:** the interesting asymmetry — *demotion* can be fully hidden,
*promotion* can't (the request needs the pages), yet async still wins.

All numbers below are from the report (`milestone4_hicache_report.md`) and the
committed runs under `logs/milestone4/20260531_*`.

---

## Slide 1 — Title + the hook (~20s)

**On the slide:**
- Title: **HiRadixCache: a CPU KV tier under the GPU prefix cache**
- Subtitle: Track 1 — Hierarchical KV cache (GPU + CPU), on the Milestone-3 engine
- Authors, course, date
- One-line hook: *"When the GPU prefix cache fills on long chats, hit rate falls off a cliff. We catch the evicted prefixes in DRAM instead of dropping them."*
- (Optional) a tiny teaser of the cliff-vs-flat chart in the corner

**Speaker notes:**
"We took Track 1 — HiCache. The Milestone-3 engine has a GPU radix prefix cache,
but it lives entirely in HBM. Our one-line story: on long multi-turn workloads
that cache overflows and the hit rate collapses — and we fix that by adding a
cheap CPU-memory tier underneath it. I'll show you the collapse, the fix, and
one surprising result about overlap."

---

## Slide 2 — The problem: the GPU-only cliff (~55s)

**On the slide:**
- Heading: *HBM-only prefix caching falls off a cliff on multi-turn*
- The mechanism in one line: GPU pool full → LRU eviction **drops** the prefix → next turn re-prefills from scratch.
- The evidence (small table or callout), GPU-only baseline, Qwen3-8B / L4, 32 sessions × 8 turns, round-robin:

  | Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | Hit rate | 72.7% | 75.1% | 47.1% | 37.9% | 18.9% | 9.7% | **4.6%** |

- Capacity wall: GPU KV pool = **0.97 GB / 205 pages** (`mem-fraction-static 0.75`); working set exceeds it around turn 3.

**Speaker notes:**
"Here's the failure mode. In a multi-turn chat each turn reuses the whole earlier
conversation, so if we cache prefixes we skip most of the prefill. But our GPU KV
pool only holds about 200 pages. With 32 concurrent sessions and growing prompts,
the working set blows past that around turn 3. Once the cache is full, eviction
just *drops* the prefix — so later turns re-prefill from scratch. You can see it:
hit rate starts at 73%, and by turn 7 it's down to under 5%. That's the cliff,
and it hits exactly when caching should be helping most."

---

## Slide 3 — The design: demote instead of drop, promote on hit (~60s)

**On the slide:**
- Heading: *Add a DRAM tier; move pages between tiers instead of dropping them*
- A simple 2-tier diagram: **GPU pool (execution tier)** ⇄ **pinned CPU pool (capacity tier)**
  - **Demote (D2H):** on GPU eviction, copy the LRU prefix to CPU instead of freeing it
  - **Promote (H2D):** on a cache hit for a CPU-resident prefix, copy it back to GPU, then run
  - **CPU LRU evict** only when the DRAM tier itself fills (no lower tier)
- Where it lives: `miniengine/hicache.py` (tier-aware radix tree + pinned CPU pool), under the existing `radix_cache.py` interface — chunked prefill / paged attention / batching unchanged
- The key invariant (one line): **requests only ever see GPU page IDs; a CPU-resident match is promoted *before* the request runs.**
- Capacity: 12 GiB CPU tier ≈ **13× the GPU pool** (DRAM is 10–100× bigger than HBM on the same box)

**Speaker notes:**
"The fix is a hierarchy. GPU memory stays the execution tier — that's the only
place PagedAttention can read. Underneath it we add a pre-allocated, pinned CPU
pool as a capacity tier. Two operations: when the GPU pool needs space, instead
of dropping the least-recently-used prefix we *demote* it — a device-to-host copy
into DRAM. And when a later request matches a prefix that's sitting in CPU, we
*promote* it back — host-to-device — and then run. The radix tree just tracks
which tier each node is in. The important invariant for correctness: the
scheduler never sees CPU pages — anything matched in CPU is promoted to GPU
before the request executes. It's a drop-in under our existing cache. Our CPU
tier is 12 gigs — about 13× the GPU pool, and DRAM scales far past that."

---

## Slide 4 — Key trade-off: blocking vs. async overlap (~55s)

**On the slide:**
- Heading: *Hiding the PCIe cost — and why demotion is easier than promotion*
- The asymmetry (the secondary takeaway):
  - **Demotion** → request *doesn't* need those pages after the copy is launched → **fully hideable**
  - **Promotion** → request *needs* the pages before it runs → true data dependency → can't be fully hidden
- How: `--hicache-overlap` = dedicated CUDA stream + events + pinned host memory; the **compute stream waits on the copy-complete event** (correctness, no stale KV)
- Measured (overlap run): **D2H demote 99.999% hidden** (host waited 0.006 ms over ~1062 ms of copy); 1647 pages demoted / 1304 promoted, 0 pending at end
- Trade-off framing: blocking *already* wins (+16% throughput within the same session), but the synchronous copies sit on the critical path; overlap hides them and pushes the win to +25%

**Speaker notes:**
"Naively, all this copying could cost more than it saves — so the interesting
engineering is hiding it. There's a nice asymmetry. Demotion is easy to hide:
once we've launched the copy, the current request doesn't care about those pages,
so the copy runs on a separate CUDA stream behind ongoing compute. Our counters
say 99.999% of demote-copy time is hidden — the host essentially never blocks.
Promotion is fundamentally harder: the request that triggered it *needs* those
pages to run, so the compute stream has to wait on the copy-complete event before
it reads them — that's also what guarantees we never attend over stale KV. Even
with that dependency, doing the copies asynchronously on pinned memory is what
takes us from a +16% blocking win to +25% with overlap."

---

## Slide 5 — Results: the cliff becomes a rising line, plus an e2e win (~70s)

**On the slide (this is the climax — lead with the chart):**
- **Line chart, per-turn hit rate, two lines:**
  - GPU-only: 72.7 → 75.1 → 47.1 → 37.9 → 18.9 → 9.7 → **4.6%** (collapses)
  - HiRadixCache: 72.7 → 76.6 → 79.3 → 81.6 → 83.6 → 84.9 → **86.1%** (rises)
  - *(this hit-rate line is the same with or without overlap — overlap changes timing, not what's cached)*
- Overall hit rate: **29.6% → 76.0%**
- End-to-end (overlap vs. Milestone-3 baseline), same workload:
  - Request throughput **+24.9%**, generated-token throughput **+25.2%**
  - TTFT p50 **−34.8%** (827 → 539 ms); wall time 35.3 s → 28.2 s
- Correctness: MMLU **60.5% / 60.5% / 61.0%** (baseline / blocking / overlap) — within noise; 256/256 completions, no OOM in either tier

**Speaker notes:**
"Here's the payoff. Same workload, HiCache on. The red line is the GPU-only cliff
from before. The green line is HiCache — and it doesn't just stay flat, it *rises*
to 86% by turn 7, because as conversations get longer there's more shared prefix
to reuse. Overall hit rate goes from 30% to 76%. And the milestone says we don't
even have to win on wall-clock — but with overlap we do: about +25% throughput and
−35% on median time-to-first-token versus the Milestone-3 baseline. One honest
caveat — that overlap run is a separate session, but blocking on its own is already
+16% in the same session, so the speedup is solid either way. And it's still
correct — MMLU is flat at ~60.5% across all three configs, and everything completes
with no OOM. So: more capacity, faster, and no accuracy hit."

---

## Slide 6 — Takeaways & open directions (~40s)

**On the slide:**
- **Takeaways (the 1–2 things to remember):**
  1. A cheap DRAM tier under HBM converts the prefix-cache *cliff* into a *rising* hit rate — capacity is the lever on long-horizon workloads.
  2. Counter-intuitive: we *add* PCIe copy work yet get faster, because we remove far more prefill compute — and async overlap hides almost all of the copy.
- **A finding worth sharing:** the workload schedule matters — a *round-robin* turn order was needed to create realistic reuse distance; running each session start-to-finish hid the cliff.
- **Open directions / next:**
  - Strongest correctness check to add next: greedy **token-identity** test (HiCache output == GPU-only output)
  - Predictive **prefetch** of next-turn prefixes; disk L3 tier; O(1) LRU eviction
  - Harden the CPU-tier-full path under async overlap (not exercised here — tier stayed <15% full)

**Speaker notes:**
"Two things to take away. First, on long-horizon workloads, *capacity* is the
lever — a tier an order of magnitude bigger flips the cliff into a rising line.
Second, the counter-intuitive part: we add copy work but get faster, because we
remove much more prefill compute, and the copies overlap with it. One practical
finding: the cliff only shows up if you interleave sessions round-robin — run each
chat end-to-end and the GPU cache hides the problem, so workload design matters.
Next steps: a token-identity test to nail down correctness, and prefetching
prefixes before the turn arrives. Thanks — happy to take questions."

---

## Design / delivery tips

- **The centerpiece is the Slide 5 two-line chart** (cliff vs. rising). If you make
  one good visual, make that one. Slide 2's table is the same data in raw form —
  consider showing only the chart on 5 and the short table on 2.
- Keep text sparse — these notes are for *you*, not the slide. 4–6 bullets max per slide.
- Time budget: S1 20s · S2 55s · S3 60s · S4 55s · S5 70s · S6 40s ≈ 5:00. Practice S3/S4
  out loud — they're the easiest to overrun.
- Honesty guardrail: the e2e win compares the overlap run (`20260531_000407`) to the
  baseline, which is a *separate serving session* from the GPU-only + blocking runs
  (`20260531_000152`). If asked, the within-session blocking-vs-baseline number is
  already +16.3% throughput; overlap adds the rest. CPU tier really is ~13× (12 GiB),
  so ">10×" is safe to claim.
- If you want, I can turn this into actual slides (Marp or reveal.js markdown, or a
  PPTX outline) and generate the Slide-5 chart from the logged per-turn numbers.

---

## Likely questions (prep — not slides)

Short, defensible answers, all backed by the report / `logs/milestone4/20260531_*`.

1. **Why does promotion not hide as well as demotion?**
   The request that triggered the promotion *needs* those pages to run, so the
   compute stream must wait on the copy-complete event. Demotion has no waiting
   consumer, so it overlaps almost perfectly (99.999% hidden).

2. **If you add PCIe copies, why is it faster, not slower?**
   A prefix hit replaces re-running prefill over thousands of tokens (heavy GPU
   compute) with a page copy that overlaps with other work. We remove far more
   work than we add — hit rate goes 29.6% → 76.0%, so most prefill is skipped.

3. **Is the output still correct?**
   Yes. Requests only ever consume GPU page IDs; a CPU-resident match is promoted
   *before* the request runs; the compute stream waits on the copy event so it
   never reads stale KV. MMLU is flat (60.5 / 60.5 / 61.0%) and all 256 requests
   complete with no OOM.

4. **Why the round-robin schedule?**
   It spaces each session's next turn behind the other 31 sessions, which creates
   the long reuse distance real multi-turn has. Running each chat start-to-finish
   keeps its prefix hot in GPU and hides the cliff — not the scenario we're after.

5. **Is the end-to-end comparison fair?**
   GPU-only and blocking are the *same* serving session; overlap is a separate
   session ~2 min later. Within-session, blocking alone is already +16%
   throughput, so the win is robust; overlap adds the rest (to +25%).

6. **What happens when the CPU tier fills?**
   LRU eviction drops the oldest CPU prefix entirely (no lower tier). In these
   runs the tier stayed <15% full, so that path wasn't stressed — hardening it
   under async overlap is future work.

7. **Why is the GPU KV pool so small (0.97 GB)?**
   Qwen3-8B weights take ~16.8 GB of the 17.7 GB static budget on the L4, leaving
   ~1 GB for KV at `mem-fraction-static 0.75`. That's a realistic setting, not an
   artificial squeeze — and it's what makes the cliff appear at this session count.
