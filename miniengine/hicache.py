"""Hierarchical radix prefix cache: GPU KV pages backed by a CPU tier.

HiCache keeps the milestone-3 radix-cache API but changes GPU eviction:
unlocked GPU cache entries are demoted into pinned CPU memory instead of being
discarded.  Later prefix hits promote CPU pages back into fresh GPU pages before
the request page table is built, so model attention still only sees GPU page
indices.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from miniengine.radix_cache import MatchResult

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)

GPU_TIER = "gpu"
CPU_TIER = "cpu"
PENDING_CPU_TIER = "pending_cpu"


@dataclass
class HiCacheMetrics:
    """Aggregate HiCache statistics surfaced through /cache_stats."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    # Permanent drops from the hierarchy. GPU demotions are counted separately.
    total_evicted_pages: int = 0
    total_demoted_pages: int = 0
    total_promoted_pages: int = 0
    total_cpu_evicted_pages: int = 0
    total_gpu_hit_tokens: int = 0
    total_cpu_hit_tokens: int = 0
    promote_time_ms: float = 0.0
    demote_time_ms: float = 0.0
    promote_copy_time_ms: float = 0.0
    promote_wait_time_ms: float = 0.0
    promote_hidden_time_ms: float = 0.0
    demote_copy_time_ms: float = 0.0
    demote_wait_time_ms: float = 0.0
    demote_hidden_time_ms: float = 0.0
    async_demote_submitted_pages: int = 0
    async_demote_completed_pages: int = 0
    async_demote_cancelled_pages: int = 0
    async_promote_submitted_pages: int = 0
    async_promote_completed_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens

    @property
    def demote_hidden_ratio(self) -> float:
        if self.demote_copy_time_ms == 0:
            return 0.0
        return self.demote_hidden_time_ms / self.demote_copy_time_ms

    @property
    def promote_hidden_ratio(self) -> float:
        if self.promote_copy_time_ms == 0:
            return 0.0
        return self.promote_hidden_time_ms / self.promote_copy_time_ms


@dataclass
class _PendingDemotion:
    node: "HiRadixNode"
    gpu_pages: list[int]
    cpu_pages: list[int]
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    submitted_at: float


@dataclass
class _PendingPromotion:
    cpu_pages: list[int]
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    submitted_at: float


class CPUKVMemoryPool:
    """Pinned host-memory KV page pool matching KVMemoryPool's page layout."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        if num_pages < 1:
            raise ValueError(
                f"CPU KV pool needs at least 1 page, got {num_pages}"
            )

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = "cpu"

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self.k_caches = [
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(num_layers)
        ]
        self._free: list[int] = list(range(num_pages))

        logger.info(
            "CPUKVMemoryPool: %d pages x %d tokens (%.2f GiB total)",
            num_pages,
            page_size,
            self.bytes_total / 1024**3,
        )

    @property
    def bytes_per_page(self) -> int:
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.page_size
            * self.head_dim
            * elem
        )

    @property
    def bytes_total(self) -> int:
        return self.bytes_per_page * self.num_pages

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return list(zip(self.k_caches, self.v_caches))

    def allocate(self, num_pages: int) -> list[int]:
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages == 0:
            return []
        if len(self._free) < num_pages:
            raise RuntimeError(
                f"CPU KV pool exhausted: requested {num_pages}, "
                f"have {len(self._free)} free pages"
            )
        out = self._free[:num_pages]
        self._free = self._free[num_pages:]
        return out

    def free(self, page_indices: list[int]) -> None:
        if not page_indices:
            return
        self._free.extend(page_indices)

    def copy_gpu_to_cpu(
        self,
        gpu_pool: "KVMemoryPool",
        gpu_pages: list[int],
        cpu_pages: list[int],
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy K/V pages from GPU pool into pinned CPU pages."""
        if len(gpu_pages) != len(cpu_pages):
            raise ValueError("gpu_pages and cpu_pages must have the same length")

        for layer_idx in range(self.num_layers):
            gpu_k = gpu_pool.k_caches[layer_idx]
            gpu_v = gpu_pool.v_caches[layer_idx]
            cpu_k = self.k_caches[layer_idx]
            cpu_v = self.v_caches[layer_idx]
            for gpu_page, cpu_page in zip(gpu_pages, cpu_pages):
                cpu_k[cpu_page].copy_(gpu_k[gpu_page], non_blocking=non_blocking)
                cpu_v[cpu_page].copy_(gpu_v[gpu_page], non_blocking=non_blocking)

    def copy_cpu_to_gpu(
        self,
        gpu_pool: "KVMemoryPool",
        cpu_pages: list[int],
        gpu_pages: list[int],
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy K/V pages from pinned CPU pages into the GPU pool."""
        if len(cpu_pages) != len(gpu_pages):
            raise ValueError("cpu_pages and gpu_pages must have the same length")

        for layer_idx in range(self.num_layers):
            cpu_k = self.k_caches[layer_idx]
            cpu_v = self.v_caches[layer_idx]
            gpu_k = gpu_pool.k_caches[layer_idx]
            gpu_v = gpu_pool.v_caches[layer_idx]
            for cpu_page, gpu_page in zip(cpu_pages, gpu_pages):
                gpu_k[gpu_page].copy_(cpu_k[cpu_page], non_blocking=non_blocking)
                gpu_v[gpu_page].copy_(cpu_v[cpu_page], non_blocking=non_blocking)

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        bytes_budget: int,
    ) -> "CPUKVMemoryPool":
        elem = torch.tensor([], dtype=dtype).element_size()
        bytes_per_page = (
            2 * num_layers * num_kv_heads * page_size * head_dim * elem
        )
        num_pages = int(bytes_budget // bytes_per_page)
        if num_pages < 1:
            raise ValueError(
                "CPU cache size is too small to hold one KV page "
                f"(budget={bytes_budget} bytes, page={bytes_per_page} bytes)"
            )
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )


class HiRadixNode:
    """Radix node whose pages live in exactly one tier."""

    __slots__ = (
        "parent",
        "children",
        "key",
        "pages",
        "tier",
        "ref_count",
        "last_access",
    )

    def __init__(self) -> None:
        self.parent: HiRadixNode | None = None
        self.children: dict[tuple[int, ...], HiRadixNode] = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.tier: str = GPU_TIER
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


class HiRadixCache:
    """Radix prefix cache with GPU and CPU KV tiers.

    Public methods intentionally mirror RadixCache so Engine/Scheduler can use
    this class as a drop-in replacement.
    """

    def __init__(
        self,
        pool: "KVMemoryPool",
        cpu_cache_size_gb: float,
        overlap: bool = False,
    ) -> None:
        if cpu_cache_size_gb <= 0:
            raise ValueError("cpu_cache_size_gb must be positive for HiRadixCache")

        self.pool = pool
        self.page_size = pool.page_size
        self.root = HiRadixNode()
        self.metrics = HiCacheMetrics()
        self.overlap = overlap
        self.cpu_pool = CPUKVMemoryPool.from_budget(
            num_layers=pool.num_layers,
            num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim,
            page_size=pool.page_size,
            dtype=pool.dtype,
            bytes_budget=int(cpu_cache_size_gb * 1024**3),
        )

        if overlap:
            if not torch.cuda.is_available():
                raise RuntimeError("--hicache-overlap requires CUDA")
            self.copy_stream: torch.cuda.Stream | None = torch.cuda.Stream()
        else:
            self.copy_stream = None
        self._pending_demotions: list[_PendingDemotion] = []
        self._pending_promotions: list[_PendingPromotion] = []

    # -- Helpers -----------------------------------------------------

    @property
    def num_cached_pages(self) -> int:
        self.poll_async_events()
        return self.num_gpu_cached_pages + self.num_cpu_cached_pages

    @property
    def num_gpu_cached_pages(self) -> int:
        self.poll_async_events()
        return self._count_pages(GPU_TIER)

    @property
    def num_cpu_cached_pages(self) -> int:
        self.poll_async_events()
        return self._count_pages(CPU_TIER)

    def _count_pages(self, tier: str) -> int:
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.tier == tier:
                total += len(node.pages)
            stack.extend(node.children.values())
        return total

    def num_evictable_pages(self) -> int:
        """GPU pages that can be freed by demotion right now."""
        self.poll_async_events()
        count = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.ref_count != 0:
                stack.extend(node.children.values())
                continue
            if node.tier == GPU_TIER:
                count += len(node.pages)
            stack.extend(node.children.values())
        return count

    @property
    def pending_demote_pages(self) -> int:
        return sum(len(p.gpu_pages) for p in self._pending_demotions)

    @property
    def pending_promote_pages(self) -> int:
        return sum(len(p.cpu_pages) for p in self._pending_promotions)

    def poll_async_events(self) -> None:
        """Retire completed async copies without blocking the host."""
        if self.copy_stream is None:
            return
        self._drain_pending_promotions(wait=False)
        self._drain_pending_demotions(wait=False)

    # -- Lookup ------------------------------------------------------

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix, promoting CPU hits to GPU."""
        self.poll_async_events()
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        node = self.root
        matched_pages: list[int] = []
        matched_tokens = 0
        cpu_hit_tokens = 0
        gpu_hit_tokens = 0
        now = time.monotonic()

        num_full_pages = len(tokens) // self.page_size
        for page_idx in range(num_full_pages):
            start = page_idx * self.page_size
            end = start + self.page_size
            child = node.children.get(tuple(tokens[start:end]))
            if child is None:
                break

            if child.tier == CPU_TIER:
                cpu_hit_tokens += len(child.key)
                self._promote_node(child)
            else:
                gpu_hit_tokens += len(child.key)

            if child.tier not in (GPU_TIER, PENDING_CPU_TIER):
                raise RuntimeError("HiCache lookup failed to promote CPU page to GPU")

            child.last_access = now
            matched_pages.extend(child.pages)
            matched_tokens += len(child.key)
            node = child

        self.metrics.total_hit_tokens += matched_tokens
        self.metrics.total_cpu_hit_tokens += cpu_hit_tokens
        self.metrics.total_gpu_hit_tokens += gpu_hit_tokens

        if matched_tokens == 0:
            return MatchResult()
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=node,
        )

    # -- Lock ref counting ------------------------------------------

    def inc_lock_ref(self, node: HiRadixNode | None) -> None:
        """Lock node and its ancestors against eviction."""
        if node is None:
            return
        node.ref_count += 1
        if node.parent:
            self.inc_lock_ref(node.parent)

    def dec_lock_ref(self, node: HiRadixNode | None) -> None:
        """Release a lock and refresh LRU timestamps along the path."""
        if node is None:
            return
        if node.ref_count <= 0:
            raise RuntimeError("hicache lock ref underflow")
        node.ref_count -= 1
        node.last_access = time.monotonic()
        if node.parent:
            self.dec_lock_ref(node.parent)

    # -- Insertion ---------------------------------------------------

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[HiRadixNode, list[int]]:
        """Insert GPU pages and return request-owned duplicates to caller."""
        self.poll_async_events()
        node = self.root
        redundant_pages: list[int] = []
        num_pages = min(len(tokens) // self.page_size, len(pages))
        now = time.monotonic()

        for page_idx in range(num_pages):
            token_start = page_idx * self.page_size
            token_end = token_start + self.page_size
            edge_key = tuple(tokens[token_start:token_end])
            page_id = pages[page_idx]

            child = node.children.get(edge_key)
            if child is not None:
                if child.tier == CPU_TIER:
                    self._promote_node(child)
                child.last_access = now
                redundant_pages.append(page_id)
                node = child
                continue

            child = HiRadixNode()
            child.parent = node
            child.key = list(edge_key)
            child.pages = [page_id]
            child.tier = GPU_TIER
            child.last_access = now
            node.children[edge_key] = child
            node = child
            self.metrics.total_inserted_pages += 1

        self._maybe_schedule_background_demotions()
        return node, redundant_pages

    # -- GPU eviction / demotion ------------------------------------

    def evict(self, n_pages_needed: int) -> int:
        """Free GPU pages by demoting unlocked GPU cache leaves to CPU."""
        if n_pages_needed <= 0:
            return 0

        start_free = self.pool.num_free
        self.poll_async_events()

        while self.pool.num_free - start_free < n_pages_needed:
            victim = self._pick_lru_gpu_victim()
            if victim is None:
                break
            self._demote_node(victim)
            self.poll_async_events()
            if (
                self.copy_stream is not None
                and self.pool.num_free - start_free < n_pages_needed
                and self._pending_demotions
            ):
                self._drain_pending_demotions(wait=True, max_items=1)

        while (
            self.copy_stream is not None
            and self.pool.num_free - start_free < n_pages_needed
            and self._pending_demotions
        ):
            self._drain_pending_demotions(wait=True, max_items=1)

        return max(0, self.pool.num_free - start_free)

    def _pick_lru_gpu_victim(self) -> HiRadixNode | None:
        victims: list[HiRadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if (
                node is not self.root
                and node.ref_count == 0
                and node.tier == GPU_TIER
                and node.pages
            ):
                victims.append(node)
            stack.extend(node.children.values())
        if not victims:
            return None
        return min(victims, key=lambda n: n.last_access)

    def _demote_node(self, node: HiRadixNode) -> int:
        if node.tier != GPU_TIER or not node.pages or node.ref_count != 0:
            return 0

        gpu_pages = list(node.pages)
        start = time.perf_counter()
        self._ensure_cpu_free_pages(len(gpu_pages))
        cpu_pages = self.cpu_pool.allocate(len(gpu_pages))

        if self.copy_stream is not None:
            try:
                self._schedule_gpu_to_cpu(node, gpu_pages, cpu_pages)
            except Exception:
                self.cpu_pool.free(cpu_pages)
                raise
            return 0

        try:
            self._copy_gpu_to_cpu(gpu_pages, cpu_pages)
        except Exception:
            self.cpu_pool.free(cpu_pages)
            raise

        node.pages = cpu_pages
        node.tier = CPU_TIER
        node.last_access = time.monotonic()
        self.pool.free(gpu_pages)

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics.total_demoted_pages += len(gpu_pages)
        self.metrics.demote_time_ms += elapsed_ms
        self.metrics.demote_copy_time_ms += elapsed_ms
        return len(gpu_pages)

    def _schedule_gpu_to_cpu(
        self,
        node: HiRadixNode,
        gpu_pages: list[int],
        cpu_pages: list[int],
    ) -> None:
        assert self.copy_stream is not None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            start_event.record()
            self.cpu_pool.copy_gpu_to_cpu(
                self.pool,
                gpu_pages,
                cpu_pages,
                non_blocking=True,
            )
            end_event.record()
        node.tier = PENDING_CPU_TIER
        node.last_access = time.monotonic()
        self._pending_demotions.append(
            _PendingDemotion(
                node=node,
                gpu_pages=gpu_pages,
                cpu_pages=cpu_pages,
                start_event=start_event,
                end_event=end_event,
                submitted_at=time.perf_counter(),
            )
        )
        self.metrics.async_demote_submitted_pages += len(gpu_pages)

    def _maybe_schedule_background_demotions(self) -> None:
        if self.copy_stream is None:
            return
        self.poll_async_events()
        usable_pages = max(1, self.pool.num_pages - 1)
        target_free = max(1, usable_pages // 10)
        projected_free = self.pool.num_free + self.pending_demote_pages
        while projected_free < target_free:
            victim = self._pick_lru_gpu_victim()
            if victim is None:
                break
            n = len(victim.pages)
            self._demote_node(victim)
            projected_free += n

    # -- CPU eviction / permanent drop -------------------------------

    def _ensure_cpu_free_pages(self, needed: int) -> None:
        if self.copy_stream is not None and self.cpu_pool.num_free < needed:
            self._drain_pending_promotions(wait=False)
            self._drain_pending_demotions(wait=False)
        if self.copy_stream is not None and self.cpu_pool.num_free < needed:
            self._drain_pending_promotions(wait=True)
            self._drain_pending_demotions(wait=True)
        while self.cpu_pool.num_free < needed:
            victim = self._pick_lru_cpu_victim()
            if victim is None:
                raise RuntimeError(
                    "CPU cache exhausted and no unlocked CPU cache entry "
                    "can be evicted"
                )
            self._drop_cpu_subtree(victim)

    def _pick_lru_cpu_victim(self) -> HiRadixNode | None:
        victims: list[HiRadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if (
                node is not self.root
                and node.ref_count == 0
                and node.tier == CPU_TIER
                and node.pages
            ):
                victims.append(node)
            stack.extend(node.children.values())
        if not victims:
            return None
        return min(victims, key=lambda n: n.last_access)

    def _drop_cpu_subtree(self, node: HiRadixNode) -> int:
        if node.tier != CPU_TIER:
            raise RuntimeError("attempted to CPU-evict a non-CPU node")
        if node.ref_count != 0:
            raise RuntimeError("attempted to CPU-evict a locked node")

        cpu_pages: list[int] = []
        gpu_pages: list[int] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.ref_count != 0:
                raise RuntimeError("attempted to drop a subtree with a locked node")
            if cur.tier == CPU_TIER:
                cpu_pages.extend(cur.pages)
            elif cur.tier == GPU_TIER:
                gpu_pages.extend(cur.pages)
            elif cur.tier == PENDING_CPU_TIER:
                raise RuntimeError("cannot CPU-evict a subtree with pending demotions")
            stack.extend(cur.children.values())

        parent = node.parent
        if parent is not None:
            parent.children.pop(tuple(node.key), None)
        node.parent = None
        node.children.clear()
        node.pages = []

        self.cpu_pool.free(cpu_pages)
        self.pool.free(gpu_pages)
        self.metrics.total_cpu_evicted_pages += len(cpu_pages)
        self.metrics.total_evicted_pages += len(cpu_pages) + len(gpu_pages)
        return len(cpu_pages)

    def _drain_pending_demotions(
        self,
        *,
        wait: bool,
        max_items: int | None = None,
        force_cancel: bool = False,
    ) -> int:
        if not self._pending_demotions:
            return 0

        freed_gpu_pages = 0
        remaining: list[_PendingDemotion] = []
        processed = 0
        for pending in self._pending_demotions:
            if max_items is not None and processed >= max_items:
                remaining.append(pending)
                continue

            was_ready = pending.end_event.query()
            wait_ms = 0.0
            if not was_ready:
                if not wait:
                    remaining.append(pending)
                    continue
                wait_start = time.perf_counter()
                pending.end_event.synchronize()
                wait_ms = (time.perf_counter() - wait_start) * 1000

            copy_ms = pending.start_event.elapsed_time(pending.end_event)
            self.metrics.demote_copy_time_ms += copy_ms
            self.metrics.demote_wait_time_ms += wait_ms
            self.metrics.demote_hidden_time_ms += max(0.0, copy_ms - wait_ms)
            self.metrics.demote_time_ms += copy_ms

            node = pending.node
            can_complete = (
                not force_cancel
                and node.tier == PENDING_CPU_TIER
                and node.pages == pending.gpu_pages
                and node.ref_count == 0
            )
            if can_complete:
                node.pages = pending.cpu_pages
                node.tier = CPU_TIER
                node.last_access = time.monotonic()
                self.pool.free(pending.gpu_pages)
                self.metrics.total_demoted_pages += len(pending.gpu_pages)
                self.metrics.async_demote_completed_pages += len(pending.gpu_pages)
                freed_gpu_pages += len(pending.gpu_pages)
            else:
                if node.tier == PENDING_CPU_TIER and node.pages == pending.gpu_pages:
                    node.tier = GPU_TIER
                self.cpu_pool.free(pending.cpu_pages)
                self.metrics.async_demote_cancelled_pages += len(pending.gpu_pages)
            processed += 1

        self._pending_demotions = remaining
        return freed_gpu_pages

    def _drain_pending_promotions(
        self,
        *,
        wait: bool,
        max_items: int | None = None,
    ) -> int:
        if not self._pending_promotions:
            return 0

        freed_cpu_pages = 0
        remaining: list[_PendingPromotion] = []
        processed = 0
        for pending in self._pending_promotions:
            if max_items is not None and processed >= max_items:
                remaining.append(pending)
                continue

            was_ready = pending.end_event.query()
            wait_ms = 0.0
            if not was_ready:
                if not wait:
                    remaining.append(pending)
                    continue
                wait_start = time.perf_counter()
                pending.end_event.synchronize()
                wait_ms = (time.perf_counter() - wait_start) * 1000

            copy_ms = pending.start_event.elapsed_time(pending.end_event)
            self.cpu_pool.free(pending.cpu_pages)
            self.metrics.promote_copy_time_ms += copy_ms
            self.metrics.promote_wait_time_ms += wait_ms
            self.metrics.promote_hidden_time_ms += max(0.0, copy_ms - wait_ms)
            self.metrics.promote_time_ms += copy_ms
            self.metrics.async_promote_completed_pages += len(pending.cpu_pages)
            freed_cpu_pages += len(pending.cpu_pages)
            processed += 1

        self._pending_promotions = remaining
        return freed_cpu_pages

    # -- Promotion ---------------------------------------------------

    def _promote_node(self, node: HiRadixNode) -> None:
        if node.tier == GPU_TIER:
            return
        if node.tier != CPU_TIER:
            raise RuntimeError(f"unknown HiCache tier: {node.tier!r}")
        if not node.pages:
            raise RuntimeError("cannot promote a CPU cache node with no pages")

        self.inc_lock_ref(node)
        try:
            cpu_pages = list(node.pages)
            start = time.perf_counter()
            if self.copy_stream is not None:
                self._drain_pending_promotions(wait=False)
            gpu_pages = self.pool.allocate(len(cpu_pages))

            if self.copy_stream is not None:
                try:
                    self._schedule_cpu_to_gpu(cpu_pages, gpu_pages)
                except Exception:
                    self.pool.free(gpu_pages)
                    raise
            else:
                try:
                    self._copy_cpu_to_gpu(cpu_pages, gpu_pages)
                except Exception:
                    self.pool.free(gpu_pages)
                    raise
                self.cpu_pool.free(cpu_pages)
                elapsed_ms = (time.perf_counter() - start) * 1000
                self.metrics.promote_time_ms += elapsed_ms
                self.metrics.promote_copy_time_ms += elapsed_ms

            node.pages = gpu_pages
            node.tier = GPU_TIER
            node.last_access = time.monotonic()

            self.metrics.total_promoted_pages += len(gpu_pages)
        finally:
            self.dec_lock_ref(node)

    def _schedule_cpu_to_gpu(
        self,
        cpu_pages: list[int],
        gpu_pages: list[int],
    ) -> None:
        assert self.copy_stream is not None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            start_event.record()
            self.cpu_pool.copy_cpu_to_gpu(
                self.pool,
                cpu_pages,
                gpu_pages,
                non_blocking=True,
            )
            end_event.record()
        torch.cuda.current_stream().wait_event(end_event)
        self._pending_promotions.append(
            _PendingPromotion(
                cpu_pages=cpu_pages,
                start_event=start_event,
                end_event=end_event,
                submitted_at=time.perf_counter(),
            )
        )
        self.metrics.async_promote_submitted_pages += len(cpu_pages)

    # -- Copy dispatch -----------------------------------------------

    def _copy_gpu_to_cpu(self, gpu_pages: list[int], cpu_pages: list[int]) -> None:
        if self.copy_stream is None:
            self.cpu_pool.copy_gpu_to_cpu(self.pool, gpu_pages, cpu_pages)
            return

        current = torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.cpu_pool.copy_gpu_to_cpu(
                self.pool,
                gpu_pages,
                cpu_pages,
                non_blocking=True,
            )
        current.wait_stream(self.copy_stream)

    def _copy_cpu_to_gpu(self, cpu_pages: list[int], gpu_pages: list[int]) -> None:
        if self.copy_stream is None:
            self.cpu_pool.copy_cpu_to_gpu(self.pool, cpu_pages, gpu_pages)
            return

        current = torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.cpu_pool.copy_cpu_to_gpu(
                self.pool,
                cpu_pages,
                gpu_pages,
                non_blocking=True,
            )
        current.wait_stream(self.copy_stream)

    # -- Maintenance -------------------------------------------------

    def reset(self) -> None:
        """Drop the full hierarchy and return pages to both pools."""
        if self.copy_stream is not None:
            self._drain_pending_promotions(wait=True)
            self._drain_pending_demotions(wait=True, force_cancel=True)

        gpu_pages: list[int] = []
        cpu_pages: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.tier == GPU_TIER:
                gpu_pages.extend(node.pages)
            elif node.tier == CPU_TIER:
                cpu_pages.extend(node.pages)
            stack.extend(node.children.values())

        self.pool.free(gpu_pages)
        self.cpu_pool.free(cpu_pages)
        self.root = HiRadixNode()
