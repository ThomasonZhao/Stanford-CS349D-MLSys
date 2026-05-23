"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.  This file is the **skeleton** — fill in the methods marked
``TODO``.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an in-flight
insert chooses to free a redundant duplicate.

Performance counters in ``CacheMetrics`` are read by the ``/cache_stats``
endpoint and by the scheduler's per-prefill-batch INFO log line.  Update them
inside your implementation so those observability hooks light up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node.  Design hints — adjust to fit your implementation:

    * ``parent`` / ``children`` form the tree.
    * ``key`` carries the tokens on the edge from the parent.
    * ``pages`` are the KV pages corresponding to those tokens; for safety
      against partial-page sharing, keep ``len(key)`` a multiple of
      ``page_size`` and ``len(pages) == len(key) // page_size``.
    * ``ref_count`` should reflect "number of locked leaves in this
      subtree" so eviction can check a single field.  Manipulated by
      ``inc_lock_ref`` / ``dec_lock_ref``.
    * ``last_access`` drives LRU.
    """

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        self.children: dict = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: RadixNode | None = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Required behaviours (see milestone 3 doc):
      * page-aligned matching — never return a partial-page result
      * LRU eviction of unlocked subtrees
      * eviction-on-allocate: ``KVMemoryPool.allocate`` should call
        ``cache.evict(n)`` when the free list is short
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
        (same names as sglang's radix cache).

    The scheduler is single-threaded today, so this structure intentionally
    does not add its own locking.
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += len(node.pages)
            stack.extend(node.children.values())
        return total

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and node.ref_count == 0:
                total += len(node.pages)
            stack.extend(node.children.values())
        return total

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(
        self, tokens: list[int], max_tokens: int | None = None
    ) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree.

        ``max_tokens`` optionally caps the returned hit.  It is rounded down
        to a page boundary; query-token metrics still count the full request.
        Update ``metrics.total_lookups`` / ``total_query_tokens`` /
        ``total_hit_tokens`` so the perf counters are accurate.
        """
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)
        if max_tokens is None:
            hit_cap = len(tokens)
        else:
            hit_cap = max(0, min(max_tokens, len(tokens)))
        hit_cap = (hit_cap // self.page_size) * self.page_size

        now = time.monotonic()
        self.root.last_access = now

        matched_pages: list[int] = []
        matched_tokens = 0
        node = self.root
        last_node: RadixNode | None = None
        pos = 0

        while matched_tokens < hit_cap and pos + self.page_size <= len(tokens):
            child = node.children.get(self._page_key(tokens, pos))
            if child is None:
                break

            common = self._common_prefix_len(child.key, tokens[pos:])
            matched_in_child = (common // self.page_size) * self.page_size
            matched_in_child = min(matched_in_child, hit_cap - matched_tokens)
            if matched_in_child == 0:
                break

            matched_pages.extend(child.pages[: matched_in_child // self.page_size])
            matched_tokens += matched_in_child
            child.last_access = now
            last_node = child

            if matched_in_child < len(child.key):
                break

            pos += len(child.key)
            node = child

        self.metrics.total_hit_tokens += matched_tokens
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=last_node,
        )

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: RadixNode | None) -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        while node is not None:
            node.ref_count += 1
            node = node.parent

    def dec_lock_ref(self, node: RadixNode | None) -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        now = time.monotonic()
        while node is not None:
            node.ref_count -= 1
            assert node.ref_count >= 0
            node.last_access = now
            node = node.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[RadixNode, list[int]]:
        """Insert (tokens, pages) into the tree.

        Returns ``(leaf_node, redundant_pages)``: ``redundant_pages`` are
        pages the caller handed in that were duplicates of pages already
        cached at the same prefix — the caller should return them to the
        pool.  Update ``metrics.total_inserted_pages`` to reflect what
        actually got added.
        """
        assert len(tokens) % self.page_size == 0
        assert len(pages) == len(tokens) // self.page_size

        if not tokens:
            return self.root, []

        redundant_pages: list[int] = []
        node = self.root
        pos = 0
        new_pages_added = 0

        while pos < len(tokens):
            page_pos = pos // self.page_size
            child = node.children.get(self._page_key(tokens, pos))
            if child is None:
                leaf = self._new_child(node, tokens[pos:], pages[page_pos:])
                new_pages_added += len(leaf.pages)
                self.metrics.total_inserted_pages += new_pages_added
                return leaf, redundant_pages

            common = self._common_prefix_len(child.key, tokens[pos:])
            split_tokens = (common // self.page_size) * self.page_size
            if split_tokens == 0:
                raise AssertionError("radix child matched without a full shared page")

            split_pages = split_tokens // self.page_size

            if split_tokens < len(child.key):
                node = self._split_child(node, child, split_tokens)
                redundant_pages.extend(pages[page_pos : page_pos + split_pages])
                pos += split_tokens
                if pos == len(tokens):
                    self.metrics.total_inserted_pages += new_pages_added
                    return node, redundant_pages

                page_pos = pos // self.page_size
                leaf = self._new_child(node, tokens[pos:], pages[page_pos:])
                new_pages_added += len(leaf.pages)
                self.metrics.total_inserted_pages += new_pages_added
                return leaf, redundant_pages

            redundant_pages.extend(pages[page_pos : page_pos + len(child.pages)])
            pos += len(child.key)
            node = child

        self.metrics.total_inserted_pages += new_pages_added
        return node, redundant_pages

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Return the number actually freed.  Bump
        ``metrics.total_evicted_pages``.  Never touch a locked node.
        """
        if n_pages_needed <= 0:
            return 0

        freed = 0
        while freed < n_pages_needed:
            leaf = self._oldest_evictable_leaf()
            if leaf is None:
                break

            pages = leaf.pages
            leaf.pages = []
            self.pool.free(pages)
            freed += len(pages)
            self._prune_empty_unlocked(leaf)

        self.metrics.total_evicted_pages += freed
        return freed

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        pages: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            pages.extend(node.pages)
            stack.extend(node.children.values())
        self.pool.free(pages)
        self.root = RadixNode()

    # ── Internal helpers ───────────────────────────────────────────────

    def _page_key(self, tokens: list[int], pos: int = 0) -> tuple[int, ...]:
        return tuple(tokens[pos : pos + self.page_size])

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return i

    def _new_child(
        self, parent: RadixNode, key: list[int], pages: list[int]
    ) -> RadixNode:
        assert key
        assert len(key) % self.page_size == 0
        assert len(pages) == len(key) // self.page_size

        child = RadixNode()
        child.parent = parent
        child.key = list(key)
        child.pages = list(pages)
        parent.children[self._page_key(child.key)] = child
        return child

    def _split_child(
        self, parent: RadixNode, child: RadixNode, split_tokens: int
    ) -> RadixNode:
        assert 0 < split_tokens < len(child.key)
        assert split_tokens % self.page_size == 0

        split_pages = split_tokens // self.page_size
        old_key = child.key
        old_pages = child.pages

        prefix = RadixNode()
        prefix.parent = parent
        prefix.key = old_key[:split_tokens]
        prefix.pages = old_pages[:split_pages]
        prefix.ref_count = child.ref_count
        prefix.last_access = child.last_access

        self._replace_child(parent, child, prefix)

        child.parent = prefix
        child.key = old_key[split_tokens:]
        child.pages = old_pages[split_pages:]
        prefix.children[self._page_key(child.key)] = child
        return prefix

    def _replace_child(
        self, parent: RadixNode, old_child: RadixNode, new_child: RadixNode
    ) -> None:
        for key, child in list(parent.children.items()):
            if child is old_child:
                parent.children[key] = new_child
                return
        raise AssertionError("radix child missing from parent")

    def _oldest_evictable_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
                continue
            if node.ref_count > 0 or not node.pages:
                continue
            if best is None or node.last_access < best.last_access:
                best = node
        return best

    def _prune_empty_unlocked(self, node: RadixNode) -> None:
        while (
            node is not self.root
            and not node.children
            and not node.pages
            and node.ref_count == 0
        ):
            parent = node.parent
            assert parent is not None
            self._detach_child(parent, node)
            node.parent = None
            node = parent

    @staticmethod
    def _detach_child(parent: RadixNode, child: RadixNode) -> None:
        for key, candidate in list(parent.children.items()):
            if candidate is child:
                del parent.children[key]
                return
        raise AssertionError("radix child missing from parent")
