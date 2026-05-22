import torch

from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.radix_cache import RadixCache


def _pool(page_size: int = 4, num_pages: int = 64) -> KVMemoryPool:
    return KVMemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )


def test_insert_match_and_metrics_are_page_aligned():
    pool = _pool()
    cache = RadixCache(pool)
    tokens = list(range(12))
    pages = pool.allocate(3)

    leaf, redundant = cache.insert_and_return(tokens, pages)

    assert redundant == []
    assert leaf.pages == pages
    assert cache.num_cached_pages == 3
    assert cache.num_evictable_pages() == 3
    assert cache.metrics.total_inserted_pages == 3

    before_access = leaf.last_access
    result = cache.match_prefix(tokens[:10])

    assert result.matched_pages == pages[:2]
    assert result.matched_tokens == 8
    assert result.last_node is leaf
    assert leaf.last_access >= before_access
    assert cache.metrics.total_lookups == 1
    assert cache.metrics.total_query_tokens == 10
    assert cache.metrics.total_hit_tokens == 8

    miss = cache.match_prefix(tokens[:3])
    assert miss.matched_pages == []
    assert miss.matched_tokens == 0
    assert miss.last_node is None
    assert cache.metrics.total_lookups == 2
    assert cache.metrics.total_query_tokens == 13
    assert cache.metrics.total_hit_tokens == 8


def test_insert_splits_at_page_boundary_and_returns_redundant_pages():
    pool = _pool()
    cache = RadixCache(pool)
    base_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    base_pages = pool.allocate(2)
    cache.insert_and_return(base_tokens, base_pages)

    branch_tokens = [1, 2, 3, 4, 90, 91, 92, 93]
    branch_pages = pool.allocate(2)
    branch_leaf, redundant = cache.insert_and_return(branch_tokens, branch_pages)

    assert redundant == [branch_pages[0]]
    pool.free(redundant)
    assert branch_leaf.pages == [branch_pages[1]]
    assert cache.num_cached_pages == 3
    assert cache.metrics.total_inserted_pages == 3

    base_match = cache.match_prefix(base_tokens)
    branch_match = cache.match_prefix(branch_tokens)

    assert base_match.matched_pages == base_pages
    assert base_match.matched_tokens == 8
    assert branch_match.matched_pages == [base_pages[0], branch_pages[1]]
    assert branch_match.matched_tokens == 8

    duplicate_pages = pool.allocate(2)
    duplicate_leaf, duplicate_redundant = cache.insert_and_return(
        branch_tokens, duplicate_pages
    )

    assert duplicate_leaf is branch_leaf
    assert duplicate_redundant == duplicate_pages
    assert cache.metrics.total_inserted_pages == 3
    pool.free(duplicate_redundant)


def test_insert_shorter_prefix_splits_existing_node_without_adding_pages():
    pool = _pool()
    cache = RadixCache(pool)
    long_tokens = list(range(12))
    long_pages = pool.allocate(3)
    cache.insert_and_return(long_tokens, long_pages)

    prefix_tokens = long_tokens[:8]
    duplicate_pages = pool.allocate(2)
    prefix_node, redundant = cache.insert_and_return(prefix_tokens, duplicate_pages)

    assert redundant == duplicate_pages
    assert prefix_node.pages == long_pages[:2]
    assert prefix_node.parent is cache.root
    assert cache.num_cached_pages == 3
    assert cache.metrics.total_inserted_pages == 3

    result = cache.match_prefix(prefix_tokens)
    assert result.matched_pages == long_pages[:2]
    assert result.matched_tokens == 8
    assert result.last_node is prefix_node
    pool.free(redundant)


def test_lock_ref_pins_matched_node_from_eviction():
    pool = _pool()
    cache = RadixCache(pool)
    old_pages = pool.allocate(2)
    locked_pages = pool.allocate(2)
    old_leaf, _ = cache.insert_and_return(list(range(8)), old_pages)
    locked_leaf, _ = cache.insert_and_return(list(range(100, 108)), locked_pages)
    old_leaf.last_access = 10.0
    locked_leaf.last_access = 1.0

    cache.inc_lock_ref(locked_leaf)

    assert locked_leaf.ref_count == 1
    assert cache.root.ref_count == 1
    assert cache.num_evictable_pages() == len(old_pages)

    freed = cache.evict(2)

    assert freed == 2
    assert cache.num_cached_pages == 2
    assert old_pages[0] in pool._free
    assert old_pages[1] in pool._free
    assert cache.metrics.total_evicted_pages == 2
    assert cache.match_prefix(list(range(100, 108))).matched_pages == locked_pages

    before_unlock_access = locked_leaf.last_access
    cache.dec_lock_ref(locked_leaf)

    assert locked_leaf.ref_count == 0
    assert cache.root.ref_count == 0
    assert locked_leaf.last_access >= before_unlock_access
    assert cache.evict(2) == 2
    assert cache.num_cached_pages == 0
    assert cache.metrics.total_evicted_pages == 4


def test_lru_eviction_walks_leaves_then_internal_prefix_pages():
    pool = _pool()
    cache = RadixCache(pool)
    base_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    base_pages = pool.allocate(2)
    cache.insert_and_return(base_tokens, base_pages)
    branch_tokens = [1, 2, 3, 4, 90, 91, 92, 93]
    branch_pages = pool.allocate(2)
    _, redundant = cache.insert_and_return(branch_tokens, branch_pages)
    pool.free(redundant)

    prefix = cache.root.children[(1, 2, 3, 4)]
    old_suffix = prefix.children[(5, 6, 7, 8)]
    branch_suffix = prefix.children[(90, 91, 92, 93)]
    prefix.last_access = 0.0
    old_suffix.last_access = 1.0
    branch_suffix.last_access = 2.0

    assert cache.evict(1) == 1
    assert cache.num_cached_pages == 2
    assert cache.match_prefix(base_tokens).matched_pages == [base_pages[0]]

    assert cache.evict(1) == 1
    assert cache.num_cached_pages == 1
    assert cache.match_prefix(branch_tokens).matched_pages == [base_pages[0]]

    assert cache.evict(1) == 1
    assert cache.num_cached_pages == 0
    assert cache.root.children == {}
    assert cache.metrics.total_evicted_pages == 3
    assert base_pages[0] in pool._free
    assert base_pages[1] in pool._free
    assert branch_pages[1] in pool._free


def test_reset_returns_all_pages_and_preserves_metrics():
    pool = _pool()
    cache = RadixCache(pool)
    pages = pool.allocate(2)
    cache.insert_and_return(list(range(8)), pages)
    cache.match_prefix(list(range(8)))
    free_before_reset = pool.num_free

    cache.reset()

    assert pool.num_free == free_before_reset + 2
    assert cache.num_cached_pages == 0
    assert cache.num_evictable_pages() == 0
    assert cache.root.children == {}
    assert cache.metrics.total_inserted_pages == 2
    assert cache.metrics.total_lookups == 1
    assert cache.metrics.total_hit_tokens == 8
