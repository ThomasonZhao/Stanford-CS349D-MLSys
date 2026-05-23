import asyncio
import sys

import pytest
import torch

from miniengine import server
from miniengine.__main__ import parse_args
from miniengine.core import Request, SamplingParams
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.radix_cache import RadixCache
from miniengine.scheduler import Scheduler


def _pool(page_size: int = 4, num_pages: int = 16) -> KVMemoryPool:
    return KVMemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def _engine(page_size: int = 4, num_pages: int = 16, cache_enabled: bool = True):
    engine = Engine.__new__(Engine)
    engine.pool = _pool(page_size=page_size, num_pages=num_pages)
    engine.page_size = page_size
    engine._paged_state = {}
    engine.radix_cache = RadixCache(engine.pool) if cache_enabled else None
    if engine.radix_cache is not None:
        engine.pool.evict_cb = engine.radix_cache.evict
    return engine


def _request(request_id: str, tokens: list[int], max_new_tokens: int = 1) -> Request:
    return Request(
        request_id=request_id,
        input_ids=tokens,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, temperature=0),
    )


def _scheduler(engine) -> Scheduler:
    return Scheduler(engine, max_running=1, mode="paged", prefill_chunk_size=4)


class FakeModel:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, input_ids, position_ids, **kwargs):
        logits_indices = kwargs["logits_indices"]
        self.calls.append(
            {
                "input_ids": input_ids.cpu().tolist(),
                "position_ids": position_ids.cpu().tolist(),
                "logits_indices": logits_indices.cpu().tolist(),
            }
        )
        logits = torch.zeros(1, logits_indices.numel(), 16, dtype=torch.float32)
        for i in range(logits_indices.numel()):
            logits[0, i, 7 + i] = 1.0
        return logits, None


def _install_fake_prefill_model(engine) -> None:
    engine.device = "cpu"
    engine.max_position = 128
    engine._kv_pool_caches = []
    engine.attention_backend = "flash_attn"
    engine.model = FakeModel()
    engine._fa_kwargs_chunked_prefill = lambda *args: {}


def _finish_with_materialized_prompt(
    scheduler: Scheduler, engine, req: Request
) -> None:
    state = engine._state(req)
    state.cache_seq_len = len(req.input_ids)
    req.output_ids = [999]
    scheduler._finish_paged_request(req)


def test_same_prompt_twice_hits_and_skips_cached_prefill_tokens():
    engine = _engine()
    scheduler = _scheduler(engine)
    first = _request("first", list(range(10)))

    scheduler._prepare_paged_prefill_for_admission(first)
    assert first.cache_hit_tokens == 0
    _finish_with_materialized_prompt(scheduler, engine, first)

    second = _request("second", list(range(10)))
    scheduler._prepare_paged_prefill_for_admission(second)
    state = engine._state(second)

    assert second.cache_hit_tokens == 8
    assert state.prefill_offset == 8
    assert state.cache_hit_pages == state.page_table[:2]

    _install_fake_prefill_model(engine)
    assert engine.paged_chunked_prefill([second], 4) == [7]
    assert engine.model.calls[-1] == {
        "input_ids": [[8, 9]],
        "position_ids": [[8, 9]],
        "logits_indices": [1],
    }


def test_multiturn_prefix_hit_rate_climbs_across_turns():
    engine = _engine()
    scheduler = _scheduler(engine)
    hits = []

    for turn, length in enumerate([10, 18, 26]):
        req = _request(f"turn-{turn}", list(range(length)))
        scheduler._prepare_paged_prefill_for_admission(req)
        hits.append(req.cache_hit_tokens)
        _finish_with_materialized_prompt(scheduler, engine, req)

    assert hits == [0, 8, 16]
    assert engine.radix_cache.metrics.hit_rate > 0


def test_eviction_on_oom_lets_allocation_succeed():
    pool = _pool(num_pages=5)
    cache = RadixCache(pool)
    pool.evict_cb = cache.evict
    pages = pool.allocate(3)
    cache.insert_and_return(list(range(12)), pages)

    allocated = pool.allocate(2)

    assert len(allocated) == 2
    assert cache.metrics.total_evicted_pages == 3


def test_lock_protects_cache_pages_and_oom_is_raised():
    pool = _pool(num_pages=5)
    cache = RadixCache(pool)
    pool.evict_cb = cache.evict
    pages = pool.allocate(3)
    leaf, _ = cache.insert_and_return(list(range(12)), pages)
    cache.inc_lock_ref(leaf)

    with pytest.raises(RuntimeError):
        pool.allocate(2)

    assert cache.match_prefix(list(range(12))).matched_pages == pages
    assert cache.num_cached_pages == 3
    cache.dec_lock_ref(leaf)


def test_finish_donates_new_pages_without_freeing_borrowed_cache_pages():
    engine = _engine()
    scheduler = _scheduler(engine)
    cache = engine.radix_cache
    assert cache is not None

    prefix_pages = engine.pool.allocate(2)
    cache.insert_and_return(list(range(8)), prefix_pages)

    req = _request("extend", list(range(12)))
    scheduler._prepare_paged_prefill_for_admission(req)
    state = engine._state(req)
    fresh_page = state.page_table[2]

    _finish_with_materialized_prompt(scheduler, engine, req)

    assert cache.num_cached_pages == 3
    assert fresh_page not in engine.pool._free
    assert all(page not in engine.pool._free for page in prefix_pages)
    assert cache.root.ref_count == 0


def test_disable_radix_cache_flag_and_stats(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["miniengine", "--model", "dummy", "--disable-radix-cache"]
    )
    assert parse_args().disable_radix_cache is True

    engine = _engine(cache_enabled=False)
    req = _request("disabled", list(range(8)))

    old_engine = server.engine
    try:
        server.engine = engine
        assert engine.radix_cache is None
        assert req.cache_hit_tokens == 0
        assert asyncio.run(server.cache_stats()) == {"enabled": False}
    finally:
        server.engine = old_engine
