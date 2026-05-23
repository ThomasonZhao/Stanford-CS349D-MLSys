import asyncio
import logging
import sys
import threading
import time

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


def _install_fake_scheduler_methods(engine) -> None:
    def paged_batched_prefill(requests: list[Request]) -> list[int]:
        out = []
        for req in requests:
            state = engine._state(req)
            state.cache_seq_len = len(req.input_ids)
            state.prefill_offset = len(req.input_ids)
            state.prefill_target = len(req.input_ids)
            out.append(7000 + len(req.input_ids))
        return out

    def paged_chunked_prefill(
        requests: list[Request], chunk_size: int
    ) -> list[int | None]:
        out: list[int | None] = []
        for req in requests:
            state = engine._state(req)
            end = min(state.prefill_offset + chunk_size, state.prefill_target)
            state.cache_seq_len = end
            state.prefill_offset = end
            out.append(7000 + len(req.input_ids) if state.is_prefill_done else None)
        return out

    engine.paged_batched_prefill = paged_batched_prefill
    engine.paged_chunked_prefill = paged_chunked_prefill
    engine.paged_batched_decode = lambda requests: [8000 for _ in requests]
    engine.decode_token = lambda token_id: str(token_id)
    engine.is_stop_token = lambda token_id: False


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


def test_identical_page_aligned_prompt_caps_hit_and_metrics():
    engine = _engine()
    scheduler = _scheduler(engine)
    tokens = list(range(12))

    first = _request("first", tokens)
    scheduler._prepare_paged_prefill_for_admission(first)
    _finish_with_materialized_prompt(scheduler, engine, first)

    second = _request("second", tokens)
    scheduler._prepare_paged_prefill_for_admission(second)
    state = engine._state(second)

    assert second.cache_hit_tokens == len(tokens) - engine.pool.page_size
    assert state.prefill_offset == 8
    assert len(state.cache_hit_pages) == 2
    assert engine.radix_cache.metrics.total_hit_tokens == 8


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


def test_unchunked_cache_log_snapshots_pages_before_retirement(caplog):
    engine = _engine()
    _install_fake_scheduler_methods(engine)
    scheduler = Scheduler(engine, max_running=1, mode="paged", prefill_chunk_size=0)
    req = _request("log", list(range(8)), max_new_tokens=1)
    scheduler.add_request(req)

    with caplog.at_level(logging.INFO, logger="miniengine.scheduler"):
        assert scheduler.step() == [req]

    cache_logs = [
        record.message for record in caplog.records if "[cache]" in record.message
    ]
    assert len(cache_logs) == 1
    assert "running_pages=2" in cache_logs[0]


def test_scheduler_add_request_and_step_concurrent_smoke():
    engine = _engine(num_pages=128)
    _install_fake_scheduler_methods(engine)
    scheduler = Scheduler(engine, max_running=8, mode="paged", prefill_chunk_size=4)
    producer_count = 4
    requests_per_producer = 8
    total_requests = producer_count * requests_per_producer
    producers_done = threading.Event()
    finished_ids: set[str] = set()
    errors: list[BaseException] = []

    def producer(worker_id: int) -> None:
        try:
            for i in range(requests_per_producer):
                req = _request(
                    f"{worker_id}-{i}",
                    list(range(12)),
                    max_new_tokens=1,
                )
                scheduler.add_request(req)
                time.sleep(0.0001)
        except BaseException as exc:
            errors.append(exc)

    def stepper() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                for req in scheduler.step():
                    finished_ids.add(req.request_id)
            except BaseException as exc:
                errors.append(exc)
                return

            with scheduler._lock:
                waiting_empty = not scheduler.waiting
            if (
                producers_done.is_set()
                and waiting_empty
                and not scheduler.prefilling
                and not scheduler.running
            ):
                return
            time.sleep(0.0001)
        errors.append(TimeoutError("scheduler did not drain before watchdog"))

    threads = [
        threading.Thread(target=producer, args=(worker_id,), daemon=True)
        for worker_id in range(producer_count)
    ]
    step_thread = threading.Thread(target=stepper, daemon=True)
    step_thread.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        if thread.is_alive():
            errors.append(TimeoutError("producer did not finish before watchdog"))
    producers_done.set()
    step_thread.join(timeout=6.0)
    if step_thread.is_alive():
        errors.append(TimeoutError("stepper did not finish before watchdog"))

    assert not errors
    assert len(finished_ids) == total_requests
    assert engine.radix_cache.root.ref_count == 0


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
