import logging
import math

import torch

from miniengine.core import Request, RequestStatus, SamplingParams
from miniengine.engine import Engine, PagedDecodeOOM, _PagedState
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.radix_cache import RadixCache
from miniengine.scheduler import Scheduler, _select_retraction_victim


class FakePool:
    def __init__(self, page_size: int = 4, num_pages: int = 32):
        self.page_size = page_size
        self._free = list(range(1, num_pages + 1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def pages_needed(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        return math.ceil(seq_len / self.page_size)

    def allocate(self, num_pages: int) -> list[int]:
        if len(self._free) < num_pages:
            raise RuntimeError(
                f"KV pool exhausted: requested {num_pages}, have {len(self._free)}"
            )
        out = self._free[:num_pages]
        self._free = self._free[num_pages:]
        return out

    def free(self, page_indices: list[int]) -> None:
        self._free.extend(page_indices)


class FakeRetractionEngine:
    def __init__(self):
        self.pool = FakePool()
        self.radix_cache = None
        self._paged_state: dict[str, _PagedState] = {}
        self._paged_replay_prefixes: dict[str, list[int]] = {}
        self.prefill_prefix_lens: dict[str, list[int]] = {}
        self.decode_calls: list[list[str]] = []
        self._raised_once = False

    _state = Engine._state
    paged_prefill_input_ids = Engine.paged_prefill_input_ids
    _clear_paged_replay_prefix = Engine._clear_paged_replay_prefix
    prepare_paged_prefill = Engine.prepare_paged_prefill
    retract_paged_state = Engine.retract_paged_state
    free_paged_state = Engine.free_paged_state

    def paged_batched_prefill(self, requests: list[Request]) -> list[int]:
        out = []
        for req in requests:
            prefix_ids = self.paged_prefill_input_ids(req)
            self.prefill_prefix_lens.setdefault(req.request_id, []).append(
                len(prefix_ids)
            )
            state = self._state(req)
            if not state.page_table:
                self.prepare_paged_prefill(req)
            state.cache_seq_len = len(prefix_ids)
            state.prefill_offset = len(prefix_ids)
            state.prefill_target = len(prefix_ids)
            self._clear_paged_replay_prefix(req)
            out.append(1000 + len(prefix_ids))
        return out

    def paged_batched_decode(self, requests: list[Request]) -> list[int]:
        self.decode_calls.append([req.request_id for req in requests])
        if not self._raised_once and any(
            req.request_id == "victim" for req in requests
        ):
            self._raised_once = True
            victim = next(req for req in requests if req.request_id == "victim")
            state = self._state(victim)
            raise PagedDecodeOOM(
                request_id="victim",
                requested_pages=1,
                cache_seq_len=state.cache_seq_len,
                cause=RuntimeError("KV pool exhausted"),
            )
        return [2000 + req.num_output_tokens for req in requests]

    def decode_token(self, token_id: int) -> str:
        return str(token_id)

    def is_stop_token(self, token_id: int) -> bool:
        return False


def _request(
    request_id: str,
    prompt_len: int = 4,
    max_new_tokens: int = 8,
    output_len: int = 0,
    arrival_time: float = 0.0,
) -> Request:
    req = Request(
        request_id=request_id,
        input_ids=list(range(prompt_len)),
        output_ids=list(range(100, 100 + output_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, temperature=0),
        arrival_time=arrival_time,
    )
    req.status = RequestStatus.RUNNING
    return req


def _pool(page_size: int = 4, num_pages: int = 12) -> KVMemoryPool:
    return KVMemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def test_select_retraction_victim_prefers_decode_oom_request():
    older = _request("older", output_len=2, arrival_time=1.0)
    preferred = _request("preferred", output_len=7, arrival_time=2.0)
    states = {
        "older": _PagedState(page_table=[1, 2, 3, 4]),
        "preferred": _PagedState(page_table=[5]),
    }

    assert (
        _select_retraction_victim(
            [older, preferred], states, preferred_request_id="preferred"
        )
        is preferred
    )


def test_select_retraction_victim_frees_most_pages_then_remaining_budget():
    small = _request("small", max_new_tokens=16, output_len=2, arrival_time=1.0)
    low_budget = _request("low-budget", max_new_tokens=4, output_len=3, arrival_time=2.0)
    high_budget = _request(
        "high-budget", max_new_tokens=16, output_len=3, arrival_time=3.0
    )
    states = {
        "small": _PagedState(page_table=[1]),
        "low-budget": _PagedState(page_table=[2, 3, 4]),
        "high-budget": _PagedState(page_table=[5, 6, 7]),
    }

    assert (
        _select_retraction_victim([small, low_budget, high_budget], states)
        is high_budget
    )
    assert (
        _select_retraction_victim(
            [small, low_budget, high_budget],
            states,
            exclude_request_ids={"high-budget"},
        )
        is low_budget
    )


def test_retract_paged_state_frees_owned_pages_and_keeps_cache_pages_resident():
    engine = Engine.__new__(Engine)
    engine.pool = _pool()
    engine.radix_cache = RadixCache(engine.pool)
    engine._paged_state = {}
    engine._paged_replay_prefixes = {}

    req = _request("victim", prompt_len=4, output_len=2)
    cached_page = engine.pool.allocate(1)
    leaf, _ = engine.radix_cache.insert_and_return(req.input_ids, cached_page)
    engine.radix_cache.inc_lock_ref(leaf)
    owned_pages = engine.pool.allocate(2)

    state = engine._state(req)
    state.page_table = cached_page + owned_pages
    state.cache_hit_pages = list(cached_page)
    state.cache_node = leaf
    state.cache_seq_len = 9
    state.prefill_offset = 4
    state.prefill_target = 12

    assert engine.retract_paged_state(req) == 2

    assert all(page in engine.pool._free for page in owned_pages)
    assert cached_page[0] not in engine.pool._free
    assert engine.radix_cache.num_cached_pages == 1
    assert engine.radix_cache.root.ref_count == 0
    assert state.page_table == []
    assert state.cache_seq_len == 0
    assert state.prefill_offset == 0
    assert state.prefill_target == 0
    assert state.cache_node is None
    assert state.cache_hit_pages == []
    assert engine.paged_prefill_input_ids(req) == req.input_ids + req.output_ids


def test_decode_oom_retracts_and_reprefills_visible_prefix(caplog):
    engine = FakeRetractionEngine()
    scheduler = Scheduler(engine, max_running=2, mode="paged", prefill_chunk_size=0)
    victim = _request("victim", prompt_len=4, max_new_tokens=2)
    survivor = _request("survivor", prompt_len=4, max_new_tokens=2)
    victim.status = RequestStatus.WAITING
    survivor.status = RequestStatus.WAITING

    scheduler.add_request(victim)
    scheduler.add_request(survivor)

    with caplog.at_level(logging.INFO, logger="miniengine.scheduler"):
        first_step = scheduler.step()
        second_step = scheduler.step()

    assert first_step == [survivor]
    assert second_step == [victim]
    assert victim.output_ids == [1004, 1005]
    assert survivor.output_ids == [1004, 2001]
    assert engine.prefill_prefix_lens["victim"] == [4, 5]
    assert engine.decode_calls == [["victim", "survivor"], ["survivor"]]
    assert not scheduler.waiting
    assert not scheduler.running
    assert any(
        "[retract] victim=victim pages_freed=1" in r.message
        for r in caplog.records
    )
