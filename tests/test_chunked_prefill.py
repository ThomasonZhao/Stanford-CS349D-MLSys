import io
import logging
import math
from contextlib import contextmanager

import torch

from miniengine.core import Request, RequestStatus, SamplingParams
from miniengine.engine import Engine
from miniengine.scheduler import Scheduler


class FakePool:
    def __init__(self, page_size: int = 32, num_pages: int = 10_000):
        self.page_size = page_size
        self._free = list(range(1, num_pages + 1))
        self.allocations: list[int] = []

    def pages_needed(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        return math.ceil(seq_len / self.page_size)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, num_pages: int) -> list[int]:
        self.allocations.append(num_pages)
        out = self._free[:num_pages]
        self._free = self._free[num_pages:]
        return out

    def free(self, page_indices: list[int]) -> None:
        self._free.extend(page_indices)


class FakeSchedulerEngine:
    def __init__(self, page_size: int = 32):
        self.pool = FakePool(page_size=page_size)
        self.state: dict[str, dict[str, int | list[int]]] = {}
        self.prefill_calls: list[list[str]] = []
        self.chunk_calls: list[list[tuple[str, int, int]]] = []
        self.decode_calls: list[list[str]] = []
        self.freed: list[str] = []

    def prepare_paged_prefill(self, req: Request) -> None:
        pages = self.pool.allocate(self.pool.pages_needed(len(req.input_ids)))
        self.state[req.request_id] = {
            "offset": 0,
            "target": len(req.input_ids),
            "pages": pages,
        }

    def paged_batched_prefill(self, requests: list[Request]) -> list[int]:
        self.prefill_calls.append([req.request_id for req in requests])
        out = []
        for req in requests:
            self.prepare_paged_prefill(req)
            st = self.state[req.request_id]
            st["offset"] = st["target"]
            out.append(self._first_token(req))
        return out

    def paged_chunked_prefill(
        self, requests: list[Request], chunk_size: int
    ) -> list[int | None]:
        call: list[tuple[str, int, int]] = []
        out: list[int | None] = []
        for req in requests:
            st = self.state[req.request_id]
            start = int(st["offset"])
            target = int(st["target"])
            end = min(start + chunk_size, target)
            call.append((req.request_id, start, end))
            st["offset"] = end
            out.append(self._first_token(req) if end >= target else None)
        self.chunk_calls.append(call)
        return out

    def paged_batched_decode(self, requests: list[Request]) -> list[int]:
        self.decode_calls.append([req.request_id for req in requests])
        return [3000 + len(req.output_ids) for req in requests]

    def free_paged_state(self, req: Request) -> None:
        st = self.state.pop(req.request_id, None)
        if st is not None:
            self.pool.free(st["pages"])  # type: ignore[arg-type]
        self.freed.append(req.request_id)

    def decode_token(self, token_id: int) -> str:
        return str(token_id)

    def is_stop_token(self, token_id: int) -> bool:
        return False

    def _first_token(self, req: Request) -> int:
        return 1000 + len(req.input_ids)


class FakeModel:
    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
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
        logits = torch.zeros(
            1,
            logits_indices.numel(),
            self.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row in range(logits_indices.numel()):
            logits[0, row, 7 + row] = 1.0
        return logits, None


def _request(request_id: str, prompt_len: int, max_new_tokens: int = 1) -> Request:
    return Request(
        request_id=request_id,
        input_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, temperature=0),
    )


def _run_to_finish(scheduler: Scheduler, limit: int = 100) -> list[Request]:
    finished: list[Request] = []
    for _ in range(limit):
        finished.extend(scheduler.step())
        if (
            finished
            and not scheduler.waiting
            and not scheduler.prefilling
            and not scheduler.running
        ):
            return finished
    raise AssertionError("scheduler did not finish within step limit")


@contextmanager
def _temporary_attr(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextmanager
def _capture_engine_logs():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("miniengine.engine")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def test_chunk_size_zero_uses_unchunked_paged_path_and_matches_tokens():
    prompt_len = 128

    unchunked_engine = FakeSchedulerEngine()
    unchunked = Scheduler(
        unchunked_engine,
        max_running=1,
        mode="paged",
        prefill_chunk_size=0,
    )
    req_a = _request("a", prompt_len, max_new_tokens=3)
    unchunked.add_request(req_a)
    _run_to_finish(unchunked)

    chunked_engine = FakeSchedulerEngine()
    chunked = Scheduler(
        chunked_engine,
        max_running=1,
        mode="paged",
        prefill_chunk_size=512,
    )
    req_b = _request("b", prompt_len, max_new_tokens=3)
    chunked.add_request(req_b)
    _run_to_finish(chunked)

    assert req_a.output_ids == req_b.output_ids
    assert unchunked_engine.prefill_calls == [["a"]]
    assert unchunked_engine.chunk_calls == []
    assert chunked_engine.prefill_calls == []
    assert chunked_engine.chunk_calls == [[("b", 0, prompt_len)]]


def test_long_prompt_samples_after_exact_number_of_chunks():
    engine = FakeSchedulerEngine()
    scheduler = Scheduler(
        engine,
        max_running=1,
        mode="paged",
        prefill_chunk_size=512,
    )
    req = _request("long", 4096, max_new_tokens=1)
    scheduler.add_request(req)

    for _ in range(7):
        assert scheduler.step() == []
        assert req.output_ids == []
        assert req.status is RequestStatus.RUNNING

    finished = scheduler.step()

    assert finished == [req]
    assert req.output_ids == [5096]
    assert len(engine.chunk_calls) == 8
    assert engine.chunk_calls[0] == [("long", 0, 512)]
    assert engine.chunk_calls[-1] == [("long", 3584, 4096)]


def test_mixed_batch_completes_each_request_on_its_expected_chunk():
    engine = FakeSchedulerEngine()
    scheduler = Scheduler(
        engine,
        max_running=3,
        mode="paged",
        prefill_chunk_size=512,
    )
    requests = [
        _request("short", 128, max_new_tokens=1),
        _request("medium", 513, max_new_tokens=1),
        _request("long", 1024, max_new_tokens=1),
    ]
    for req in requests:
        scheduler.add_request(req)

    finished_at: dict[str, int] = {}
    for step in range(1, 4):
        for req in scheduler.step():
            finished_at[req.request_id] = step

    assert finished_at == {"short": 1, "medium": 2, "long": 2}
    assert engine.pool.allocations[:3] == [
        engine.pool.pages_needed(len(req.input_ids)) for req in requests
    ]
    assert engine.chunk_calls[0] == [
        ("short", 0, 128),
        ("medium", 0, 512),
        ("long", 0, 512),
    ]
    assert engine.chunk_calls[1] == [
        ("medium", 512, 513),
        ("long", 512, 1024),
    ]


def test_engine_chunked_prefill_positions_and_final_logits_only():
    engine = Engine.__new__(Engine)
    engine.device = "cpu"
    engine.pool = FakePool(page_size=4)
    engine.page_size = 4
    engine.max_position = 128
    engine._kv_pool_caches = []
    engine._paged_state = {}
    engine.attention_backend = "flash_attn"
    engine.model = FakeModel()

    req = _request("r", 10, max_new_tokens=1)
    state = engine._state(req)
    state.page_table = [1, 2, 3]
    state.prefill_offset = 0
    state.prefill_target = len(req.input_ids)

    with (
        _temporary_attr(engine, "_fa_kwargs_chunked_prefill", lambda *args: {}),
        _capture_engine_logs() as logs,
    ):
        assert engine.paged_chunked_prefill([req], 4) == [None]
        assert engine.model.calls[-1] == {
            "input_ids": [[0, 1, 2, 3]],
            "position_ids": [[0, 1, 2, 3]],
            "logits_indices": [],
        }

        assert engine.paged_chunked_prefill([req], 4) == [None]
        assert engine.model.calls[-1] == {
            "input_ids": [[4, 5, 6, 7]],
            "position_ids": [[4, 5, 6, 7]],
            "logits_indices": [],
        }

        assert engine.paged_chunked_prefill([req], 4) == [7]
        assert engine.model.calls[-1] == {
            "input_ids": [[8, 9]],
            "position_ids": [[8, 9]],
            "logits_indices": [1],
        }
    assert state.cache_seq_len == 10
    assert state.is_prefill_done
    assert logs.getvalue().splitlines() == [
        "[chunked_prefill] req=r chunk=1/3",
        "[chunked_prefill] req=r chunk=2/3",
        "[chunked_prefill] req=r chunk=3/3",
    ]


def test_engine_chunked_prefill_mixed_final_entries_keep_output_order():
    engine = Engine.__new__(Engine)
    engine.device = "cpu"
    engine.pool = FakePool(page_size=4)
    engine.page_size = 4
    engine.max_position = 128
    engine._kv_pool_caches = []
    engine._paged_state = {}
    engine.attention_backend = "flash_attn"
    engine.model = FakeModel()

    first_chunk_done = _request("first", 2, max_new_tokens=1)
    second_chunk_done = _request("second", 5, max_new_tokens=1)

    state_a = engine._state(first_chunk_done)
    state_a.page_table = [1]
    state_a.prefill_offset = 0
    state_a.prefill_target = len(first_chunk_done.input_ids)

    state_b = engine._state(second_chunk_done)
    state_b.page_table = [2, 3]
    state_b.prefill_offset = 4
    state_b.prefill_target = len(second_chunk_done.input_ids)

    with _temporary_attr(engine, "_fa_kwargs_chunked_prefill", lambda *args: {}):
        token_ids = engine.paged_chunked_prefill(
            [first_chunk_done, second_chunk_done], 4
        )

    assert token_ids == [7, 8]
    assert engine.model.calls[-1] == {
        "input_ids": [[0, 1, 4]],
        "position_ids": [[0, 1, 4]],
        "logits_indices": [1, 2],
    }
    assert state_a.cache_seq_len == 2
    assert state_b.cache_seq_len == 5
