"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class Scheduler:
    """
    FCFS scheduler with three modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.
      paged    : KV-pool-backed scheduling with packed prefill and paged
                 decode.  With ``prefill_chunk_size > 0``, admitted
                 requests sit in ``prefilling`` until their prompt chunks
                 have all been processed.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down
    """

    def __init__(
        self,
        engine: Engine,
        max_running: int = 16,
        mode: str = "paged",
        prefill_chunk_size: int = 0,
    ):
        if prefill_chunk_size < 0:
            raise ValueError("prefill_chunk_size must be non-negative")
        self.engine = engine
        self.max_running = max_running
        self.mode = mode
        self.prefill_chunk_size = prefill_chunk_size

        # Queues
        self.waiting: deque[Request] = deque()
        self.prefilling: list[Request] = []
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = (
                bool(self.waiting) or bool(self.prefilling) or bool(self.running)
            )
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration.  Behaviour depends on self.mode.

        Returns list of requests that finished in this step.
        """
        if self.mode == "baseline":
            return self._step_baseline()
        if self.mode == "paged":
            return self._step_paged()
        return self._step_batched()

    def _step_baseline(self) -> list[Request]:
        """One request to completion per step. Maximally naive."""
        finished: list[Request] = []

        with self._lock:
            if not self.waiting:
                return finished
            req = self.waiting.popleft()

        req.status = RequestStatus.RUNNING
        token_id = self.engine.prefill(req)
        req.output_ids.append(token_id)
        self._stream_token(req, token_id)

        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        self._finish_request(req, finished)
        return finished

    def _step_batched(self) -> list[Request]:
        """
        Iteration-level batched step:
          Phase 1 — admit waiting requests and prefill them (per-request).
          Phase 2 — batched decode: one token for every running request.
        Newly prefilled requests join the decode batch in the same step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill ────────────────────────────────────
        with self._lock:
            to_prefill: list[Request] = []
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                to_prefill.append(self.waiting.popleft())

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        # ── Phase 2: batched decode ─────────────────────────────────────
        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_paged(self) -> list[Request]:
        if self.prefill_chunk_size == 0:
            return self._step_paged_unchunked()
        return self._step_paged_chunked()

    def _step_paged_unchunked(self) -> list[Request]:
        """
        Paged-mode step — same shape as ``_step_batched`` but:

          - admission is gated by KV-pool page availability,
          - prefill is varlen-batched (one packed forward for all admissions),
          - decode is paged + flash_attn (with optional CUDA graph),
          - finishing a request frees its KV pages via
            ``engine.free_paged_state``.
        """
        finished: list[Request] = []
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"
        cache = getattr(self.engine, "radix_cache", None)

        # ── Phase 1: admit + batched paged prefill ──────────────────────
        to_prefill: list[Request] = []
        if cache is None:
            with self._lock:
                while (
                    self.waiting
                    and len(self.running) + len(to_prefill) < self.max_running
                ):
                    req = self.waiting[0]
                    if pool.num_free < pool.pages_needed(len(req.input_ids)):
                        break  # can't fit; wait for pages to free
                    to_prefill.append(self.waiting.popleft())
        else:
            with self._lock:
                while (
                    self.waiting
                    and len(self.running) + len(to_prefill) < self.max_running
                ):
                    req = self.waiting.popleft()
                    try:
                        self._prepare_paged_prefill_for_admission(req)
                    except RuntimeError:
                        self.waiting.appendleft(req)
                        break
                    to_prefill.append(req)

        if to_prefill:
            for req in to_prefill:
                req.status = RequestStatus.RUNNING
            for req, token_id in zip(
                to_prefill, self.engine.paged_batched_prefill(to_prefill)
            ):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)
            self._log_cache_prefill_batch(to_prefill)

        # ── Phase 2: paged batched decode ───────────────────────────────
        if self.running:
            token_ids = self.engine.paged_batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_paged_chunked(self) -> list[Request]:
        """
        Paged-mode step with chunked prefill enabled.

        Admission still reserves all prompt KV pages up front.  Each step
        advances every prefilling request by one prompt chunk in a single
        packed forward, then runs decode for the active running set.
        Interleaving decode between chunks is future work; this first cut
        keeps the phase order simple and deterministic.
        """
        finished: list[Request] = []
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"
        cache = getattr(self.engine, "radix_cache", None)

        # ── Phase 1: admit requests whose full prompt KV fits ──────────
        to_admit: list[Request] = []
        if cache is None:
            with self._lock:
                planned_pages = 0
                while (
                    self.waiting
                    and len(self.running) + len(self.prefilling) + len(to_admit)
                    < self.max_running
                ):
                    req = self.waiting[0]
                    need = pool.pages_needed(len(req.input_ids))
                    if pool.num_free - planned_pages < need:
                        break  # head-of-line request waits for full prompt capacity
                    planned_pages += need
                    to_admit.append(self.waiting.popleft())
        else:
            with self._lock:
                while (
                    self.waiting
                    and len(self.running) + len(self.prefilling) + len(to_admit)
                    < self.max_running
                ):
                    req = self.waiting.popleft()
                    try:
                        self._prepare_paged_prefill_for_admission(req)
                    except RuntimeError:
                        self.waiting.appendleft(req)
                        break
                    to_admit.append(req)

        for req in to_admit:
            req.status = RequestStatus.RUNNING
            if cache is None:
                self.engine.prepare_paged_prefill(req)
            self.prefilling.append(req)

        # ── Phase 2: one packed chunk for all prefilling requests ───────
        if self.prefilling:
            prefill_batch = list(self.prefilling)
            token_ids = self.engine.paged_chunked_prefill(
                self.prefilling, self.prefill_chunk_size
            )
            still_prefilling: list[Request] = []
            completed = 0
            for req, token_id in zip(self.prefilling, token_ids):
                if token_id is None:
                    still_prefilling.append(req)
                    continue
                completed += 1
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)
            self.prefilling = still_prefilling
            logger.info(
                "Chunked prefill step  batch=%d  chunk_size=%d  "
                "completed=%d  remaining=%d",
                len(token_ids),
                self.prefill_chunk_size,
                completed,
                len(self.prefilling),
            )
            self._log_cache_prefill_batch(prefill_batch)

        # ── Phase 3: paged batched decode ───────────────────────────────
        if self.running:
            token_ids = self.engine.paged_batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

    def _prepare_paged_prefill_for_admission(self, req: Request) -> None:
        """Reserve paged KV for a newly admitted request, applying cache hits."""
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"
        cache = getattr(self.engine, "radix_cache", None)
        if cache is None:
            self.engine.prepare_paged_prefill(req)
            return

        state = self.engine._state(req)
        if state.page_table:
            raise RuntimeError(f"request {req.request_id} already has paged KV state")

        result = cache.match_prefix(req.input_ids)

        # The cache stores K/V, not the logits needed to sample the first
        # generated token, so prefill must recompute at least one prompt token.
        max_hit_tokens = max(0, (len(req.input_ids) - 1) // pool.page_size)
        max_hit_tokens *= pool.page_size
        hit_tokens = min(result.matched_tokens, max_hit_tokens)
        if hit_tokens < result.matched_tokens:
            cache.metrics.total_hit_tokens -= result.matched_tokens - hit_tokens
        cached_pages = result.matched_pages[: hit_tokens // pool.page_size]

        req.cache_hit_tokens = hit_tokens
        state.cache_hit_pages = list(cached_pages)
        state.cache_node = result.last_node if hit_tokens > 0 else None
        state.cache_seq_len = hit_tokens
        state.prefill_offset = hit_tokens
        state.prefill_target = len(req.input_ids)

        if state.cache_node is not None:
            cache.inc_lock_ref(state.cache_node)

        fresh_needed = pool.pages_needed(len(req.input_ids)) - len(cached_pages)
        try:
            fresh_pages = pool.allocate(fresh_needed)
        except RuntimeError:
            if state.cache_node is not None:
                cache.dec_lock_ref(state.cache_node)
            state.cache_node = None
            state.cache_hit_pages = []
            state.cache_seq_len = 0
            state.prefill_offset = 0
            state.prefill_target = 0
            raise

        state.page_table = list(cached_pages) + fresh_pages

    def _log_cache_prefill_batch(self, batch: list[Request]) -> None:
        cache = getattr(self.engine, "radix_cache", None)
        if cache is None or not batch:
            return
        pool = self.engine.pool
        assert pool is not None
        query_tokens = sum(len(req.input_ids) for req in batch)
        hit_tokens = sum(req.cache_hit_tokens for req in batch)
        pct = (100.0 * hit_tokens / query_tokens) if query_tokens else 0.0
        states = getattr(self.engine, "_paged_state", {})
        running_pages = sum(len(s.page_table) for s in states.values())
        logger.info(
            "[cache] batch=%d hit_tokens=%d/%d (%.1f%%) "
            "running_pages=%d cache_pages=%d evictable=%d free=%d",
            len(batch),
            hit_tokens,
            query_tokens,
            pct,
            running_pages,
            cache.num_cached_pages,
            cache.num_evictable_pages(),
            pool.num_free,
        )

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(
            TokenOutput(token_id=token_id, token_text=text, finished=False)
        )

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        req.kv_cache = None  # release GPU memory (baseline / batched modes)
        if self.mode == "paged":
            self._finish_paged_request(req)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )

    def _finish_paged_request(self, req: Request) -> None:
        cache = getattr(self.engine, "radix_cache", None)
        if cache is None:
            self.engine.free_paged_state(req)
            return

        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"
        state = self.engine.pop_paged_state(req)
        if state is None:
            return

        borrowed = set(state.cache_hit_pages)
        try:
            # The final sampled token has not been fed back through decode yet,
            # so its KV is not materialized. Cache prompt + generated tokens
            # only up to the engine's realized cache length.
            materialized_tokens = req.input_ids + req.output_ids[:-1]
            materialized_len = min(len(materialized_tokens), state.cache_seq_len)
            page_aligned = (materialized_len // pool.page_size) * pool.page_size

            if page_aligned > 0:
                page_count = page_aligned // pool.page_size
                cached_tokens = materialized_tokens[:page_aligned]
                cached_pages = state.page_table[:page_count]
                _, redundant = cache.insert_and_return(cached_tokens, cached_pages)
                pool.free([p for p in redundant if p not in borrowed])
                pool.free(
                    [p for p in state.page_table[page_count:] if p not in borrowed]
                )
            else:
                pool.free([p for p in state.page_table if p not in borrowed])
        finally:
            cache.dec_lock_ref(state.cache_node)
            state.cache_node = None
