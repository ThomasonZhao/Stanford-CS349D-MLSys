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
    FCFS scheduler with two modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.

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
        retraction_enabled: bool = True,
    ):
        self.engine = engine
        self.max_running = max_running
        self.mode = mode
        self.prefill_chunk_size = prefill_chunk_size
        self.retraction_enabled = retraction_enabled

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.prefilling: list[Request] = []

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
            has_work = bool(self.waiting) or bool(self.running) or bool(self.prefilling)
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
        if self.prefill_chunk_size > 0:
            return self._step_paged_chunked()
        return self._step_paged_unchunked()

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

        # ── Phase 1: admit + batched paged prefill ──────────────────────
        to_prefill_info, _ = self._collect_paged_admissions(len(self.running))
        to_prefill = [item[0] for item in to_prefill_info]

        if to_prefill:
            for req, prefix_pages, prefix_len, cache_node in to_prefill_info:
                req.status = RequestStatus.RUNNING
                self.engine.init_paged_prefill_state(
                    req,
                    prefix_pages=prefix_pages,
                    prefix_len=prefix_len,
                    cache_node=cache_node,
                )
            for req, token_id in zip(
                to_prefill, self.engine.paged_batched_prefill(to_prefill)
            ):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)

        # ── Phase 2: paged batched decode ───────────────────────────────
        if self.running:
            self._ensure_decode_capacity()
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
        Paged-mode step with chunked prefill.  Same as ``_step_paged_unchunked``
        """
        finished = []
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"

        # ── Phase 1: admit + chunked paged prefill ──────────────────────
        to_admit_info, admitted_pages = self._collect_paged_admissions(
            len(self.running) + len(self.prefilling)
        )
        to_admit = [item[0] for item in to_admit_info]

        logger.info(
            "Admitting %d new requests (%d pages) with chunked prefill (chunk size=%d), currently prefilling %d requests",
            len(to_admit),
            admitted_pages,
            self.prefill_chunk_size,
            len(self.prefilling),
        )

        for req, prefix_pages, prefix_len, cache_node in to_admit_info:
            req.status = RequestStatus.RUNNING
            self.engine.init_paged_prefill_state(
                req,
                prefix_pages=prefix_pages,
                prefix_len=prefix_len,
                cache_node=cache_node,
            )
            self.prefilling.append(req)

        # ── Phase 2: chunked prefill  ────────────────────────────────────
        if self.prefilling:
            still_prefilling: list[Request] = []
            for req, token_id in zip(
                self.prefilling,
                self.engine.paged_batched_chunked_prefill(
                    self.prefilling, self.prefill_chunk_size
                ),
            ):
                if token_id is None:
                    still_prefilling.append(req)
                    continue

                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.running.append(req)
            self.prefilling = still_prefilling
            logger.info(
                "Chunked prefill step: admitted %d new requests (%d pages), still prefilling %d requests",
                len(to_admit),
                admitted_pages,
                len(still_prefilling),
            )

        # ── Phase 3: paged batched decode ───────────────────────────────
        if self.running:
            self._ensure_decode_capacity()
        if self.running:
            token_ids = self.engine.paged_batched_decode(self.running)
            still_running = []
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

    def _collect_paged_admissions(
        self, active_count: int
    ) -> tuple[list[tuple[Request, list[int], int, object | None]], int]:
        """Pop waiting requests that fit, locking any borrowed radix pages."""
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"
        cache = getattr(self.engine, "radix_cache", None)

        admitted: list[tuple[Request, list[int], int, object | None]] = []
        admitted_pages = 0
        with self._lock:
            while self.waiting and active_count + len(admitted) < self.max_running:
                req = self.waiting[0]
                prefix_pages, prefix_len, cache_node = self.engine.lookup_radix_cache(
                    req
                )
                if cache is not None and cache_node is not None:
                    cache.inc_lock_ref(cache_node)

                needed_pages = max(
                    0,
                    pool.pages_needed(len(self.engine.prefill_token_ids(req)))
                    - len(prefix_pages),
                )
                available_pages = (
                    pool.num_free + getattr(pool, "num_evictable", 0) - admitted_pages
                )
                if needed_pages > available_pages:
                    if cache is not None and cache_node is not None:
                        cache.dec_lock_ref(cache_node)
                    break

                admitted_pages += needed_pages
                admitted.append(
                    (self.waiting.popleft(), prefix_pages, prefix_len, cache_node)
                )

        return admitted, admitted_pages

    def _ensure_decode_capacity(self) -> None:
        """Retract active requests until the next decode allocation fits."""
        if not self.retraction_enabled:
            return
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"

        while self.running:
            needed_pages = self.engine.paged_decode_page_need(self.running)
            available_pages = pool.num_free + getattr(pool, "num_evictable", 0)
            if needed_pages <= available_pages:
                return

            victim = self._retract_one_request()
            if victim is None:
                return

            logger.warning(
                "Retracted request %s for decode capacity "
                "(needed_pages=%d, available_pages=%d, output_len=%d)",
                victim.request_id,
                needed_pages,
                available_pages,
                victim.num_output_tokens,
            )

    def _retract_one_request(self) -> Request | None:
        """Free one active request's paged KV and put it back in waiting."""
        if self.prefilling:
            victim = self._pick_retraction_victim(self.prefilling)
            self.prefilling.remove(victim)
        elif self.running:
            victim = self._pick_retraction_victim(self.running)
            self.running.remove(victim)
        else:
            return None

        self.engine.retract_paged_state(victim)
        victim.status = RequestStatus.WAITING
        with self._lock:
            self.waiting.append(victim)
        return victim

    def _pick_retraction_victim(self, candidates: list[Request]) -> Request:
        """Pick largest remaining decode work, breaking ties by youngest."""
        return max(
            candidates,
            key=lambda req: (
                req.sampling_params.max_new_tokens - len(req.output_ids),
                req.arrival_time,
            ),
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
            self.engine.free_paged_state(req)
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
