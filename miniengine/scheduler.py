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
    FCFS scheduler with continuous (iteration-level) batching.

    Each step:
      1. Admits any waiting requests up to max_running and prefills them
         (prefill stays unbatched per the milestone spec).
      2. Runs ONE batched decode across all currently-running requests
         (including those just prefilled in this same step).
      3. Retires any requests that finished, freeing their slots for the
         next step's admit phase.

    Unlike static batching, the batch composition can change every
    iteration: a request that finishes mid-batch frees its slot
    immediately, and a new request from the waiting queue joins the
    batch on the very next step rather than waiting for the entire
    batch to drain.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down

    Internal (called from the scheduler thread):
        step()             — one full scheduling iteration
    """

    def __init__(self, engine: Engine, max_running: int = 16):
        self.engine = engine
        self.max_running = max_running

        # Queues
        self.waiting: deque[Request] = deque()
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
            has_work = bool(self.waiting) or bool(self.running)
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
        One scheduling iteration (continuous / iteration-level batching).

        Phases:
          1. Admit any waiting requests up to max_running, then prefill
             those new admits one by one.
          2. Batched-decode every currently-running request (including
             the just-prefilled ones) in a single forward pass.
          3. Retire any finished requests; their slots free up for the
             next step's admit phase.

        Returns the list of requests that finished in this step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill new requests ───────────────────────
        new_admits: list[Request] = []
        with self._lock:
            while self.waiting and len(self.running) < self.max_running:
                req = self.waiting.popleft()
                self.running.append(req)
                new_admits.append(req)

        for req in new_admits:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        # Drop any new admit that already finished on its first sampled
        # token (immediate stop token / max_new_tokens == 1) so we don't
        # waste a decode slot on it.
        if new_admits:
            self.running, _ = self._retire_finished(self.running, finished)

        if not self.running:
            self.engine.end_batch()
            return finished

        logger.debug(
            "step: batch=%d (new=%d), waiting=%d",
            len(self.running), len(new_admits), len(self.waiting),
        )

        # ── Phase 2: keep a persistent batched KV across decode steps.
        # Rebuild only when the running set changes; otherwise reuse it.
        try:
            if not self.engine.batch_matches(self.running):
                self.engine.rebuild_batch(self.running)
            tokens = self.engine.batched_decode(self.running)
        except Exception:
            self.engine.end_batch()
            if self.running:
                logger.warning(
                    "Failing %d requests still in-flight after step error",
                    len(self.running),
                )
                for req in self.running:
                    self._finish_request(req, finished)
                self.running = []
            raise

        for req, tok in zip(self.running, tokens):
            req.output_ids.append(tok)
            self._stream_token(req, tok)

        # ── Phase 3: retire finished requests ───────────────────────────
        self.running, keep_indices = self._retire_finished(self.running, finished)
        if self.running:
            self.engine.drop_batch_rows(keep_indices)
        else:
            self.engine.end_batch()

        return finished

    # ── Helpers ─────────────────────────────────────────────────────────

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
        req.token_queue.put(TokenOutput(token_id=token_id, token_text=text, finished=False))

    def _retire_finished(
        self, batch: list[Request], finished: list[Request]
    ) -> tuple[list[Request], list[int]]:
        """
        Split `batch` into still-running and finished; retire the latter.

        Returns the surviving requests and their indices into `batch`.
        """
        still_running: list[Request] = []
        keep_indices: list[int] = []
        for i, req in enumerate(batch):
            if self._check_finished(req, req.output_ids[-1]):
                self._finish_request(req, finished)
            else:
                still_running.append(req)
                keep_indices.append(i)
        return still_running, keep_indices

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        req.kv_cache = None  # release GPU memory
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
