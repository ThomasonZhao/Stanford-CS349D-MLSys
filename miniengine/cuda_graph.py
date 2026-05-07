"""Manual CUDA-graph capture for paged decode (Part C extra credit).

Captures one CUDA graph per bucket batch size against the engine's
pre-allocated decode input buffers. At runtime, the actual batch is
padded up to the nearest bucket and the graph is replayed.

Padded rows write their K/V into the pool's scratch page so the
in-place flash-attn write cannot corrupt any active request.

Sampling stays outside the captured region. The runner returns the
logits slice corresponding to the actual batch size, and the engine
samples in eager mode.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from miniengine.model import PagedContext

if TYPE_CHECKING:
    from miniengine.engine import Engine

logger = logging.getLogger(__name__)


class CudaGraphRunner:
    """Capture-and-replay decode forwards bucketed by batch size."""

    def __init__(
        self,
        engine: "Engine",
        bucket_sizes: list[int],
        max_pages_per_seq: int,
    ) -> None:
        if not bucket_sizes:
            raise ValueError("bucket_sizes must be non-empty")
        for b in bucket_sizes:
            if b <= 0:
                raise ValueError("bucket sizes must be positive")
        self.engine = engine
        self.bucket_sizes = sorted(set(bucket_sizes))
        self.max_pages_per_seq = max_pages_per_seq
        self.max_bucket = max(self.bucket_sizes)

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits_buffers: dict[int, torch.Tensor] = {}

        pool = engine.kv_pool
        if pool is None:
            raise RuntimeError("CudaGraphRunner requires a paged KV pool")
        if pool.scratch_page_idx is None:
            raise RuntimeError(
                "CudaGraphRunner requires the KV pool to reserve a scratch page"
            )
        self._scratch_page = pool.scratch_page_idx

    def _make_paged_ctx(self, batch_size: int) -> PagedContext:
        pool = self.engine.kv_pool
        assert pool is not None
        block_table = self.engine._dec_block_table[
            :batch_size, : self.max_pages_per_seq
        ]
        cache_seqlens = self.engine._dec_cache_seqlens[:batch_size]
        return PagedContext(
            k_cache_pool=pool.k_caches,
            v_cache_pool=pool.v_caches,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            is_prefill=False,
        )

    def _set_padding_safe(self, batch_size: int, bucket: int) -> None:
        if batch_size >= bucket:
            return
        pad_slice = slice(batch_size, bucket)
        # Padded rows must point block_table at the scratch page so the
        # flash-attn in-place K/V write lands somewhere safe, and the
        # attention compute on those rows is valid (just discarded).
        self.engine._dec_input_ids[pad_slice, 0] = 0
        self.engine._dec_position_ids[pad_slice, 0] = 0
        self.engine._dec_cache_seqlens[pad_slice] = 0
        self.engine._dec_block_table[pad_slice, : self.max_pages_per_seq] = (
            self._scratch_page
        )

    def _seed_capture_inputs(self, bucket: int) -> None:
        # Fill capture-time inputs with a valid configuration: every row
        # treated as if it were a real request whose first decode step
        # writes at position 0 of the scratch page. Safe and shape-stable.
        self.engine._dec_input_ids[:bucket, 0] = 0
        self.engine._dec_position_ids[:bucket, 0] = 0
        self.engine._dec_cache_seqlens[:bucket] = 0
        self.engine._dec_block_table[:bucket, : self.max_pages_per_seq] = (
            self._scratch_page
        )

    @torch.inference_mode()
    def capture(self, warmup_iters: int = 3) -> None:
        """Capture a graph for every configured bucket size.

        Must be called after the model is loaded and decode buffers are
        allocated. The engine's `max_running` buffer capacity must be at
        least `max(bucket_sizes)`.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs require a CUDA device")
        if self.engine._dec_input_ids is None:
            raise RuntimeError("decode buffers must be allocated before capture")
        engine = self.engine
        if engine._dec_buffer_capacity < self.max_bucket:
            raise RuntimeError(
                "decode buffer capacity "
                f"({engine._dec_buffer_capacity}) is smaller than the "
                f"largest CUDA-graph bucket ({self.max_bucket})"
            )
        if engine._dec_buffer_pages_per_seq < self.max_pages_per_seq:
            raise RuntimeError(
                "decode buffer pages_per_seq "
                f"({engine._dec_buffer_pages_per_seq}) is smaller than the "
                f"runner's max_pages_per_seq ({self.max_pages_per_seq})"
            )

        for bucket in self.bucket_sizes:
            self._seed_capture_inputs(bucket)

            input_ids = engine._dec_input_ids[:bucket]
            position_ids = engine._dec_position_ids[:bucket]
            paged_ctx = self._make_paged_ctx(bucket)

            # Warmup so dynamo / cublas / flash-attn settle before capture.
            for _ in range(warmup_iters):
                engine.model(
                    input_ids,
                    position_ids,
                    kv_caches=None,
                    paged_ctx=paged_ctx,
                )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, _ = engine.model(
                    input_ids,
                    position_ids,
                    kv_caches=None,
                    paged_ctx=paged_ctx,
                )

            self.graphs[bucket] = graph
            self.logits_buffers[bucket] = logits
            logger.info(
                "captured CUDA graph bucket=%d pages_per_seq=%d",
                bucket,
                self.max_pages_per_seq,
            )

    def _pick_bucket(self, batch_size: int) -> int:
        for b in self.bucket_sizes:
            if b >= batch_size:
                return b
        raise RuntimeError(
            f"batch_size {batch_size} exceeds largest CUDA-graph bucket "
            f"{self.max_bucket}; capture more buckets or fall back to eager"
        )

    @torch.inference_mode()
    def replay(self, batch_size: int) -> torch.Tensor:
        """Replay the smallest bucket >= batch_size; return logits[:batch_size]."""
        bucket = self._pick_bucket(batch_size)
        self._set_padding_safe(batch_size, bucket)
        self.graphs[bucket].replay()
        # Sampling reads logits in eager; final shape (batch_size, 1, vocab).
        return self.logits_buffers[bucket][:batch_size]
