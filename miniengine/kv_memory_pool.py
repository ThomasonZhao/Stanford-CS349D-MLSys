"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

This is a SKELETON. Implement the methods below.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout (page-major vs token-major, contiguous K+V vs separate,
shape conventions, etc.) is YOUR design decision — pick something and
document the tradeoffs.
"""

from __future__ import annotations

import torch


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

    __slots__ = (
        "num_pages",
        "page_size",
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "dtype",
        "device",
        "_kv_caches",
        "_free_stack",
    )

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        # Only keep necessary checks here
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Page-major layout keeps one page contiguous for all KV heads/tokens,
        # matching the page-table addressing used by paged attention kernels.
        layout = (num_pages, num_kv_heads, page_size, head_dim)
        self._kv_caches = [
            (
                torch.empty(layout, dtype=dtype, device=device),  # keys
                torch.empty(layout, dtype=dtype, device=device),  # values
            )
            for _ in range(num_layers)
        ]
        self._free_stack: list[int] = list(range(num_pages))

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages == 0:
            return []

        free_stack = self._free_stack
        if num_pages > len(free_stack):
            raise RuntimeError(
                f"Not enough free pages: requested {num_pages}, "
                f"but only {len(free_stack)} available"
            )

        allocated = free_stack[-num_pages:]
        del free_stack[-num_pages:]
        return allocated

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        self._free_stack.extend(page_indices)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {seq_len}")
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free_stack)

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables. The exact shape is up to your
        design — but it must be STABLE: no reallocation, no resizing,
        no swapping out the tensors after construction.
        """
        return self._kv_caches

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget."""
        if bytes_budget <= 0:
            raise ValueError(f"bytes_budget must be positive, got {bytes_budget}")

        dtype_size = torch.empty((), dtype=dtype).element_size()

        bytes_per_page = (
            num_layers * page_size * num_kv_heads * head_dim * dtype_size * 2  # K + V
        )
        num_pages = bytes_budget // bytes_per_page

        return cls(
            num_pages,
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype,
            device,
        )
