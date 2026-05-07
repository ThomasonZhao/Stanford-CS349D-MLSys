"""Pre-allocated paged KV cache memory pool.

The pool owns a fixed amount of device memory split into token pages.
Each request stores a page table in ``Request.page_table`` and addresses
tokens as ``page_idx * page_size + page_offset``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

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

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        reserve_scratch_page: bool = False,
    ) -> None:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if reserve_scratch_page and num_pages < 2:
            raise ValueError("reserve_scratch_page requires num_pages >= 2")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self._k = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        self._v = [
            torch.empty(shape, dtype=dtype, device=self.device)
            for _ in range(num_layers)
        ]
        # Scratch page for CUDA-graph padded rows: never allocated to a real
        # request, so flash-attn's in-place K/V write into the padded slots
        # cannot corrupt any active request's cache.
        self._scratch_page_idx: int | None = 0 if reserve_scratch_page else None
        first_free = 1 if reserve_scratch_page else 0
        self._free: deque[int] = deque(range(first_free, num_pages))
        self._free_set: set[int] = set(range(first_free, num_pages))

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages < 0:
            raise ValueError("num_pages must be non-negative")
        if num_pages > len(self._free):
            raise RuntimeError(
                f"KV pool OOM: need {num_pages} pages, have {len(self._free)} free"
            )
        pages = [self._free.popleft() for _ in range(num_pages)]
        self._free_set.difference_update(pages)
        return pages

    def free(self, page_indices: Iterable[int]) -> None:
        """Return the listed pages to the free pool."""
        returned = list(page_indices)
        for idx in returned:
            if idx < 0 or idx >= self.num_pages:
                raise ValueError(f"page index out of range: {idx}")
            if idx in self._free_set:
                raise ValueError(f"page is already free: {idx}")
        if len(returned) != len(set(returned)):
            raise ValueError("duplicate page indices in free list")
        self._free.extend(returned)
        self._free_set.update(returned)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def scratch_page_idx(self) -> int | None:
        """Reserved page index for CUDA-graph padded rows, if any."""
        return self._scratch_page_idx

    @property
    def k_caches(self) -> list[torch.Tensor]:
        """Per-layer K cache tensors.

        Shape per layer: ``(num_pages, page_size, num_kv_heads, head_dim)``.
        """
        return self._k

    @property
    def v_caches(self) -> list[torch.Tensor]:
        """Per-layer V cache tensors.

        Shape per layer: ``(num_pages, page_size, num_kv_heads, head_dim)``.
        """
        return self._v

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        Shape per tensor:
            ``(num_pages, page_size, num_kv_heads, head_dim)``

        K and V are stored separately to match flash-attn's paged KV
        cache layout and avoid transposes on the hot path.
        """
        return list(zip(self._k, self._v))

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
        reserve_scratch_page: bool = False,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget."""
        if bytes_budget <= 0:
            raise ValueError("bytes_budget must be positive")

        dtype_size = torch.empty((), dtype=dtype).element_size()
        bytes_per_page = (
            2
            * num_layers
            * page_size
            * num_kv_heads
            * head_dim
            * dtype_size
        )
        if bytes_per_page <= 0:
            raise ValueError("bytes_per_page must be positive")

        num_pages = bytes_budget // bytes_per_page
        if num_pages <= 0:
            raise ValueError(
                "bytes_budget is too small for one KV page "
                f"(budget={bytes_budget}, bytes_per_page={bytes_per_page})"
            )
        return cls(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            reserve_scratch_page=reserve_scratch_page,
        )


def _sanity_check() -> None:
    pool = KVMemoryPool(
        num_pages=3,
        page_size=2,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    assert pool.pages_needed(0) == 0
    assert pool.pages_needed(1) == 1
    assert pool.pages_needed(2) == 1
    assert pool.pages_needed(3) == 2
    first = pool.allocate(2)
    assert first == [0, 1]
    assert pool.num_free == 1
    try:
        pool.allocate(2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected allocation failure")
    pool.free(first)
    assert pool.allocate(3) == [2, 0, 1]
    print("KVMemoryPool sanity check passed")


if __name__ == "__main__":
    _sanity_check()
