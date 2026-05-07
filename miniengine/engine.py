"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths:
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, PagedContext, load_weights
from miniengine.sampler import sample_token, sample_tokens_batched

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        mem_fraction_static: float = 0.85,
        torch_compile: bool = False,
        max_running: int = 16,
        max_decode_pages_per_seq: int = 64,
        cuda_graph: bool = False,
        cuda_graph_batch_sizes: list[int] | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static
        self.torch_compile = torch_compile
        self.max_running = max_running
        self.max_decode_pages_per_seq = max_decode_pages_per_seq
        self.cuda_graph = cuda_graph and mode == "paged"
        self.cuda_graph_batch_sizes = cuda_graph_batch_sizes or []
        self.kv_pool: KVMemoryPool | None = None
        self.cuda_graph_runner = None
        self._dec_buffer_capacity = 0
        self._dec_buffer_pages_per_seq = 0
        self._dec_input_ids: torch.Tensor | None = None
        self._dec_position_ids: torch.Tensor | None = None
        self._dec_cache_seqlens: torch.Tensor | None = None
        self._dec_block_table: torch.Tensor | None = None
        self._dec_input_ids_cpu: torch.Tensor | None = None
        self._dec_position_ids_cpu: torch.Tensor | None = None
        self._dec_cache_seqlens_cpu: torch.Tensor | None = None
        self._dec_block_table_cpu: torch.Tensor | None = None

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()

        if torch_compile:
            logger.info("Enabling torch.compile for paged decode MLPs")
            self.model.enable_torch_compile()

        if mode == "paged":
            bytes_budget = self._compute_kv_budget(mem_fraction_static)
            self.kv_pool = KVMemoryPool.from_budget(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=page_size,
                dtype=dtype,
                device=device,
                bytes_budget=bytes_budget,
                reserve_scratch_page=self.cuda_graph,
            )
            logger.info(
                "Paged KV pool ready  pages=%d  page_size=%d  free=%d  scratch=%s",
                self.kv_pool.num_pages,
                self.kv_pool.page_size,
                self.kv_pool.num_free,
                self.kv_pool.scratch_page_idx,
            )
            graph_capacity = (
                max(max_running, max(self.cuda_graph_batch_sizes))
                if self.cuda_graph
                else max_running
            )
            self._allocate_decode_buffers(
                capacity=graph_capacity,
                pages_per_seq=max_decode_pages_per_seq,
            )
            if self.cuda_graph:
                from miniengine.cuda_graph import CudaGraphRunner

                self.cuda_graph_runner = CudaGraphRunner(
                    engine=self,
                    bucket_sizes=self.cuda_graph_batch_sizes,
                    max_pages_per_seq=max_decode_pages_per_seq,
                )
                self.cuda_graph_runner.capture()
                logger.info(
                    "CUDA graph runner ready  buckets=%s",
                    self.cuda_graph_runner.bucket_sizes,
                )

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    def _compute_kv_budget(self, mem_fraction_static: float) -> int:
        if not 0 < mem_fraction_static <= 1:
            raise ValueError("mem_fraction_static must be in (0, 1]")

        model_param_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(device).total_memory
            safety_margin = 512 * 1024 * 1024
        else:
            total_mem = model_param_bytes + 512 * 1024 * 1024
            safety_margin = 64 * 1024 * 1024

        static_budget = int(mem_fraction_static * total_mem)
        kv_budget = static_budget - model_param_bytes - safety_margin
        if kv_budget <= 0:
            raise RuntimeError(
                "Not enough static memory budget for paged KV cache "
                f"(static_budget={static_budget}, model_params={model_param_bytes}, "
                f"safety_margin={safety_margin})"
            )
        return kv_budget

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def _require_kv_pool(self) -> KVMemoryPool:
        if self.kv_pool is None:
            raise RuntimeError("paged KV pool is not initialized")
        return self.kv_pool

    def _make_paged_context(
        self,
        *,
        is_prefill: bool,
        block_table: torch.Tensor | None,
        cache_seqlens: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> PagedContext:
        pool = self._require_kv_pool()
        return PagedContext(
            k_cache_pool=pool.k_caches,
            v_cache_pool=pool.v_caches,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            is_prefill=is_prefill,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=slot_mapping,
        )

    def _validate_rope_budget(self, request: Request) -> None:
        rotary = self.model.model.rotary_emb
        requested_tokens = (
            request.num_input_tokens + request.sampling_params.max_new_tokens
        )
        if requested_tokens > rotary.max_positions:
            raise ValueError(
                "request length exceeds RoPE cache cap "
                f"(prompt={request.num_input_tokens}, "
                f"max_new_tokens={request.sampling_params.max_new_tokens}, "
                f"cap={rotary.max_positions})"
            )

    def _cpu_staging_empty(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        device = torch.device(self.device)
        pin_memory = device.type == "cuda" and torch.cuda.is_available()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)

    def _allocate_decode_buffers(self, capacity: int, pages_per_seq: int) -> None:
        capacity = max(1, capacity)
        pages_per_seq = max(1, pages_per_seq)
        device = torch.device(self.device)

        self._dec_buffer_capacity = capacity
        self._dec_buffer_pages_per_seq = pages_per_seq
        self._dec_input_ids = torch.empty(
            (capacity, 1), dtype=torch.long, device=device
        )
        self._dec_position_ids = torch.empty(
            (capacity, 1), dtype=torch.long, device=device
        )
        self._dec_cache_seqlens = torch.empty(
            (capacity,), dtype=torch.int32, device=device
        )
        self._dec_block_table = torch.empty(
            (capacity, pages_per_seq), dtype=torch.int32, device=device
        )

        self._dec_input_ids_cpu = self._cpu_staging_empty((capacity, 1), torch.long)
        self._dec_position_ids_cpu = self._cpu_staging_empty(
            (capacity, 1), torch.long
        )
        self._dec_cache_seqlens_cpu = self._cpu_staging_empty(
            (capacity,), torch.int32
        )
        self._dec_block_table_cpu = self._cpu_staging_empty(
            (capacity, pages_per_seq), torch.int32
        )

    def _ensure_decode_buffers(self, batch_size: int, pages_per_seq: int) -> None:
        if (
            self._dec_input_ids is not None
            and batch_size <= self._dec_buffer_capacity
            and pages_per_seq <= self._dec_buffer_pages_per_seq
        ):
            return

        capacity = max(batch_size, self._dec_buffer_capacity, self.max_running)
        page_capacity = max(
            pages_per_seq,
            self._dec_buffer_pages_per_seq,
            self.max_decode_pages_per_seq,
        )
        self._allocate_decode_buffers(capacity=capacity, pages_per_seq=page_capacity)

    def _prepare_paged_decode_inputs(
        self, requests: list[Request]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(requests)
        max_pages = max(len(req.page_table) for req in requests)
        self._ensure_decode_buffers(batch_size=batch_size, pages_per_seq=max_pages)

        if (
            self._dec_input_ids is None
            or self._dec_position_ids is None
            or self._dec_cache_seqlens is None
            or self._dec_block_table is None
            or self._dec_input_ids_cpu is None
            or self._dec_position_ids_cpu is None
            or self._dec_cache_seqlens_cpu is None
            or self._dec_block_table_cpu is None
        ):
            raise RuntimeError("paged decode buffers are not initialized")

        input_ids_cpu = self._dec_input_ids_cpu[:batch_size]
        position_ids_cpu = self._dec_position_ids_cpu[:batch_size]
        cache_seqlens_cpu = self._dec_cache_seqlens_cpu[:batch_size]
        block_table_cpu = self._dec_block_table_cpu[:batch_size, :max_pages]
        block_table_cpu.zero_()

        for i, req in enumerate(requests):
            cache_len = req.cache_len
            input_ids_cpu[i, 0] = req.output_ids[-1]
            position_ids_cpu[i, 0] = cache_len
            cache_seqlens_cpu[i] = cache_len

            page_tensor = torch.as_tensor(req.page_table, dtype=torch.int32)
            block_table_cpu[i, : len(req.page_table)].copy_(page_tensor)

        input_ids = self._dec_input_ids[:batch_size]
        position_ids = self._dec_position_ids[:batch_size]
        cache_seqlens = self._dec_cache_seqlens[:batch_size]
        block_table = self._dec_block_table[:batch_size, :max_pages]

        input_ids.copy_(input_ids_cpu, non_blocking=True)
        position_ids.copy_(position_ids_cpu, non_blocking=True)
        cache_seqlens.copy_(cache_seqlens_cpu, non_blocking=True)
        block_table.copy_(block_table_cpu, non_blocking=True)

        return block_table, cache_seqlens, input_ids, position_ids

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        self._validate_rope_budget(request)
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def paged_prefill(self, requests: list[Request]) -> list[int]:
        """
        Packed prefill for paged mode.

        Prompts are flattened into one sequence with per-request RoPE
        positions and block-diagonal causal attention. Fresh K/V values
        are written directly into each request's allocated pool pages.
        """
        if not requests:
            return []

        pool = self._require_kv_pool()
        prompt_lens = [req.num_input_tokens for req in requests]
        if any(length <= 0 for length in prompt_lens):
            raise ValueError("paged prefill requires non-empty prompts")
        for req in requests:
            self._validate_rope_budget(req)

        page_counts = [pool.pages_needed(length) for length in prompt_lens]
        total_pages = sum(page_counts)
        if total_pages > pool.num_free:
            raise RuntimeError(
                f"KV pool OOM during prefill: need {total_pages} pages, "
                f"have {pool.num_free} free"
            )

        allocated: list[list[int]] = []
        try:
            for req, num_pages in zip(requests, page_counts):
                req.page_table = pool.allocate(num_pages)
                allocated.append(req.page_table)

            flat_ids: list[int] = []
            flat_positions: list[int] = []
            slot_mapping: list[int] = []
            cu_values = [0]
            for req in requests:
                flat_ids.extend(req.input_ids)
                flat_positions.extend(range(req.num_input_tokens))
                for pos in range(req.num_input_tokens):
                    page_idx = req.page_table[pos // pool.page_size]
                    slot_mapping.append(
                        page_idx * pool.page_size + pos % pool.page_size
                    )
                cu_values.append(cu_values[-1] + req.num_input_tokens)

            input_ids = torch.tensor(
                [flat_ids], dtype=torch.long, device=self.device
            )
            position_ids = torch.tensor(
                [flat_positions], dtype=torch.long, device=self.device
            )
            cu_seqlens = torch.tensor(
                cu_values, dtype=torch.int32, device=self.device
            )
            slot_mapping_tensor = torch.tensor(
                slot_mapping, dtype=torch.long, device=self.device
            )
            logit_positions = cu_seqlens[1:].to(dtype=torch.long) - 1
            paged_ctx = self._make_paged_context(
                is_prefill=True,
                block_table=None,
                cache_seqlens=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=max(prompt_lens),
                slot_mapping=slot_mapping_tensor,
            )

            logits, _ = self.model(
                input_ids,
                position_ids,
                kv_caches=None,
                paged_ctx=paged_ctx,
                logit_positions=logit_positions,
            )

            token_ids: list[int] = []
            for i, req in enumerate(requests):
                req.cache_len = req.num_input_tokens
                token_ids.append(
                    sample_token(
                        logits[:, i, :],
                        req.sampling_params,
                        req.output_ids,
                    )
                )
            return token_ids
        except Exception:
            for pages in allocated:
                if pages:
                    pool.free(pages)
            for req in requests:
                req.page_table = []
                req.cache_len = 0
            raise

    @torch.inference_mode()
    def paged_decode(self, requests: list[Request]) -> list[int]:
        """Decode one token per request using the paged KV cache."""
        if not requests:
            return []

        pool = self._require_kv_pool()
        grow_requests = [
            req for req in requests if req.cache_len % pool.page_size == 0
        ]
        if len(grow_requests) > pool.num_free:
            raise RuntimeError(
                f"KV pool OOM during decode: need {len(grow_requests)} pages, "
                f"have {pool.num_free} free"
            )

        grown_pages: list[tuple[Request, int]] = []
        try:
            for req in grow_requests:
                page = pool.allocate(1)[0]
                req.page_table.append(page)
                grown_pages.append((req, page))

            (
                block_table,
                cache_seqlens,
                input_ids,
                position_ids,
            ) = self._prepare_paged_decode_inputs(requests)

            if (
                self.cuda_graph_runner is not None
                and len(requests) <= self.cuda_graph_runner.max_bucket
            ):
                logits = self.cuda_graph_runner.replay(len(requests))
            else:
                paged_ctx = self._make_paged_context(
                    is_prefill=False,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    slot_mapping=None,
                )
                logits, _ = self.model(
                    input_ids,
                    position_ids,
                    kv_caches=None,
                    paged_ctx=paged_ctx,
                )

            token_ids = sample_tokens_batched(
                logits[:, -1, :],
                [req.sampling_params for req in requests],
                [req.output_ids for req in requests],
            )
            for req in requests:
                req.cache_len += 1
            return token_ids
        except Exception:
            for req, page in reversed(grown_pages):
                if req.page_table and req.page_table[-1] == page:
                    req.page_table.pop()
                pool.free([page])
            raise

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids
