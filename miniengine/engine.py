"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Bare-bone model wrapper for single-request prefill and decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype

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

        self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()

        # ── Batched-decode session state ───────────────────────────────
        self._batch_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._batch_orig_lens: list[int] | None = None
        self._batch_orig_lens_t: torch.Tensor | None = None
        self._batch_orig_max: int = 0
        self._batch_requests: list[Request] | None = None
        self._batch_seq_len: int = 0

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

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=request.kv_cache)
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    # ── Batched-decode session state ────────────────────────────────────

    def batch_matches(self, requests: list[Request]) -> bool:
        """Return True when the active batched state matches `requests`."""
        batch_requests = self._batch_requests
        if batch_requests is None or len(batch_requests) != len(requests):
            return False
        return all(a is b for a, b in zip(batch_requests, requests))

    def _materialize_batch_row(self, row_idx: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert one row from the active batched KV back into a contiguous
        per-request cache.
        """
        kv = self._batch_kv
        orig_lens = self._batch_orig_lens
        if kv is None or orig_lens is None:
            raise RuntimeError("No active batched KV to materialize from.")

        orig_len = orig_lens[row_idx]
        cur_seq_len = self._batch_seq_len
        tail_start = self._batch_orig_max

        per_req: list[tuple[torch.Tensor, torch.Tensor]] = []
        for K, V in kv:
            prefix_k = K[row_idx : row_idx + 1, :, :orig_len, :]
            prefix_v = V[row_idx : row_idx + 1, :, :orig_len, :]
            if cur_seq_len > tail_start:
                tail_k = K[row_idx : row_idx + 1, :, tail_start:cur_seq_len, :]
                tail_v = V[row_idx : row_idx + 1, :, tail_start:cur_seq_len, :]
                per_req.append(
                    (
                        torch.cat([prefix_k, tail_k], dim=2),
                        torch.cat([prefix_v, tail_v], dim=2),
                    )
                )
            else:
                per_req.append((prefix_k.clone(), prefix_v.clone()))
        return per_req

    def start_batch(self, requests: list[Request]) -> None:
        """
        Build the persistent batched KV state from per-request caches.

        The running set keeps this batched state across decode iterations.
        We only rebuild when the batch composition changes.
        """
        if not requests:
            self.end_batch()
            return

        num_layers = len(requests[0].kv_cache)
        cache_lens = [r.kv_cache[0][0].shape[2] for r in requests]
        max_len = max(cache_lens)
        max_remaining = max(
            max(1, req.sampling_params.max_new_tokens - req.num_output_tokens)
            for req in requests
        )
        capacity = max_len + max_remaining
        B = len(requests)

        sample_k = requests[0].kv_cache[0][0]
        kvh, hd = sample_k.shape[1], sample_k.shape[3]
        dtype, device = sample_k.dtype, sample_k.device

        batched_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(num_layers):
            K = torch.empty(B, kvh, capacity, hd, dtype=dtype, device=device)
            V = torch.empty(B, kvh, capacity, hd, dtype=dtype, device=device)
            for i, (req, L) in enumerate(zip(requests, cache_lens)):
                K[i, :, :L, :] = req.kv_cache[layer][0][0]
                V[i, :, :L, :] = req.kv_cache[layer][1][0]
            batched_kv.append((K, V))

        self._batch_kv = batched_kv
        self._batch_orig_lens = list(cache_lens)
        self._batch_orig_lens_t = torch.tensor(
            cache_lens, dtype=torch.long, device=device
        )
        self._batch_orig_max = max_len
        self._batch_requests = list(requests)
        self._batch_seq_len = max_len

        # Batched KV is now the source of truth.
        for req in requests:
            req.kv_cache = None

    def rebuild_batch(self, requests: list[Request]) -> None:
        """
        Rebuild the persistent batched KV when batch composition changes.

        Existing running requests recover their contiguous per-request KV
        from the prior batched state; newly admitted requests already carry
        a per-request cache from prefill.
        """
        if self._batch_kv is not None and self._batch_requests is not None:
            row_for_request = {id(req): i for i, req in enumerate(self._batch_requests)}
            for req in requests:
                if req.kv_cache is None:
                    row_idx = row_for_request.get(id(req))
                    if row_idx is None:
                        raise RuntimeError(
                            "Cannot rebuild batch: request missing both "
                            "per-request KV and prior batched row."
                        )
                    req.kv_cache = self._materialize_batch_row(row_idx)
        self.start_batch(requests)

    def drop_batch_rows(self, keep_indices: list[int]) -> None:
        """Drop retired requests from the active batched KV state."""
        if self._batch_kv is None:
            return
        if not keep_indices:
            self.end_batch()
            return

        current_B = self._batch_kv[0][0].shape[0]
        if len(keep_indices) == current_B:
            return

        idx = torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        self._batch_kv = [
            (K.index_select(0, idx), V.index_select(0, idx))
            for K, V in self._batch_kv
        ]
        self._batch_orig_lens = [self._batch_orig_lens[i] for i in keep_indices]
        self._batch_orig_lens_t = self._batch_orig_lens_t.index_select(0, idx)
        self._batch_requests = [self._batch_requests[i] for i in keep_indices]

    def end_batch(self) -> None:
        """Free the persistent batched KV state after the batch drains."""
        self._batch_kv = None
        self._batch_orig_lens = None
        self._batch_orig_lens_t = None
        self._batch_orig_max = 0
        self._batch_requests = None
        self._batch_seq_len = 0

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Run one decode step using the active persistent batched KV state.

        Returns the next token id for each request, in input order.
        """
        kv = self._batch_kv
        orig_lens = self._batch_orig_lens
        if kv is None or orig_lens is None or not self.batch_matches(requests):
            raise RuntimeError(
                "batched_decode called without a matching active batch. "
                "Scheduler must rebuild or start the batch first."
            )
        B = len(requests)

        # ── 1. Stack the last generated token from each request ────────
        input_ids = torch.tensor(
            [[r.output_ids[-1]] for r in requests],
            dtype=torch.long,
            device=self.device,
        )

        cur_seq_len = self._batch_seq_len
        decoded_so_far = cur_seq_len - self._batch_orig_max
        cur_lens = [L + decoded_so_far for L in orig_lens]

        # ── 2. Per-request RoPE positions = each request's logical cache len
        position_ids = torch.tensor(
            [[L] for L in cur_lens], dtype=torch.long, device=self.device,
        )

        # ── 3. Build the attention mask only when the batch is non-uniform.
        if all(L == self._batch_orig_max for L in orig_lens):
            attn_mask = None
        else:
            total_len = cur_seq_len + 1
            arange = torch.arange(total_len, device=self.device)
            valid = (arange[None, :] < self._batch_orig_lens_t[:, None]) | (
                arange[None, :] >= self._batch_orig_max
            )
            attn_mask = valid.view(B, 1, 1, total_len)

        # ── 4. Single batched forward pass. Model concatenates the new
        # K/V to the cache internally, growing seq dim by 1.
        logits, new_kv = self.model(
            input_ids,
            position_ids,
            kv_caches=kv,
            attn_mask=attn_mask,
            kv_write_pos=cur_seq_len,
        )
        self._batch_kv = new_kv
        self._batch_seq_len += input_ids.shape[1]

        # Fast path for the serving benchmark's common greedy decode:
        # avoid per-request clone/filter/sample work when every request is
        # simple argmax with no repetition penalty.
        if all(
            req.sampling_params.temperature == 0
            and req.sampling_params.repetition_penalty == 1.0
            for req in requests
        ):
            return logits[:, -1, :].argmax(dim=-1).tolist()

        # ── 5. Sample one token per request.  Slice keeps the (1, vocab)
        # shape sample_token / apply_repetition_penalty expect.
        return [
            sample_token(
                logits[i:i+1, -1, :], req.sampling_params, req.output_ids
            )
            for i, req in enumerate(requests)
        ]

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
