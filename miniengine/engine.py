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

        # ── Batched-decode session state (set by start_batch) ──────────
        self._batch_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._batch_orig_lens: list[int] | None = None
        self._batch_orig_max: int = 0

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
    # The batched KV is built once per scheduler batch (start_batch),
    # grown in place across decode steps (batched_decode), shrunk on
    # request retirement (drop_batch_rows), and freed at batch end
    # (end_batch).  This avoids re-padding and re-unpacking the entire
    # KV cache on every decode token, which dominated the per-step cost
    # in the original implementation.
    #
    # Layout per layer:
    #   K, V: (B, num_kv_heads, seq_len, head_dim)
    # For request i, the "real" cached tokens occupy positions
    #   [0, _batch_orig_lens[i])      — original cache from prefill
    #   [_batch_orig_max, seq_len)    — tokens generated during this
    #                                   batched-decode session
    # Positions in [_batch_orig_lens[i], _batch_orig_max) are padding
    # garbage and are masked out by the attention mask.

    def start_batch(self, requests: list[Request]) -> None:
        """
        Build the batched KV state from each request's per-request KV.

        Called once by the scheduler after prefill, before the
        batched_decode loop.  Transfers ownership of the KV cache from
        the per-request tensors to a single set of batched tensors.
        """
        if not requests:
            self.end_batch()
            return

        num_layers = len(requests[0].kv_cache)
        cache_lens = [r.kv_cache[0][0].shape[2] for r in requests]
        max_len = max(cache_lens)
        B = len(requests)

        sample_k = requests[0].kv_cache[0][0]
        kvh, hd = sample_k.shape[1], sample_k.shape[3]
        dtype, device = sample_k.dtype, sample_k.device

        batched_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(num_layers):
            K = torch.empty(B, kvh, max_len, hd, dtype=dtype, device=device)
            V = torch.empty(B, kvh, max_len, hd, dtype=dtype, device=device)
            for i, (req, L) in enumerate(zip(requests, cache_lens)):
                K[i, :, :L, :] = req.kv_cache[layer][0][0]
                V[i, :, :L, :] = req.kv_cache[layer][1][0]
            batched_kv.append((K, V))

        self._batch_kv = batched_kv
        self._batch_orig_lens = list(cache_lens)
        self._batch_orig_max = max_len

        # Per-request caches are no longer the source of truth.  Free
        # them so we don't keep duplicate KV memory around.
        for r in requests:
            r.kv_cache = None

    def end_batch(self) -> None:
        """Free the batched KV state after the scheduler batch drains."""
        self._batch_kv = None
        self._batch_orig_lens = None
        self._batch_orig_max = 0

    def drop_batch_rows(self, keep_indices: list[int]) -> None:
        """
        Drop retired requests' rows from the batched KV state.

        `keep_indices` are indices into the current batch (i.e., the
        order matches the request list passed to the most recent
        batched_decode call).
        """
        if self._batch_kv is None:
            return
        if not keep_indices:
            self.end_batch()
            return
        current_B = self._batch_kv[0][0].shape[0]
        if len(keep_indices) == current_B:
            return  # no-op when nothing was retired

        idx = torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        self._batch_kv = [
            (K.index_select(0, idx), V.index_select(0, idx))
            for K, V in self._batch_kv
        ]
        self._batch_orig_lens = [self._batch_orig_lens[i] for i in keep_indices]

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Run one decode step using the batched KV state set up by
        start_batch().

        Returns the next token id for each request, in input order.
        """
        kv = self._batch_kv
        if kv is None:
            raise RuntimeError(
                "batched_decode called without an active batch — "
                "Scheduler must call start_batch() first."
            )

        B = len(requests)
        cur_seq_len = kv[0][0].shape[2]                 # grows by 1 each step
        orig_max = self._batch_orig_max
        orig_lens = self._batch_orig_lens
        decoded_so_far = cur_seq_len - orig_max         # tokens generated so far
        # Per-request logical cache len (count of real tokens), used for RoPE.
        cur_lens = [L + decoded_so_far for L in orig_lens]

        # ── 1. Stack the last generated token from each request ────────
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # ── 2. Per-request RoPE positions = each request's own cache_len
        position_ids = torch.tensor(
            [[L] for L in cur_lens], dtype=torch.long, device=self.device,
        )

        # ── 3. Build the attention mask only when the batch is non-uniform
        # If every row started with the same cache length, there is no
        # padding gap and SDPA's faster maskless path can be used.
        if all(L == orig_max for L in orig_lens):
            attn_mask = None
        else:
            # Mask shape: (B, 1, 1, cur_seq_len + 1).  Valid for row i at
            # position j iff (j < orig_lens[i]) OR (j >= orig_max).
            total_len = cur_seq_len + 1
            arange = torch.arange(total_len, device=self.device)
            orig_lens_t = torch.tensor(orig_lens, device=self.device)
            valid = (arange[None, :] < orig_lens_t[:, None]) | (
                arange[None, :] >= orig_max
            )
            attn_mask = valid.view(B, 1, 1, total_len)

        # ── 4. Single batched forward pass — model concatenates the new
        # token's K/V to the cache, growing seq dim by 1.
        logits, new_kv = self.model(
            input_ids, position_ids, kv_caches=kv, attn_mask=attn_mask,
        )
        self._batch_kv = new_kv

        # ── 5. Sample one token per request.  Slice keeps the (1, vocab)
        # shape that sample_token / apply_repetition_penalty expect.
        return [
            sample_token(
                logits[i:i+1, -1, :], req.sampling_params, req.output_ids
            )
            for i, req in enumerate(requests)
        ]

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
