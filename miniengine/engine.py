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
import torch.nn.functional as F
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
    
    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Run one decode step for a batch of already-prefilled requests.

        Pads each request's KV cache to the batch's max length, runs one
        batched forward pass with an attention mask that ignores padding,
        unpacks the per-request KV caches, and samples one token per
        request.

        Returns the next token id for each request, in input order.
        """
        B = len(requests)
        num_layers = len(requests[0].kv_cache)
        # Per-request cache length, read from layer 0's K seq dim.
        cache_lens = [request.kv_cache[0][0].shape[2] for request in requests]
        max_cache_len = max(cache_lens)

        # ── 1. Stack the last generated token from each request ────────
        input_ids = torch.tensor(
            [[request.output_ids[-1]] for request in requests],
            dtype=torch.long,
            device=self.device,
        )

        # ── 2. Per-request RoPE positions = each request's own cache_len
        position_ids = torch.tensor(
            [[L] for L in cache_lens], dtype=torch.long, device=self.device
        )

        # ── 3. Pad each KV to max_cache_len and stack along batch dim ──
        padded_caches = []
        for layer in range(num_layers):
            ks, vs = [], []
            for req, L in zip(requests, cache_lens):
                k, v = req.kv_cache[layer]
                pad = max_cache_len - L
                if pad:
                    k = F.pad(k, (0, 0, 0, pad))
                    v = F.pad(v, (0, 0, 0, pad))
                ks.append(k)
                vs.append(v)
            padded_caches.append((torch.cat(ks, 0), torch.cat(vs, 0)))

        # ── 4. Build the attention mask ────────────────────────────────
        total_cache_len = max_cache_len + 1
        arange = torch.arange(total_cache_len, device=self.device)
        cache_lens_t = torch.tensor(cache_lens, device=self.device)
        valid = arange[None, :] < cache_lens_t[:, None]   # (B, total_cache_len)
        valid[:, -1] = True                                # new decode token always valid
        attn_mask = valid.view(B, 1, 1, total_cache_len)

        # ── 5. Single batched forward pass ─────────────────────────────
        logits, new_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_caches,
            attn_mask=attn_mask,
        )

        # ── 6. Unpad per-request KV caches and store back ──────────────
        for i, (request, L) in enumerate(zip(requests, cache_lens)):
            new_kv_per_req = []
            for layer in range(num_layers):
                K, V = new_caches[layer]
                old_k, old_v = K[i:i+1, :, :L, :], V[i:i+1, :, :L, :]
                new_k, new_v = K[i:i+1, :, -1:, :], V[i:i+1, :, -1:, :]
                new_kv_per_req.append((
                    torch.cat([old_k, new_k], dim=2),
                    torch.cat([old_v, new_v], dim=2),
                ))
            request.kv_cache = new_kv_per_req

        # ── 7. Sample one token per request from its row of logits ─────
        return [
            sample_token(
                logits[i, -1, :], request.sampling_params, request.output_ids
            )
            for i, request in enumerate(requests)
        ]

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
