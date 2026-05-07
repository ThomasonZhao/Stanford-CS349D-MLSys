"""
Bare-bone Qwen3 transformer in pure PyTorch.

No HuggingFace model classes — just nn.Module, nn.Linear, and manual
attention with KV cache.  Weight names match the HuggingFace checkpoint
so we can load safetensors directly via load_state_dict().

Architecture (Qwen3-4B as reference):
    Embedding(151936, 2560)
    36 x TransformerBlock:
        RMSNorm → Attention(GQA + QK-Norm + RoPE) → RMSNorm → SwiGLU MLP
    RMSNorm
    LM Head (tied with embedding)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
except Exception:  # pragma: no cover - optional CUDA-only dependency
    _flash_attn_varlen_func = None
    _flash_attn_with_kvcache = None

_Attention_flash_decode_logged = [False]
_Attention_flash_decode_fallback_logged = [False]
_Attention_flash_prefill_logged = [False]
_Attention_flash_prefill_fallback_logged = [False]
_flash_decode_disabled = False


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Model architecture config, loaded from HuggingFace config.json."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128  # explicit, NOT hidden_size // num_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> ModelConfig:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=getattr(hf, "rope_theta", 10000.0),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )


@dataclass
class PagedContext:
    """Paged KV state shared by all transformer layers for one forward pass."""

    k_cache_pool: list[torch.Tensor]
    v_cache_pool: list[torch.Tensor]
    block_table: torch.Tensor | None
    cache_seqlens: torch.Tensor | None
    is_prefill: bool

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    slot_mapping: torch.Tensor | None = None


# ── Building blocks ─────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance in fp32 — bf16 mean-of-squares loses too much precision.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes and caches cos/sin tables, indexed by position_ids at
    forward time.  The cache grows on-demand so we never allocate for the
    full 256K context upfront.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (batch, seq_len) integer positions.

        Returns:
            cos, sin each of shape (batch, seq_len, 1, head_dim).
            TransformerModel reshapes these once per forward for the
            active attention layout.
        """
        max_pos = int(position_ids.max().item()) + 1

        if self._cos is None or max_pos > self._cached_len:
            length = max(max_pos, self._cached_len * 2, 256)
            t = torch.arange(
                length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(t, self.inv_freq)  # (length, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (length, head_dim)
            self._cos = emb.cos()
            self._sin = emb.sin()
            self._cached_len = length

        # Index into cache: (batch, seq_len, head_dim) → add head dim
        cos = self._cos[position_ids].unsqueeze(2)  # (batch, seq_len, 1, head_dim)
        sin = self._sin[position_ids].unsqueeze(2)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE to x.

    x:   attention tensor in the active layout
    cos: layout-and-dtype-correct table, broadcastable to x
    sin: same shape as cos
    """
    return x * cos + _rotate_half(x) * sin


# ── Attention ───────────────────────────────────────────────────────────


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA), QK-Norm,
    and Rotary Position Embeddings.

    Q projects to  num_attention_heads  × head_dim
    K projects to  num_key_value_heads  × head_dim
    V projects to  num_key_value_heads  × head_dim
    O projects back to hidden_size
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Qwen3: RMSNorm on Q and K after projection (per-head)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _expand_kv_for_gqa(
        self, k: torch.Tensor, v: torch.Tensor, bsz: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_groups <= 1:
            return k, v
        k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
        k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
        v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
        v = v.reshape(bsz, self.num_heads, -1, self.head_dim)
        return k, v

    def _paged_prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_ctx: PagedContext,
        layer_idx: int,
    ) -> torch.Tensor:
        if paged_ctx.slot_mapping is None:
            raise ValueError("paged prefill requires slot_mapping")
        if paged_ctx.cu_seqlens_q is None or paged_ctx.cu_seqlens_k is None:
            raise ValueError("paged prefill requires cu_seqlens")
        if paged_ctx.max_seqlen_q is None or paged_ctx.max_seqlen_k is None:
            raise ValueError("paged prefill requires max_seqlen values")
        if q.shape[0] != 1:
            raise ValueError("packed paged prefill expects batch size 1")

        k_cache = paged_ctx.k_cache_pool[layer_idx]
        v_cache = paged_ctx.v_cache_pool[layer_idx]
        k_packed = k[0].contiguous()
        v_packed = v[0].contiguous()
        k_cache.view(-1, self.num_kv_heads, self.head_dim)[
            paged_ctx.slot_mapping
        ] = k_packed
        v_cache.view(-1, self.num_kv_heads, self.head_dim)[
            paged_ctx.slot_mapping
        ] = v_packed

        q_packed = q[0].contiguous()
        if _flash_attn_varlen_func is not None and q_packed.is_cuda:
            out_packed = _flash_attn_varlen_func(
                q_packed,
                k_packed,
                v_packed,
                paged_ctx.cu_seqlens_q,
                paged_ctx.cu_seqlens_k,
                paged_ctx.max_seqlen_q,
                paged_ctx.max_seqlen_k,
                causal=True,
            )
            if not _Attention_flash_prefill_logged[0]:
                _Attention_flash_prefill_logged[0] = True
                logger.info("paged prefill: flash_attn_varlen_func active")
            return out_packed.unsqueeze(0)

        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        k_full, v_full = self._expand_kv_for_gqa(k_sdpa, v_sdpa, bsz=1)
        total_tokens = q.shape[1]
        seq_lens = (paged_ctx.cu_seqlens_q[1:] - paged_ctx.cu_seqlens_q[:-1]).to(
            device=q.device, dtype=torch.long
        )
        segment_ids = torch.repeat_interleave(
            torch.arange(seq_lens.numel(), device=q.device), seq_lens
        )
        token_idx = torch.arange(total_tokens, device=q.device)
        allowed = (segment_ids[:, None] == segment_ids[None, :]) & (
            token_idx[:, None] >= token_idx[None, :]
        )
        attn_mask = torch.full(
            (total_tokens, total_tokens),
            float("-inf"),
            dtype=q.dtype,
            device=q.device,
        )
        attn_mask.masked_fill_(allowed, 0.0)
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_full,
            v_full,
            attn_mask=attn_mask.view(1, 1, total_tokens, total_tokens),
        )
        return out.transpose(1, 2).contiguous()

    def _paged_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_ctx: PagedContext,
        layer_idx: int,
    ) -> torch.Tensor:
        if paged_ctx.block_table is None or paged_ctx.cache_seqlens is None:
            raise ValueError("paged decode requires block_table and cache_seqlens")

        k_cache = paged_ctx.k_cache_pool[layer_idx]
        v_cache = paged_ctx.v_cache_pool[layer_idx]

        global _flash_decode_disabled
        if (
            _flash_attn_with_kvcache is not None
            and q.is_cuda
            and not _flash_decode_disabled
        ):
            try:
                out = _flash_attn_with_kvcache(
                    q,
                    k_cache,
                    v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=paged_ctx.cache_seqlens,
                    block_table=paged_ctx.block_table,
                    causal=True,
                )
                if not _Attention_flash_decode_logged[0]:
                    _Attention_flash_decode_logged[0] = True
                    logger.info("paged decode: flash_attn_with_kvcache active")
                return out
            except Exception:
                _flash_decode_disabled = True
                if not _Attention_flash_decode_fallback_logged[0]:
                    _Attention_flash_decode_fallback_logged[0] = True
                    logger.warning(
                        "flash-attn paged decode failed; disabling flash decode "
                        "and using SDPA fallback",
                        exc_info=True,
                    )

        bsz, seq_len, _, _ = q.shape
        if paged_ctx.slot_mapping is not None:
            slot_mapping = paged_ctx.slot_mapping
        else:
            page_size = k_cache.shape[1]
            cache_pos = paged_ctx.cache_seqlens.to(torch.long)
            page_offsets = cache_pos // page_size
            in_page_offsets = cache_pos % page_size
            page_ids = paged_ctx.block_table.gather(
                1, page_offsets.view(-1, 1).to(paged_ctx.block_table.device)
            ).squeeze(1)
            slot_mapping = page_ids.to(torch.long) * page_size + in_page_offsets

        if seq_len != 1:
            raise ValueError("paged decode fallback currently expects one token")
        k_new = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v_new = v.reshape(-1, self.num_kv_heads, self.head_dim)
        k_cache.view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = k_new
        v_cache.view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = v_new

        page_size = k_cache.shape[1]
        kv_lens = paged_ctx.cache_seqlens.to(torch.long) + seq_len
        max_kv_len = int(kv_lens.max().item())
        positions = torch.arange(max_kv_len, device=q.device)
        page_offsets = positions // page_size
        in_page_offsets = positions % page_size
        page_ids = paged_ctx.block_table[:, page_offsets].to(torch.long)
        flat_offsets = page_ids * page_size + in_page_offsets

        flat_k = k_cache.view(-1, self.num_kv_heads, self.head_dim)
        flat_v = v_cache.view(-1, self.num_kv_heads, self.head_dim)
        gathered_k = flat_k[flat_offsets].permute(0, 2, 1, 3).contiguous()
        gathered_v = flat_v[flat_offsets].permute(0, 2, 1, 3).contiguous()
        gathered_k, gathered_v = self._expand_kv_for_gqa(gathered_k, gathered_v, bsz)

        mask_positions = torch.arange(max_kv_len, device=q.device).view(1, 1, 1, -1)
        valid_lens = kv_lens.to(q.device).view(bsz, 1, 1, 1)
        attn_mask = torch.zeros(
            bsz, 1, 1, max_kv_len, dtype=q.dtype, device=q.device
        )
        attn_mask.masked_fill_(mask_positions >= valid_lens, float("-inf"))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), gathered_k, gathered_v, attn_mask=attn_mask
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedContext | None = None,
        layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden:         (batch, seq_len, hidden_size)
            cos, sin:       from RotaryEmbedding, broadcastable
            kv_cache:       optional (cached_k, cached_v), each
                            (batch, num_kv_heads, cache_len, head_dim)
            attention_mask: optional float mask (batch, 1, q_len, kv_len)
                            for batched decode with padded KV; 0 = attend,
                            -inf = ignore.
            paged_ctx:     optional paged KV cache context.
            layer_idx:     transformer layer index when using paged_ctx.

        Returns:
            output:       (batch, seq_len, hidden_size)
            new_kv_cache: (k, v) with updated cache
        """
        bsz, seq_len, _ = hidden.shape

        if paged_ctx is not None:
            q = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(hidden).view(
                bsz, seq_len, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj(hidden).view(
                bsz, seq_len, self.num_kv_heads, self.head_dim
            )
        else:
            # Non-paged SDPA keeps the original (batch, heads, seq, dim) layout.
            q = (
                self.q_proj(hidden)
                .view(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj(hidden)
                .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(hidden)
                .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
                .transpose(1, 2)
            )

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if paged_ctx is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required for paged attention")
            if paged_ctx.is_prefill:
                out = self._paged_prefill_attention(q, k, v, paged_ctx, layer_idx)
            else:
                out = self._paged_decode_attention(q, k, v, paged_ctx, layer_idx)
            out = out.reshape(bsz, seq_len, self.num_heads * self.head_dim)
            return self.o_proj(out), (k, v)

        # Append to KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        # GQA: expand KV heads to match Q heads
        k, v = self._expand_kv_for_gqa(k, v, bsz)

        # Batched decode passes an explicit float mask; otherwise fall
        # back to the is_causal kernel path.
        if attention_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            is_causal = kv_cache is None and seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Merge heads → project back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_kv


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer block ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: LN → Attn → residual → LN → MLP → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self._compiled_decode_ffn = None
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _decode_ffn(self, residual: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Fused post-norm + MLP + residual; safe to compile (no attention,
        # no Python branching, stable shapes during decode).
        return residual + self.mlp(self.post_attention_layernorm(hidden))

    def enable_torch_compile(self) -> None:
        self._compiled_decode_ffn = torch.compile(self._decode_ffn, dynamic=True)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedContext | None = None,
        layer_idx: int | None = None,
        use_compiled_mlp: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, new_kv = self.self_attn(
            hidden, cos, sin, kv_cache, attention_mask, paged_ctx, layer_idx
        )
        hidden = residual + hidden

        if use_compiled_mlp and self._compiled_decode_ffn is not None:
            hidden = self._compiled_decode_ffn(hidden, hidden)
        else:
            residual = hidden
            hidden = self.post_attention_layernorm(hidden)
            hidden = self.mlp(hidden)
            hidden = residual + hidden

        return hidden, new_kv


# ── Full model ──────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """The core transformer: embedding → N layers → final norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)
        self.use_torch_compile = False

    def enable_torch_compile(self) -> None:
        for layer in self.layers:
            layer.enable_torch_compile()
        self.use_torch_compile = True

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedContext | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:      (batch, seq_len)
            position_ids:   (batch, seq_len)
            kv_caches:      list of per-layer (key, value) caches, or None
            attention_mask: optional float mask for batched-decode SDPA
            paged_ctx:      optional paged KV cache context

        Returns:
            hidden:         (batch, seq_len, hidden_size)
            new_kv_caches:  list of per-layer (key, value) with appended tokens
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)
        if paged_ctx is not None:
            cos = cos.to(dtype=hidden.dtype)
            sin = sin.to(dtype=hidden.dtype)
        else:
            cos = cos.transpose(1, 2).to(dtype=hidden.dtype)
            sin = sin.transpose(1, 2).to(dtype=hidden.dtype)
        use_compiled_mlp = (
            self.use_torch_compile
            and paged_ctx is not None
            and not paged_ctx.is_prefill
        )

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden, new_kv = layer(
                hidden,
                cos,
                sin,
                kv,
                attention_mask,
                paged_ctx=paged_ctx,
                layer_idx=i,
                use_compiled_mlp=use_compiled_mlp,
            )
            new_kv_caches.append(new_kv)

        hidden = self.norm(hidden)
        return hidden, new_kv_caches


class CausalLM(nn.Module):
    """
    Complete causal language model: transformer + LM head.

    The LM head may be tied with the embedding (Qwen3-4B) or separate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def enable_torch_compile(self) -> None:
        self.model.enable_torch_compile()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        paged_ctx: PagedContext | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            logits:        (batch, seq_len, vocab_size)
            new_kv_caches: per-layer KV caches
        """
        hidden, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, attention_mask, paged_ctx
        )
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits, new_kv_caches


# ── Weight loading ──────────────────────────────────────────────────────


def load_weights(
    model: CausalLM,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """
    Load weights from HuggingFace safetensors into the model.

    Handles both single-file and sharded checkpoints.  Weight names in the
    checkpoint match our module hierarchy exactly (by design), so we can
    use load_state_dict() directly.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info("Downloading / locating model files for %s …", model_path)
    local_path = Path(
        snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    # Gather all safetensor shard files
    st_files = sorted(local_path.glob("model*.safetensors"))
    if not st_files:
        # Fallback: some repos use a single "model.safetensors"
        st_files = sorted(local_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {local_path}")

    logger.info("Loading %d safetensors shard(s) …", len(st_files))

    # Load shards straight to the target device — avoids a full CPU copy.
    state_dict: dict[str, torch.Tensor] = {}
    for f in st_files:
        for key, tensor in load_file(str(f), device=device).items():
            state_dict[key] = tensor.to(dtype=dtype)

    # Drop checkpoint keys the model doesn't expect.
    model_keys = set(model.state_dict().keys())
    extra = set(state_dict.keys()) - model_keys
    for key in extra:
        del state_dict[key]
    if extra:
        logger.info("Skipped %d unexpected checkpoint keys", len(extra))

    if "lm_head.weight" in model_keys and "lm_head.weight" not in state_dict:
        logger.info("Tying lm_head.weight to embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # assign=True: replace meta tensors in-place rather than copy_ into them.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if missing:
        logger.warning("Missing keys after load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys after load: %s", unexpected)

    # RoPE inv_freq is a non-persistent buffer (not in checkpoint), so it's
    # still on the meta device after assign=True — materialize it now.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = 1.0 / (
                module.theta
                ** (
                    torch.arange(
                        0, module.head_dim, 2, device=device, dtype=torch.float32
                    )
                    / module.head_dim
                )
            )
    logger.info(
        "Weights loaded — %d parameters on %s (%s)",
        sum(p.numel() for p in model.parameters()),
        device,
        dtype,
    )
