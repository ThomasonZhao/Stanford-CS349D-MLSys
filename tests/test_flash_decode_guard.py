import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

import miniengine.model as model_module
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import Attention, ModelConfig, PagedContext


class FlashDecodeGuardTest(unittest.TestCase):
    def test_cuda_graph_uses_exact_buckets_up_to_max_running(self) -> None:
        self.assertEqual(Engine._cuda_graph_exact_batch_sizes(4), [1, 2, 3, 4])

    def test_cuda_graph_requires_paged_cuda_flash_and_aligned_pages(self) -> None:
        with self.assertRaisesRegex(ValueError, "paged mode"):
            Engine._validate_cuda_graph_config(
                mode="batched",
                device="cuda",
                page_size=256,
            )

        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA device"):
                Engine._validate_cuda_graph_config(
                    mode="paged",
                    device="cuda",
                    page_size=256,
                )

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(model_module, "_flash_attn_with_kvcache", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "flash-attn"):
                Engine._validate_cuda_graph_config(
                    mode="paged",
                    device="cuda",
                    page_size=256,
                )

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                model_module,
                "_flash_attn_with_kvcache",
                Mock(),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "divisible by 256"):
                Engine._validate_cuda_graph_config(
                    mode="paged",
                    device="cuda",
                    page_size=32,
                )

    def test_flash_decode_failure_disables_future_attempts(self) -> None:
        config = ModelConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        attention = Attention(config)
        pool = KVMemoryPool(
            num_pages=4,
            page_size=2,
            num_layers=1,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        ctx = PagedContext(
            k_cache_pool=pool.k_caches,
            v_cache_pool=pool.v_caches,
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
            cache_seqlens=torch.tensor([0, 0], dtype=torch.int32),
            is_prefill=False,
            slot_mapping=None,
        )
        q = torch.randn(2, 1, config.num_attention_heads, config.head_dim)
        k = torch.randn(2, 1, config.num_key_value_heads, config.head_dim)
        v = torch.randn(2, 1, config.num_key_value_heads, config.head_dim)

        original_flash = model_module._flash_attn_with_kvcache
        original_disabled = model_module._flash_decode_disabled
        original_fallback_logged = model_module._Attention_flash_decode_fallback_logged[
            0
        ]
        flash = Mock(side_effect=RuntimeError("boom"))
        try:
            model_module._flash_attn_with_kvcache = flash
            model_module._flash_decode_disabled = False
            model_module._Attention_flash_decode_fallback_logged[0] = False
            with patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ):
                with self.assertLogs("miniengine.model", level="WARNING") as logs:
                    attention._paged_decode_attention(q, k, v, ctx, layer_idx=0)
                attention._paged_decode_attention(q, k, v, ctx, layer_idx=0)

            self.assertTrue(model_module._flash_decode_disabled)
            self.assertEqual(flash.call_count, 1)
            self.assertIn("disabling flash decode", "\n".join(logs.output))
        finally:
            model_module._flash_attn_with_kvcache = original_flash
            model_module._flash_decode_disabled = original_disabled
            model_module._Attention_flash_decode_fallback_logged[
                0
            ] = original_fallback_logged

    def test_sdpa_fallback_zeros_invalid_gathered_kv(self) -> None:
        config = ModelConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        attention = Attention(config)
        pool = KVMemoryPool(
            num_pages=4,
            page_size=2,
            num_layers=1,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        for cache in pool.k_caches + pool.v_caches:
            cache.fill_(float("nan"))
            cache[0].zero_()

        ctx = PagedContext(
            k_cache_pool=pool.k_caches,
            v_cache_pool=pool.v_caches,
            block_table=torch.tensor([[0, 1], [2, 2]], dtype=torch.int32),
            cache_seqlens=torch.tensor([2, 0], dtype=torch.int32),
            is_prefill=False,
            slot_mapping=None,
        )
        torch.manual_seed(1234)
        q = torch.randn(2, 1, config.num_attention_heads, config.head_dim)
        k = torch.randn(2, 1, config.num_key_value_heads, config.head_dim)
        v = torch.randn(2, 1, config.num_key_value_heads, config.head_dim)

        out = attention._paged_decode_attention(q, k, v, ctx, layer_idx=0)

        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
