import unittest

import torch

from miniengine.core import Request, SamplingParams
from miniengine.engine import Engine
from miniengine.model import CausalLM, ModelConfig, RotaryEmbedding


class RopeCacheTest(unittest.TestCase):
    def test_rotary_embedding_uses_bounded_cache(self) -> None:
        rope = RotaryEmbedding(head_dim=4, max_positions=8)
        position_ids = torch.tensor([[0, 3, 7]], dtype=torch.long)

        cos, sin = rope(position_ids)

        self.assertEqual(rope._cos.shape, (8, 4))
        self.assertEqual(rope._sin.shape, (8, 4))
        self.assertEqual(cos.shape, (1, 3, 1, 4))
        self.assertEqual(sin.shape, (1, 3, 1, 4))

    def test_engine_rejects_requests_beyond_rope_cache_cap(self) -> None:
        config = ModelConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=4,
            tie_word_embeddings=False,
        )
        engine = Engine.__new__(Engine)
        engine.model = CausalLM(config)
        request = Request(
            "too-long",
            input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_new_tokens=2),
        )

        with self.assertRaisesRegex(ValueError, "RoPE cache cap"):
            engine._validate_rope_budget(request)

    def test_meta_model_construction_keeps_rope_cache_on_meta(self) -> None:
        config = ModelConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )

        with torch.device("meta"):
            model = CausalLM(config)

        self.assertEqual(model.model.rotary_emb._cos.device.type, "meta")
        self.assertEqual(model.model.rotary_emb._cos.shape, (16, 4))


if __name__ == "__main__":
    unittest.main()
