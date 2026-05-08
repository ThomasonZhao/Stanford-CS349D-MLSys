import unittest

import torch

from miniengine.model import CausalLM, ModelConfig


def _model() -> CausalLM:
    torch.manual_seed(4321)
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return CausalLM(config).eval()


class SelectiveLogitsTest(unittest.TestCase):
    def test_logit_positions_match_full_lm_head_slice(self) -> None:
        model = _model()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        selected_positions = torch.tensor([1, 4], dtype=torch.long)

        full_logits, _ = model(input_ids, position_ids)
        selected_logits, _ = model(
            input_ids,
            position_ids,
            logit_positions=selected_positions,
        )

        torch.testing.assert_close(
            selected_logits,
            full_logits.index_select(1, selected_positions),
        )


if __name__ == "__main__":
    unittest.main()
