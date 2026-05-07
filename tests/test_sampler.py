import unittest

import torch

from miniengine.core import SamplingParams
from miniengine.sampler import sample_token, sample_tokens_batched


class BatchedSamplerTest(unittest.TestCase):
    def test_greedy_path_uses_one_argmax_for_batch(self) -> None:
        logits = torch.tensor(
            [
                [0.1, 0.9, 0.2],
                [2.0, 1.0, 3.0],
            ]
        )
        params = [
            SamplingParams(temperature=0),
            SamplingParams(temperature=0),
        ]

        self.assertEqual(sample_tokens_batched(logits, params, [[], []]), [1, 2])

    def test_repetition_penalty_falls_back_to_single_sampler(self) -> None:
        logits = torch.tensor([[5.0, 4.0, 1.0]])
        params = [SamplingParams(temperature=0, repetition_penalty=2.0)]
        output_ids = [[0]]

        self.assertEqual(
            sample_tokens_batched(logits, params, output_ids),
            [sample_token(logits, params[0], output_ids[0])],
        )

    def test_batch_size_mismatch_raises(self) -> None:
        logits = torch.tensor([[0.1, 0.9]])

        with self.assertRaisesRegex(ValueError, "matching batch size"):
            sample_tokens_batched(logits, [], [])


if __name__ == "__main__":
    unittest.main()
