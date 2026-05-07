import unittest

import torch

from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, PagedContext


def _tiny_model() -> CausalLM:
    torch.manual_seed(1234)
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = CausalLM(config).eval()
    return model


def _greedy(logits: torch.Tensor) -> int:
    return int(logits.argmax(dim=-1).item())


def _baseline_generate(model: CausalLM, prompt: list[int], num_tokens: int) -> list[int]:
    input_ids = torch.tensor([prompt], dtype=torch.long)
    position_ids = torch.arange(len(prompt), dtype=torch.long).unsqueeze(0)

    logits, kv_caches = model(input_ids, position_ids)
    output_ids = [_greedy(logits[:, -1, :])]

    for _ in range(num_tokens - 1):
        cache_len = kv_caches[0][0].shape[2]
        input_ids = torch.tensor([[output_ids[-1]]], dtype=torch.long)
        position_ids = torch.tensor([[cache_len]], dtype=torch.long)
        logits, kv_caches = model(input_ids, position_ids, kv_caches=kv_caches)
        output_ids.append(_greedy(logits[:, -1, :]))

    return output_ids


def _make_pool(model: CausalLM, page_size: int) -> KVMemoryPool:
    config = model.config
    return KVMemoryPool(
        num_pages=64,
        page_size=page_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
    )


def _paged_generate_batch(
    model: CausalLM,
    prompts: list[list[int]],
    num_tokens: int,
    page_size: int,
) -> list[list[int]]:
    pool = _make_pool(model, page_size)
    page_tables = [pool.allocate(pool.pages_needed(len(prompt))) for prompt in prompts]
    cache_lens = [len(prompt) for prompt in prompts]

    flat_ids: list[int] = []
    flat_positions: list[int] = []
    slot_mapping: list[int] = []
    cu_values = [0]
    for prompt, page_table in zip(prompts, page_tables):
        flat_ids.extend(prompt)
        flat_positions.extend(range(len(prompt)))
        for pos in range(len(prompt)):
            page = page_table[pos // page_size]
            slot_mapping.append(page * page_size + pos % page_size)
        cu_values.append(cu_values[-1] + len(prompt))

    input_ids = torch.tensor([flat_ids], dtype=torch.long)
    position_ids = torch.tensor([flat_positions], dtype=torch.long)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int32)
    paged_ctx = PagedContext(
        k_cache_pool=pool.k_caches,
        v_cache_pool=pool.v_caches,
        block_table=None,
        cache_seqlens=None,
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(cache_lens),
        max_seqlen_k=max(cache_lens),
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long),
    )

    logits, _ = model(input_ids, position_ids, paged_ctx=paged_ctx)
    output_ids = [[_greedy(logits[:, end_idx - 1, :])] for end_idx in cu_values[1:]]

    for _ in range(num_tokens - 1):
        for idx, cache_len in enumerate(cache_lens):
            if cache_len % page_size == 0:
                page_tables[idx].append(pool.allocate(1)[0])

        max_pages = max(len(page_table) for page_table in page_tables)
        block_table = torch.tensor(
            [
                page_table + [0] * (max_pages - len(page_table))
                for page_table in page_tables
            ],
            dtype=torch.int32,
        )
        cache_seqlens = torch.tensor(cache_lens, dtype=torch.int32)
        decode_slots = [
            page_table[cache_len // page_size] * page_size + cache_len % page_size
            for page_table, cache_len in zip(page_tables, cache_lens)
        ]
        input_ids = torch.tensor(
            [[tokens[-1]] for tokens in output_ids], dtype=torch.long
        )
        position_ids = torch.tensor([[length] for length in cache_lens], dtype=torch.long)
        paged_ctx = PagedContext(
            k_cache_pool=pool.k_caches,
            v_cache_pool=pool.v_caches,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            is_prefill=False,
            slot_mapping=torch.tensor(decode_slots, dtype=torch.long),
        )

        logits, _ = model(input_ids, position_ids, paged_ctx=paged_ctx)
        for idx in range(len(prompts)):
            output_ids[idx].append(_greedy(logits[idx : idx + 1, -1, :]))
            cache_lens[idx] += 1

    return output_ids


class PagedParityTest(unittest.TestCase):
    def test_paged_greedy_matches_baseline_for_fixed_prompts(self) -> None:
        torch.set_num_threads(1)
        model = _tiny_model()
        prompts = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9, 10],
        ]
        expected = [_baseline_generate(model, prompt, num_tokens=5) for prompt in prompts]
        actual = _paged_generate_batch(model, prompts, num_tokens=5, page_size=2)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
