import unittest

import torch

from miniengine.cuda_graph import CudaGraphRunner
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVMemoryPool


class _DummyGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _fake_graph_engine(max_running: int = 3, pages_per_seq: int = 2) -> Engine:
    engine = Engine.__new__(Engine)
    engine.device = "cpu"
    engine.max_running = max_running
    engine.max_decode_pages_per_seq = pages_per_seq
    engine.kv_pool = KVMemoryPool(
        num_pages=8,
        page_size=4,
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
        reserve_scratch_pages=max_running,
    )
    engine._dec_buffer_capacity = 0
    engine._dec_buffer_pages_per_seq = 0
    engine._dec_input_ids = None
    engine._dec_position_ids = None
    engine._dec_cache_seqlens = None
    engine._dec_block_table = None
    engine._dec_input_ids_cpu = None
    engine._dec_position_ids_cpu = None
    engine._dec_cache_seqlens_cpu = None
    engine._dec_block_table_cpu = None
    engine._allocate_decode_buffers(
        capacity=max_running,
        pages_per_seq=pages_per_seq,
    )
    return engine


class CudaGraphRunnerTest(unittest.TestCase):
    def test_seed_capture_inputs_use_distinct_scratch_pages(self) -> None:
        engine = _fake_graph_engine()
        runner = CudaGraphRunner(engine, bucket_sizes=[1, 2, 3], max_pages_per_seq=2)

        runner._seed_capture_inputs(3)

        scratch_pages = engine.kv_pool.scratch_page_indices[:3]
        self.assertEqual(engine._dec_cache_seqlens[:3].tolist(), [0, 0, 0])
        self.assertEqual(engine._dec_block_table[:3, 0].tolist(), scratch_pages)
        self.assertEqual(engine._dec_block_table[:3, 1].tolist(), scratch_pages)

    def test_replay_rejects_page_width_beyond_capture(self) -> None:
        engine = _fake_graph_engine()
        runner = CudaGraphRunner(engine, bucket_sizes=[1], max_pages_per_seq=2)
        runner._captured_buffer_ptrs = runner._current_buffer_ptrs()
        runner.graphs[1] = _DummyGraph()
        runner.logits_buffers[1] = torch.zeros(1, 1, 5)

        with self.assertRaisesRegex(RuntimeError, "page table width"):
            runner.replay(batch_size=1, pages_per_seq=3)

    def test_replay_rejects_reallocated_decode_buffers(self) -> None:
        engine = _fake_graph_engine()
        runner = CudaGraphRunner(engine, bucket_sizes=[1], max_pages_per_seq=2)
        runner._captured_buffer_ptrs = runner._current_buffer_ptrs()
        runner.graphs[1] = _DummyGraph()
        runner.logits_buffers[1] = torch.zeros(1, 1, 5)
        engine._allocate_decode_buffers(capacity=1, pages_per_seq=2)

        with self.assertRaisesRegex(RuntimeError, "decode buffers changed"):
            runner.replay(batch_size=1, pages_per_seq=2)


if __name__ == "__main__":
    unittest.main()
