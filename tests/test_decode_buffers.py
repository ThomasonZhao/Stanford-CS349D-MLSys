import unittest
from types import SimpleNamespace

from miniengine.core import Request
from miniengine.engine import Engine


def _fake_engine(max_running: int = 2, max_pages: int = 2) -> Engine:
    engine = Engine.__new__(Engine)
    engine.device = "cpu"
    engine.max_running = max_running
    engine.max_decode_pages_per_seq = max_pages
    engine._dec_buffer_capacity = 0
    engine._dec_buffer_pages_per_seq = 0
    engine._dec_input_ids = None
    engine._dec_position_ids = None
    engine._dec_cache_seqlens = None
    engine._dec_block_table = None
    engine.cuda_graph_runner = None
    engine._dec_input_ids_cpu = None
    engine._dec_position_ids_cpu = None
    engine._dec_cache_seqlens_cpu = None
    engine._dec_block_table_cpu = None
    engine._allocate_decode_buffers(capacity=max_running, pages_per_seq=max_pages)
    return engine


class DecodeBufferTest(unittest.TestCase):
    def test_prepare_paged_decode_inputs_reuses_buffers(self) -> None:
        engine = _fake_engine()
        requests = [
            Request("a", output_ids=[11], page_table=[3], cache_len=1),
            Request("b", output_ids=[12], page_table=[1, 2], cache_len=5),
        ]

        first_input_buffer = engine._dec_input_ids
        block_table, cache_seqlens, input_ids, position_ids = (
            engine._prepare_paged_decode_inputs(requests)
        )
        second = engine._prepare_paged_decode_inputs(requests)

        self.assertIs(engine._dec_input_ids, first_input_buffer)
        self.assertEqual(input_ids.tolist(), [[11], [12]])
        self.assertEqual(position_ids.tolist(), [[1], [5]])
        self.assertEqual(cache_seqlens.tolist(), [1, 5])
        self.assertEqual(block_table.tolist(), [[3, 0], [1, 2]])
        self.assertEqual(second[2].tolist(), [[11], [12]])

    def test_prepare_paged_decode_inputs_grows_page_capacity(self) -> None:
        engine = _fake_engine(max_running=1, max_pages=1)
        request = Request("a", output_ids=[11], page_table=[3, 4], cache_len=5)

        block_table, _, _, _ = engine._prepare_paged_decode_inputs([request])

        self.assertGreaterEqual(engine._dec_buffer_pages_per_seq, 2)
        self.assertEqual(block_table.tolist(), [[3, 4]])

    def test_graph_active_decode_buffers_do_not_reallocate(self) -> None:
        engine = _fake_engine(max_running=1, max_pages=1)
        engine.cuda_graph_runner = SimpleNamespace()
        request = Request("a", output_ids=[11], page_table=[3, 4], cache_len=5)

        with self.assertRaisesRegex(RuntimeError, "cannot grow after CUDA graph"):
            engine._prepare_paged_decode_inputs([request])

        self.assertEqual(engine._dec_buffer_capacity, 1)
        self.assertEqual(engine._dec_buffer_pages_per_seq, 1)


if __name__ == "__main__":
    unittest.main()
