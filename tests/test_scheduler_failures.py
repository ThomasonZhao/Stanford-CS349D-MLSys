import queue
import unittest

import torch

from miniengine.core import Request, RequestStatus
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.scheduler import Scheduler


class _FailingPagedEngine:
    def __init__(self) -> None:
        self.kv_pool = KVMemoryPool(
            num_pages=8,
            page_size=4,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            dtype=torch.float32,
            device="cpu",
        )

    def paged_prefill(self, requests: list[Request]) -> list[int]:
        raise RuntimeError("prefill failed")

    def paged_decode(self, requests: list[Request]) -> list[int]:
        raise RuntimeError("decode failed")

    def decode_token(self, token_id: int) -> str:
        return str(token_id)

    def is_stop_token(self, token_id: int) -> bool:
        return False


class SchedulerFailureTest(unittest.TestCase):
    def test_paged_prefill_failure_finishes_popped_request(self) -> None:
        engine = _FailingPagedEngine()
        scheduler = Scheduler(engine=engine, max_running=1, mode="paged")
        req = Request("a", input_ids=[1, 2, 3])
        scheduler.add_request(req)

        with self.assertRaisesRegex(RuntimeError, "prefill failed"):
            scheduler.step()

        output = req.token_queue.get_nowait()
        self.assertEqual(req.status, RequestStatus.FINISHED)
        self.assertTrue(output.finished)
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(scheduler.running, [])

    def test_paged_decode_failure_finishes_running_requests(self) -> None:
        engine = _FailingPagedEngine()
        scheduler = Scheduler(engine=engine, max_running=2, mode="paged")
        req = Request("a", input_ids=[1, 2], output_ids=[7])
        req.status = RequestStatus.RUNNING
        req.page_table = engine.kv_pool.allocate(1)
        req.cache_len = 2
        scheduler.running = [req]

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            scheduler.step()

        output = req.token_queue.get_nowait()
        self.assertEqual(req.status, RequestStatus.FINISHED)
        self.assertTrue(output.finished)
        self.assertEqual(req.page_table, [])
        self.assertEqual(scheduler.running, [])
        with self.assertRaises(queue.Empty):
            req.token_queue.get_nowait()


if __name__ == "__main__":
    unittest.main()
