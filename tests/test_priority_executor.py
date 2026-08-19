import threading
import time
import unittest
from concurrent.futures import CancelledError, Future

from contiguous_fuxian.priority_executor import PriorityExecutor


class PriorityExecutorTest(unittest.TestCase):
    def test_priority_overtakes_queued_work_and_ties_are_fifo(self):
        executor = PriorityExecutor()
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        execution_order = []

        def blocker():
            blocker_started.set()
            self.assertTrue(release_blocker.wait(timeout=2))

        def record(name):
            execution_order.append(name)
            return name

        blocker_future = executor.submit(blocker, priority=0, label="blocker")
        self.assertTrue(blocker_started.wait(timeout=2))
        low_one = executor.submit(record, "low-one", priority=10, label="low")
        low_two = executor.submit(record, "low-two", priority=10, label="low")
        high = executor.submit(record, "high", priority=1, label="high")

        release_blocker.set()
        executor.shutdown(wait=True)

        self.assertIsInstance(high, Future)
        self.assertIsNone(blocker_future.result())
        self.assertEqual(high.result(), "high")
        self.assertEqual(low_one.result(), "low-one")
        self.assertEqual(low_two.result(), "low-two")
        self.assertEqual(execution_order, ["high", "low-one", "low-two"])

    def test_cancelled_queued_work_never_runs(self):
        executor = PriorityExecutor()
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        cancelled_task_ran = threading.Event()

        def blocker():
            blocker_started.set()
            release_blocker.wait(timeout=2)

        executor.submit(blocker, label="blocker")
        self.assertTrue(blocker_started.wait(timeout=2))
        cancelled = executor.submit(
            cancelled_task_ran.set,
            priority=5,
            label="cancelled-work",
        )

        self.assertTrue(cancelled.cancel())
        release_blocker.set()
        executor.shutdown(wait=True)

        self.assertTrue(cancelled.cancelled())
        with self.assertRaises(CancelledError):
            cancelled.result()
        self.assertFalse(cancelled_task_ran.is_set())
        metrics = executor.metrics_snapshot()["cancelled-work"]
        self.assertEqual(metrics["submitted"], 1)
        self.assertEqual(metrics["started"], 0)
        self.assertEqual(metrics["completed"], 0)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["cancelled"], 1)
        self.assertEqual(metrics["queue_wait_ms"], 0)
        self.assertEqual(metrics["execution_ms"], 0)

    def test_exception_is_propagated_and_recorded(self):
        executor = PriorityExecutor()

        def fail():
            raise ValueError("expected failure")

        failed = executor.submit(fail, label="failure")
        with self.assertRaisesRegex(ValueError, "expected failure"):
            failed.result(timeout=2)
        executor.shutdown(wait=True)

        metrics = executor.metrics_snapshot()["failure"]
        self.assertEqual(metrics["submitted"], 1)
        self.assertEqual(metrics["started"], 1)
        self.assertEqual(metrics["completed"], 0)
        self.assertEqual(metrics["failed"], 1)
        self.assertEqual(metrics["cancelled"], 0)
        self.assertGreaterEqual(metrics["queue_wait_ms"], 0)
        self.assertGreaterEqual(metrics["execution_ms"], 0)

    def test_metrics_are_per_label_detached_and_shutdown_rejects_work(self):
        executor = PriorityExecutor()
        first = executor.submit(lambda: 3, label="reads")
        second = executor.submit(lambda value: value + 1, 3, label="reads")
        write = executor.submit(time.sleep, 0.001, label="writes")
        executor.shutdown(wait=True)

        self.assertEqual(first.result(), 3)
        self.assertEqual(second.result(), 4)
        self.assertIsNone(write.result())

        snapshot = executor.metrics_snapshot()
        self.assertEqual(
            set(snapshot["reads"]),
            {
                "submitted",
                "started",
                "completed",
                "failed",
                "cancelled",
                "queue_wait_ms",
                "execution_ms",
            },
        )
        self.assertEqual(snapshot["reads"]["submitted"], 2)
        self.assertEqual(snapshot["reads"]["started"], 2)
        self.assertEqual(snapshot["reads"]["completed"], 2)
        self.assertEqual(snapshot["writes"]["submitted"], 1)
        self.assertEqual(snapshot["writes"]["completed"], 1)
        self.assertGreaterEqual(snapshot["reads"]["queue_wait_ms"], 0)
        self.assertGreater(snapshot["writes"]["execution_ms"], 0)

        snapshot["reads"]["submitted"] = -1
        self.assertEqual(
            executor.metrics_snapshot()["reads"]["submitted"],
            2,
        )
        with self.assertRaisesRegex(RuntimeError, "after shutdown"):
            executor.submit(lambda: None)


if __name__ == "__main__":
    unittest.main()
