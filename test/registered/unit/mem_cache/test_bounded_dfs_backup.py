"""CPU regressions for bounded primary DFS backup and its lifetime boundary.

Execute production AST bodies without CUDA/native imports. These tests prove
Python concurrency, prefix and stop/ACK ordering; native client thread safety,
GPFS throughput, real collectives and host transfer correctness need canaries.
"""

from __future__ import annotations

import ast
import logging
import runpy
import threading
import time
import unittest
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"
CONTROLLER = SRT / "managers/cache_controller.py"
HYBRID = SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py"

register_cpu_ci = runpy.run_path(str(REPO / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def load_classes(path, names, scope):
    body = ast.parse("from __future__ import annotations\n").body
    body += [
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)


SCOPE = {
    "threading": threading, "time": time, "deque": deque,
    "ThreadPoolExecutor": ThreadPoolExecutor, "Queue": Queue, "Empty": Empty,
    "logger": logging.getLogger("bounded-backup-test"), "STORAGE_BATCH_SIZE": 128,
    "HiCacheStorageExtraInfo": SimpleNamespace, "trace_event": lambda *a, **kw: None,
}
load_classes(CONTROLLER, {"StorageOperation", "HiCacheController"}, SCOPE)
SCOPE["BaseHiCacheController"] = SCOPE["HiCacheController"]
load_classes(HYBRID, {"HybridCacheController"}, SCOPE)
Controller = SCOPE["HiCacheController"]
HybridController = SCOPE["HybridCacheController"]
Operation = SCOPE["StorageOperation"]


class BoundedBackupTests(unittest.TestCase):
    def controller(self, write, workers=2, cls=Controller):
        controller = cls.__new__(cls)
        controller.storage_backend_type = "mooncake"
        controller.storage_config = SimpleNamespace(extra_config={"backup_io_workers": workers})
        controller.storage_backend = SimpleNamespace(_dfs_only=True, batch_set_v1=write)
        controller.mem_pool_host = SimpleNamespace(
            layout="page_first_direct", size_per_token=12096, free=Mock()
        )
        controller.io_backend = "direct"
        controller.page_size = 64
        controller.page_set_func = controller._page_set_zero_copy
        controller.has_draft = False
        controller.backup_skip = False
        controller.enable_storage = True
        controller.storage_stop_event = threading.Event()
        controller._backup_executor = None
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.events = []
        controller._record_storage_backup_state = lambda operation, state: controller.events.append(state)
        controller._configure_backup_executor()
        controller._start_backup_executor()
        self.addCleanup(controller._shutdown_backup_executor)
        return controller

    @staticmethod
    def operation(pages=384, prefix=None):
        return Operation(
            list(range(pages * 64)), [],
            hash_value=[str(index) for index in range(pages)], prefix_keys=prefix,
        )

    def start(self, function):
        outcome = SimpleNamespace(error=None, done=threading.Event())

        def invoke():
            try:
                function()
            except BaseException as exc:
                outcome.error = exc
            finally:
                outcome.done.set()

        thread = threading.Thread(target=invoke, daemon=True)
        thread.start()
        return thread, outcome

    def finish(self, thread, outcome, error=None):
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive(), "coordinator did not drain")
        if error is None:
            self.assertIsNone(outcome.error)
        else:
            self.assertIsInstance(outcome.error, error)

    @staticmethod
    def wait(event):
        if not event.wait(timeout=2):
            raise AssertionError("test barrier timed out")

    def test_actual_overlap_and_pending_are_bounded_at_two(self):
        entered = [threading.Event() for _ in range(3)]
        release = threading.Event()
        self.addCleanup(release.set)
        snapshots = {}
        caller_prefix = ["parent"]

        def write(hashes, indices, extra):
            start = int(hashes[0])
            snapshots[start] = list(extra.prefix_keys)
            extra.prefix_keys.append("worker-private")
            entered[start // 128].set()
            if start < 256:
                self.wait(release)
            self.assertEqual(indices[0], start * 64)
            return [True] * len(hashes)

        controller = self.controller(write)
        operation = self.operation(prefix=caller_prefix)
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            self.wait(entered[0])
            self.wait(entered[1])
            self.assertFalse(entered[2].is_set())
            self.assertFalse(outcome.done.is_set())
        finally:
            release.set()
        self.finish(thread, outcome)
        self.assertEqual(operation.completed_tokens, 384 * 64)
        self.assertEqual(operation.backup_io_stats["peak_active_batches"], 2)
        self.assertEqual(operation.backup_io_stats["peak_pending_batches"], 2)
        self.assertEqual(operation.backup_io_stats["primary_bytes_limit"], 198180864)
        self.assertEqual(caller_prefix, ["parent"])
        self.assertEqual(snapshots[128], ["parent"] + operation.hash_value[:128])
        self.assertEqual(snapshots[256], ["parent"] + operation.hash_value[:256])
        controller.mem_pool_host.free.assert_not_called()

    def test_later_failure_stops_submission_and_only_confirms_prefix(self):
        release = threading.Event()
        later_failed = threading.Event()
        self.addCleanup(release.set)
        called = []

        def write(hashes, *_):
            start = int(hashes[0])
            called.append(start)
            if start == 0:
                self.wait(release)
                return [True] * len(hashes)
            return [True] + [False] * (len(hashes) - 1)

        controller = self.controller(write)
        original_worker = controller._backup_primary_batch

        def observed_worker(*args):
            results = original_worker(*args)
            if args[1] == 128:
                later_failed.set()
            return results

        controller._backup_primary_batch = observed_worker
        operation = self.operation()
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            self.wait(later_failed)
            self.assertFalse(outcome.done.is_set())
        finally:
            release.set()
        self.finish(thread, outcome)
        self.assertEqual(sorted(called), [0, 128])
        self.assertEqual(operation.completed_tokens, 128 * 64)
        self.assertEqual(operation.backup_batch_results[1]["results"][0], True)
        self.assertEqual(operation.backup_io_stats["unconfirmed_successful_pages"], 1)

    def test_success_after_a_failed_prefix_is_preserved_but_not_acknowledged(self):
        release = threading.Event()
        later_done = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            if hashes[0] == "0":
                self.wait(release)
                return [False] * len(hashes)
            later_done.set()
            return [True] * len(hashes)

        controller = self.controller(write)
        operation = self.operation()
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            self.wait(later_done)
        finally:
            release.set()
        self.finish(thread, outcome)
        self.assertEqual(operation.completed_tokens, 0)
        self.assertEqual(operation.backup_io_stats["successful_pages"], 128)
        self.assertEqual(operation.backup_io_stats["unconfirmed_successful_pages"], 128)
        self.assertEqual(len(operation.backup_batch_results), 2)

    def test_worker_exception_drains_before_no_ack_abort(self):
        release = threading.Event()
        raising = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            if hashes[0] == "0":
                self.wait(release)
                return [True] * len(hashes)
            raising.set()
            raise RuntimeError("native failure")

        controller = self.controller(write)
        operation = self.operation()
        controller.backup_queue.put(operation)
        thread, outcome = self.start(controller.backup_thread_func)
        try:
            self.wait(raising)
            self.assertFalse(outcome.done.is_set())
            self.assertTrue(controller.ack_backup_queue.empty())
            self.assertEqual(controller.events, ["started"])
        finally:
            release.set()
        self.finish(thread, outcome, RuntimeError)
        self.assertEqual(controller.events, ["started", "aborted"])
        self.assertTrue(controller.ack_backup_queue.empty())
        self.assertEqual(operation.backup_io_stats["active_batches"], 0)
        controller.mem_pool_host.free.assert_not_called()

    def test_detach_waits_for_every_worker_before_backend_close(self):
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        for event in release:
            self.addCleanup(event.set)
        completed = []
        closed = []

        def write(hashes, *_):
            index = int(hashes[0]) // 128
            entered[index].set()
            self.wait(release[index])
            completed.append(index)
            return [True] * len(hashes)

        controller = self.controller(write)
        controller.storage_backend.close = lambda: closed.append(tuple(completed))
        controller._destroy_prefetch_sync_groups = lambda: None
        operation = self.operation()
        worker, worker_outcome = self.start(lambda: controller._page_backup(operation))
        controller.backup_thread = worker
        try:
            for event in entered:
                self.wait(event)
            detach, outcome = self.start(controller.detach_storage_backend)
            self.wait(controller.storage_stop_event)
            release[0].set()
            self.assertFalse(outcome.done.is_set())
            self.assertFalse(closed)
        finally:
            for event in release:
                event.set()
        self.finish(worker, worker_outcome)
        self.finish(detach, outcome)
        self.assertEqual(set(closed[0]), {0, 1})
        self.assertEqual(operation.backup_io_stats["submitted_batches"], 2)
        self.assertIsNone(controller._backup_executor)
        self.assertFalse(controller.enable_storage)

    def test_timeout_does_not_close_backend_or_executor(self):
        controller = self.controller(lambda hashes, *_: [True] * len(hashes))
        executor = controller._backup_executor
        controller.backup_thread = SimpleNamespace(join=lambda **_: None, is_alive=lambda: True)
        controller.storage_backend.close = Mock()
        with self.assertRaisesRegex(RuntimeError, "detach aborted"):
            controller.detach_storage_backend()
        self.assertIs(controller._backup_executor, executor)
        controller.storage_backend.close.assert_not_called()

    def test_hybrid_sidecar_precedes_every_primary_and_draft_stays_coordinator(self):
        order = []
        coordinator = threading.get_ident()

        def write(hashes, *_):
            self.assertEqual(order[0], "sidecar")
            return [True] * len(hashes)

        controller = self.controller(write, cls=HybridController)
        controller.should_backup = lambda _: True
        controller._resolve_sidecar_derived_pool_transfers = lambda _: None
        controller.storage_backend.batch_set_v2 = lambda _: order.append("sidecar") or {}
        controller.has_draft = True
        controller._draft_page_set = lambda hashes, _: order.append((int(hashes[0]), threading.get_ident()))
        operation = self.operation()
        operation.pool_transfers = [SimpleNamespace()]
        operation.pool_storage_result = SimpleNamespace(update_extra_pool_hit_pages=lambda _: None)
        controller._page_backup(operation)
        self.assertEqual(order, ["sidecar", (0, coordinator), (128, coordinator), (256, coordinator)])
        self.assertEqual(operation.completed_tokens, 384 * 64)

    def test_b1_preserves_legacy_path_and_creates_no_executor(self):
        controller = self.controller(lambda *a: self.fail("must use original page_set_func"), workers=1)
        calls = []
        controller.page_set_func = lambda hashes, *_: calls.append(int(hashes[0])) or len(calls) == 1
        operation = self.operation()
        controller._page_backup(operation)
        self.assertIsNone(controller._backup_executor)
        self.assertFalse(hasattr(operation, "backup_io_stats"))
        self.assertEqual(calls, [0, 128])
        self.assertEqual(operation.completed_tokens, 128 * 64)

    def test_malformed_v1_vector_never_acknowledges_batch(self):
        for results in ([], [True], None, [-1] * 128, ["failed"] * 128):
            with self.subTest(results=results):
                controller = self.controller(lambda *a: results)
                operation = self.operation(pages=128)
                controller._page_backup(operation)
                self.assertEqual(operation.completed_tokens, 0)
                self.assertIsNone(operation.backup_batch_results[0]["results"])

    def test_four_workers_have_real_bounded_overlap(self):
        entered = [threading.Event() for _ in range(4)]
        release = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            index = int(hashes[0]) // 128
            if index < 4:
                entered[index].set()
                self.wait(release)
            return [True] * len(hashes)

        controller = self.controller(write, workers=4)
        operation = self.operation(pages=640)
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            for event in entered:
                self.wait(event)
            self.assertEqual(operation.backup_io_stats["peak_active_batches"], 4)
        finally:
            release.set()
        self.finish(thread, outcome)
        self.assertEqual(operation.completed_tokens, 640 * 64)
        self.assertEqual(operation.backup_io_stats["peak_active_batches"], 4)
        self.assertEqual(operation.backup_io_stats["peak_pending_batches"], 4)
        self.assertEqual(operation.backup_io_stats["submitted_batches"], 5)

    def test_submission_failure_drains_existing_future(self):
        entered = threading.Event()
        release = threading.Event()
        submission_failed = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            entered.set()
            self.wait(release)
            return [True] * len(hashes)

        controller = self.controller(write)
        original_submit = controller._backup_executor.submit
        submitted = []

        def submit(*args):
            if submitted:
                submission_failed.set()
                raise RuntimeError("executor submit failed")
            submitted.append(True)
            return original_submit(*args)

        controller._backup_executor.submit = submit
        operation = self.operation()
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            self.wait(entered)
            self.wait(submission_failed)
            self.assertFalse(outcome.done.is_set())
        finally:
            release.set()
        self.finish(thread, outcome, RuntimeError)
        self.assertEqual(operation.completed_tokens, 0)
        self.assertEqual(operation.backup_io_stats["active_batches"], 0)
        self.assertEqual(operation.backup_io_stats["successful_pages"], 128)
        self.assertEqual(operation.backup_io_stats["unconfirmed_successful_pages"], 128)

    def test_coordinator_draft_exception_drains_other_inflight_write(self):
        later_entered = threading.Event()
        draft_failed = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            if hashes[0] == "0":
                self.wait(later_entered)
            else:
                later_entered.set()
                self.wait(release)
            return [True] * len(hashes)

        def draft(*_):
            draft_failed.set()
            raise RuntimeError("unexpected draft failure")

        controller = self.controller(write)
        controller.has_draft = True
        controller._draft_page_set = draft
        operation = self.operation()
        thread, outcome = self.start(lambda: controller._page_backup(operation))
        try:
            self.wait(draft_failed)
            self.assertFalse(outcome.done.is_set())
        finally:
            release.set()
        self.finish(thread, outcome, RuntimeError)
        self.assertEqual(operation.completed_tokens, 0)
        self.assertEqual(operation.backup_io_stats["successful_pages"], 256)
        self.assertEqual(operation.backup_io_stats["active_batches"], 0)

    def test_operation_ack_remains_fifo_and_after_workers_finish(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def write(hashes, *_):
            if hashes[0] == "0":
                entered.set()
                self.wait(release)
            return [True] * len(hashes)

        controller = self.controller(write)
        operations = [self.operation(), self.operation(pages=64)]
        for operation in operations:
            controller.backup_queue.put(operation)
        original_get = controller.backup_queue.get

        def next_operation(*args, **kwargs):
            if controller.backup_queue.empty():
                controller.storage_stop_event.set()
                return None
            return original_get(*args, **kwargs)

        controller.backup_queue.get = next_operation
        thread, outcome = self.start(controller.backup_thread_func)
        try:
            self.wait(entered)
            self.assertTrue(controller.ack_backup_queue.empty())
        finally:
            release.set()
        self.finish(thread, outcome)
        self.assertEqual(list(controller.ack_backup_queue.queue), operations)
        self.assertEqual(controller.events, ["started", "finished", "started", "finished"])
        for operation in operations:
            self.assertEqual(operation.backup_io_stats["active_batches"], 0)
        controller.mem_pool_host.free.assert_not_called()

    def test_reset_replaces_executor_after_old_coordinator_is_joined(self):
        controller = self.controller(lambda hashes, *_: [True] * len(hashes))
        original = controller._backup_executor
        order = []
        for name in ("write_queue", "load_queue", "ack_write_queue", "ack_load_queue"):
            setattr(controller, name, [])
        for name in ("prefetch_queue", "prefetch_hit_queue", "host_mem_release_queue"):
            setattr(controller, name, Queue())
        controller.prefetch_thread = SimpleNamespace(join=lambda: order.append("prefetch_join"))
        controller.backup_thread = SimpleNamespace(join=lambda: order.append("backup_join"))
        controller.prefetch_thread_func = lambda: None
        original_shutdown = original.shutdown

        def shutdown(**kwargs):
            order.append("executor_shutdown")
            original_shutdown(**kwargs)

        original.shutdown = shutdown
        with patch.object(threading, "Thread", return_value=SimpleNamespace(start=lambda: None)):
            controller.reset()
        self.assertEqual(order, ["prefetch_join", "backup_join", "executor_shutdown"])
        self.assertIsNot(controller._backup_executor, original)
        self.assertFalse(controller.storage_stop_event.is_set())

    def test_configuration_is_explicit_and_fails_closed(self):
        controller = self.controller(lambda hashes, *_: [True] * len(hashes), workers=1)
        for value in (True, 0, 3, 8, "2", 2.0):
            controller.storage_config.extra_config = {"backup_io_workers": value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                controller._configure_backup_executor()
        controller.storage_config.extra_config = {"backup_io_workers": 2}
        for target, attribute, value in (
            (controller, "storage_backend_type", "file"),
            (controller.storage_backend, "_dfs_only", False),
            (controller, "io_backend", "kernel"),
            (controller.mem_pool_host, "layout", "page_first"),
        ):
            original = getattr(target, attribute)
            setattr(target, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                controller._configure_backup_executor()
            setattr(target, attribute, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
