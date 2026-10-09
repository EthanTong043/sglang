"""CPU regressions for the L4 (L2 -> storage/DFS) WRITE-side instrumentation.

Why this file exists: before this change the host-to-storage write path had a
completion counter (`sglang:backuped_tokens_total`, incremented at the backup-ack
drain) but no submission counter and no failure counter at all. The acceptance
rate -- L4 landed / L2 offered -- therefore had no numerator that could be
compared against anything, and a storage backend that rejected every batch looked
exactly like a backend that was never asked to write. That is the same shape as
the historical store-client incident where `--enable_offload` was off, pushes were
discarded silently, and every `*_failed_total` stayed at 0.

Four production bodies are executed here, AST-loaded with no torch / GPU / sglang
import, so the real statements run:

  1. HiCacheController._page_backup          -- batch loop; attributes
     `batch_rejected` for the pages abandoned by its `break`
  2. HiCacheController.backup_thread_func    -- daemon; attributes
     `backend_exception` instead of dying silently
  3. UnifiedRadixCache.log_storage_write_submitted -- tokens + bytes at submit
  4. UnifiedRadixCache.write_backup_storage  -- the cache-mode submission site

Testing fewer than all four would miss the original failure mode: an attribution
that nobody reports, or a report that nobody attributes. The submitted/completed
pair and the failure reason are asserted end-to-end.

Not covered: real transfers, real Mooncake IO, the Rust tree core (it returns the
same StorageBackupSpec and is not on the instrumented lines), and the per-cause
split of alloc-vs-IO-vs-timeout, which the backend collapses to a bool before the
controller can see it (see the 05-l4-write-metrics README).
"""

from __future__ import annotations

import ast
import runpy
import unittest
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"

register_cpu_ci = runpy.run_path(str(REPO / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE_SIZE = 4
STORAGE_BATCH_SIZE = 128


def _extract_function(path: Path, class_name: str, func_name: str, scope: dict):
    """Exec a single production method body into `scope` as a bare function.

    Keeps the real statements -- including the instrumentation -- so that
    deleting or inverting them in production makes these tests fail.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == func_name:
                child = ast.FunctionDef(
                    name=child.name,
                    args=child.args,
                    body=child.body,
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                    type_params=[],
                )
                mod = ast.Module(
                    body=ast.parse("from __future__ import annotations\n").body
                    + [child],
                    type_ignores=[],
                )
                exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), scope)
                return scope[func_name]
    raise AssertionError(f"{class_name}.{func_name} not found in {path}")


class _StorageMetricsCollector:
    """Records what the production code reports, with the real signatures."""

    def __init__(self):
        self.submitted = []  # (tokens, bytes)
        self.completed = []  # tokens
        self.failed = []  # (tokens, reason)
        self.durations = []

    def log_storage_write_submitted(self, num_tokens: int, num_bytes: int):
        self.submitted.append((num_tokens, num_bytes))

    def log_backuped_tokens(self, backuped_tokens: int):
        if backuped_tokens > 0:
            self.completed.append(backuped_tokens)

    def log_storage_write_failed_tokens(self, num_tokens: int, reason: str):
        if num_tokens > 0:
            self.failed.append((num_tokens, reason))

    def observe_storage_write_duration(self, duration_seconds: float):
        self.durations.append(duration_seconds)

    # -- helpers --
    def submitted_tokens(self) -> int:
        return sum(t for t, _ in self.submitted)

    def submitted_bytes(self) -> int:
        return sum(b for _, b in self.submitted)

    def failed_by_reason(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for num_tokens, reason in self.failed:
            totals[reason] = totals.get(reason, 0) + num_tokens
        return totals


class _Operation:
    """Stand-in for StorageOperation, with the production defaults."""

    def __init__(self, num_pages, *, op_id=0):
        self.id = op_id
        self.hash_value = [f"h{i}" for i in range(num_pages)]
        self.host_indices = list(range(num_pages * PAGE_SIZE))
        self.token_ids = list(range(num_pages * PAGE_SIZE))
        self.prefix_keys = None
        self.completed_tokens = 0
        # Production defaults from StorageOperation.__init__.
        self.failed_tokens = 0
        self.failed_reason = None
        self.write_duration_s = None


class _Controller:
    """Minimal host for the real HiCacheController backup bodies."""

    def __init__(self, *, set_results=None, raise_on_set=None, backup_skip=False):
        self.page_size = PAGE_SIZE
        self.backup_skip = backup_skip
        self._set_results = list(set_results or [])
        self._raise_on_set = raise_on_set
        self.set_calls = 0

        self.backup_queue = Queue()
        self.ack_backup_queue = Queue()
        self.storage_stop_event = SimpleNamespace(
            _set=False, is_set=lambda: self._stop_flag
        )
        self._stop_flag = False

        scope = {
            "STORAGE_BATCH_SIZE": STORAGE_BATCH_SIZE,
            "HiCacheStorageExtraInfo": lambda **kw: SimpleNamespace(**kw),
            "logger": SimpleNamespace(
                warning=lambda *a, **k: None,
                exception=lambda *a, **k: None,
            ),
            "Empty": Empty,
            "time": __import__("time"),
            "__name__": __name__,
        }
        self._page_backup_impl = _extract_function(
            SRT / "managers/cache_controller.py",
            "HiCacheController",
            "_page_backup",
            dict(scope),
        )
        self._thread_impl = _extract_function(
            SRT / "managers/cache_controller.py",
            "HiCacheController",
            "backup_thread_func",
            dict(scope),
        )

    def page_set_func(self, batch_hashes, batch_host_indices, extra_info):
        self.set_calls += 1
        if self._raise_on_set is not None and self.set_calls >= self._raise_on_set:
            raise RuntimeError("backend exploded")
        if not self._set_results:
            return True
        return self._set_results.pop(0)

    def _page_backup(self, operation):
        return self._page_backup_impl(self, operation)

    def run_daemon_once(self, operation):
        """Drive the real daemon body for exactly one queue item."""
        self.backup_queue.put(operation)
        self._stop_flag = False
        original_get = self.backup_queue.get

        def get(block=True, timeout=None):
            if self.backup_queue.empty():
                self._stop_flag = True
                raise Empty
            return original_get(block=block, timeout=timeout)

        self.backup_queue.get = get
        self._thread_impl(self)
        return list(self.ack_backup_queue.queue)


class _Cache:
    """Minimal host for the real UnifiedRadixCache submission bodies."""

    def __init__(
        self,
        *,
        size_per_token=55228,
        enable_storage=True,
        enable_storage_metrics=True,
        with_collector=True,
        spec=None,
    ):
        self.enable_storage = enable_storage
        self.enable_storage_metrics = enable_storage_metrics
        self.storage_metrics_collector = (
            _StorageMetricsCollector() if with_collector else None
        )
        self.hicache_storage_pass_prefix_keys = False
        self.ongoing_backup = {}
        self._spec = spec
        self.written = []

        self.cache_controller = SimpleNamespace(
            storage_host_pool=SimpleNamespace(size_per_token=size_per_token),
            write_storage=self._write_storage,
        )
        self.tree_core = SimpleNamespace(
            build_storage_backup_spec=lambda node_id, pass_prefix: self._spec
        )

        scope = {
            "PoolTransfer": lambda **kw: SimpleNamespace(**kw),
            "PoolName": SimpleNamespace(KV="kv"),
            "CacheTransferPhase": SimpleNamespace(BACKUP_STORAGE="backup_storage"),
            "__name__": __name__,
        }
        self.log_storage_write_submitted = _extract_function(
            SRT / "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "log_storage_write_submitted",
            dict(scope),
        ).__get__(self)
        self._write_backup_storage_impl = _extract_function(
            SRT / "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "write_backup_storage",
            dict(scope),
        )

    def _write_storage(self, *args, **kwargs):
        self.written.append((args, kwargs))
        return len(self.written) - 1

    def _build_sidecar_transfers(self, phase, kv_xfer, comp_xfers):
        return []

    def inc_host_lock_ref(self, node_id):
        return SimpleNamespace(to_dec_params=lambda: ("dec", node_id))

    def write_backup_storage(self, node_id):
        return self._write_backup_storage_impl(self, node_id)


def _spec(num_tokens):
    pages = num_tokens // PAGE_SIZE
    return SimpleNamespace(
        host_value=list(range(num_tokens)),
        token_ids=list(range(num_tokens)),
        hash_value=[f"h{i}" for i in range(pages)],
        prefix_keys=None,
        comp_xfers={},
    )


class TestPageBackupFailureAttribution(unittest.TestCase):
    """HiCacheController._page_backup: the `break` must not lose the count."""

    def test_all_batches_succeed_attributes_no_failure(self):
        ctl = _Controller()
        op = _Operation(num_pages=3)
        ctl._page_backup(op)
        self.assertEqual(op.completed_tokens, 3 * PAGE_SIZE)
        self.assertIsNone(op.failed_reason)
        self.assertEqual(op.failed_tokens, 0)

    def test_rejected_batch_is_attributed_to_batch_rejected(self):
        # One batch holds up to STORAGE_BATCH_SIZE pages, so a single falsy
        # result abandons the whole operation.
        ctl = _Controller(set_results=[False])
        op = _Operation(num_pages=5)
        ctl._page_backup(op)
        self.assertEqual(op.completed_tokens, 0)
        self.assertEqual(op.failed_reason, "batch_rejected")
        self.assertEqual(
            op.failed_tokens,
            5 * PAGE_SIZE,
            "every page of the abandoned operation must be counted, not just "
            "the pages of the failing batch",
        )

    def test_failed_plus_completed_covers_the_whole_operation(self):
        # The identity that makes the acceptance rate auditable: nothing in a
        # submitted operation may be silently unaccounted for.
        ctl = _Controller(set_results=[False])
        op = _Operation(num_pages=7)
        ctl._page_backup(op)
        self.assertEqual(
            op.completed_tokens + op.failed_tokens,
            7 * PAGE_SIZE,
        )

    def test_rejection_mid_stream_counts_every_abandoned_page(self):
        # Spans more than one batch so that the failing batch and the
        # abandoned remainder differ. Without this case, attributing only
        # len(batch_hashes) instead of the remaining pages is indistinguishable
        # -- a single-batch operation makes the two expressions equal.
        pages = STORAGE_BATCH_SIZE + 50
        ctl = _Controller(set_results=[True, False])
        op = _Operation(num_pages=pages)
        ctl._page_backup(op)

        # First batch landed; the second was rejected.
        self.assertEqual(op.completed_tokens, STORAGE_BATCH_SIZE * PAGE_SIZE)
        self.assertEqual(op.failed_reason, "batch_rejected")
        self.assertEqual(
            op.failed_tokens,
            50 * PAGE_SIZE,
            "the remainder abandoned by the break is 50 pages, which must not "
            "be confused with the failing batch's own size",
        )
        self.assertEqual(op.completed_tokens + op.failed_tokens, pages * PAGE_SIZE)

    def test_rejection_of_a_full_second_batch_is_not_undercounted(self):
        # Two full batches plus a tail: rejecting the second must attribute the
        # second batch AND the tail, i.e. strictly more than one batch.
        pages = 2 * STORAGE_BATCH_SIZE + 10
        ctl = _Controller(set_results=[True, False])
        op = _Operation(num_pages=pages)
        ctl._page_backup(op)
        self.assertEqual(
            op.failed_tokens,
            (STORAGE_BATCH_SIZE + 10) * PAGE_SIZE,
            "attributing only the failing batch would lose the 10-page tail",
        )
        self.assertGreater(op.failed_tokens, STORAGE_BATCH_SIZE * PAGE_SIZE)

    def test_write_duration_is_recorded(self):
        ctl = _Controller()
        op = _Operation(num_pages=2)
        ctl._page_backup(op)
        self.assertIsNotNone(op.write_duration_s)
        self.assertGreaterEqual(op.write_duration_s, 0.0)

    def test_empty_operation_attributes_nothing(self):
        ctl = _Controller()
        op = _Operation(num_pages=0)
        ctl._page_backup(op)
        self.assertEqual(op.completed_tokens, 0)
        self.assertIsNone(op.failed_reason)


class TestBackupDaemonSurvivesBackendException(unittest.TestCase):
    """backup_thread_func: an exception must be attributed AND acked."""

    def test_backend_exception_is_attributed_and_still_acked(self):
        ctl = _Controller(raise_on_set=1)
        op = _Operation(num_pages=4)
        acked = ctl.run_daemon_once(op)
        self.assertEqual(
            len(acked),
            1,
            "the operation must still be acked, otherwise the scheduler leaks "
            "the host lock and the drain never reports anything",
        )
        self.assertEqual(op.failed_reason, "backend_exception")
        self.assertEqual(op.failed_tokens, 4 * PAGE_SIZE)

    def test_successful_operation_is_acked_without_failure(self):
        ctl = _Controller()
        op = _Operation(num_pages=4)
        acked = ctl.run_daemon_once(op)
        self.assertEqual(len(acked), 1)
        self.assertIsNone(op.failed_reason)
        self.assertEqual(op.completed_tokens, 4 * PAGE_SIZE)

    def test_partial_progress_before_exception_is_not_double_counted(self):
        # Pages that already landed stay in completed_tokens; only the rest is
        # attributed as failed.
        ctl = _Controller(raise_on_set=2)
        op = _Operation(num_pages=STORAGE_BATCH_SIZE + 3)
        ctl.run_daemon_once(op)
        self.assertEqual(op.completed_tokens, STORAGE_BATCH_SIZE * PAGE_SIZE)
        self.assertEqual(op.failed_reason, "backend_exception")
        self.assertEqual(
            op.completed_tokens + op.failed_tokens,
            (STORAGE_BATCH_SIZE + 3) * PAGE_SIZE,
        )


class TestSubmissionInstrumentation(unittest.TestCase):
    """UnifiedRadixCache: tokens and bytes at the submission site."""

    def test_submitted_tokens_and_bytes_are_reported_separately(self):
        cache = _Cache(size_per_token=55228)
        cache.log_storage_write_submitted(1024)
        col = cache.storage_metrics_collector
        self.assertEqual(col.submitted_tokens(), 1024)
        self.assertEqual(col.submitted_bytes(), 1024 * 55228)

    def test_bytes_use_the_storage_host_pool_size_per_token(self):
        # Byte accounting must come from the pool actually written to L4, not
        # from a hardcoded constant.
        cache = _Cache(size_per_token=4096)
        cache.log_storage_write_submitted(10)
        self.assertEqual(cache.storage_metrics_collector.submitted_bytes(), 40960)

    def test_zero_tokens_produces_no_sample(self):
        cache = _Cache()
        cache.log_storage_write_submitted(0)
        self.assertEqual(cache.storage_metrics_collector.submitted, [])

    def test_no_collector_is_a_noop(self):
        cache = _Cache(with_collector=False)
        cache.log_storage_write_submitted(128)  # must not raise

    def test_metrics_disabled_is_a_noop(self):
        cache = _Cache(enable_storage_metrics=False)
        cache.log_storage_write_submitted(128)
        self.assertEqual(cache.storage_metrics_collector.submitted, [])

    def test_write_backup_storage_reports_submission(self):
        # End-to-end on the cache-mode submission site: the real
        # write_backup_storage body must both enqueue and instrument.
        cache = _Cache(spec=_spec(64))
        cache.write_backup_storage(node_id=7)
        self.assertEqual(len(cache.written), 1, "the write must still be enqueued")
        col = cache.storage_metrics_collector
        self.assertEqual(col.submitted_tokens(), 64)
        self.assertEqual(col.submitted_bytes(), 64 * 55228)

    def test_write_backup_storage_without_spec_reports_nothing(self):
        cache = _Cache(spec=None)
        cache.write_backup_storage(node_id=7)
        self.assertEqual(cache.written, [])
        self.assertEqual(cache.storage_metrics_collector.submitted, [])

    def test_storage_disabled_reports_nothing(self):
        cache = _Cache(spec=_spec(64), enable_storage=False)
        cache.write_backup_storage(node_id=7)
        self.assertEqual(cache.written, [])
        self.assertEqual(cache.storage_metrics_collector.submitted, [])


class TestAcceptanceRateIsComputable(unittest.TestCase):
    """The point of the whole change: a numerator that can be divided."""

    def test_full_success_yields_acceptance_one(self):
        cache = _Cache(spec=_spec(64))
        cache.write_backup_storage(node_id=1)
        col = cache.storage_metrics_collector

        ctl = _Controller()
        op = _Operation(num_pages=16)
        ctl._page_backup(op)
        col.log_backuped_tokens(op.completed_tokens)

        self.assertEqual(col.submitted_tokens(), 64)
        self.assertEqual(sum(col.completed), 64)
        self.assertEqual(sum(col.completed) / col.submitted_tokens(), 1.0)

    def test_silent_total_rejection_is_visible_as_zero_acceptance(self):
        # The historical incident: pushes discarded, every failure counter 0.
        # Submitted must be nonzero while completed stays 0, and the failure
        # counter must name a reason.
        cache = _Cache(spec=_spec(64))
        cache.write_backup_storage(node_id=1)
        col = cache.storage_metrics_collector

        ctl = _Controller(set_results=[False])
        op = _Operation(num_pages=16)
        ctl._page_backup(op)
        col.log_backuped_tokens(op.completed_tokens)
        col.log_storage_write_failed_tokens(op.failed_tokens, op.failed_reason)

        self.assertEqual(col.submitted_tokens(), 64)
        self.assertEqual(sum(col.completed), 0, "nothing landed")
        self.assertEqual(
            sum(col.completed) / col.submitted_tokens(),
            0.0,
            "acceptance rate must be computable and read 0 -- this is the "
            "detection path that did not exist before",
        )
        self.assertEqual(col.failed_by_reason(), {"batch_rejected": 64})


if __name__ == "__main__":
    unittest.main(verbosity=2)
