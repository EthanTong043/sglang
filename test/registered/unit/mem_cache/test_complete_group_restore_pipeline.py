"""CPU regressions for the pipelined complete-group DFS restore.

Runs the production ``_page_transfer_complete_groups_pipelined`` (and the
serial ``_page_transfer_complete_groups`` as the reference) extracted from this
checkout without importing Torch. Storage is a model that records call order
and overlap; it does not validate real Mooncake I/O or GPU transfers.
"""

from __future__ import annotations

import ast
import runpy
import threading
import time
import unittest
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"

register_cpu_ci = runpy.run_path(
    str(REPO / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE = 4
BATCH = 2  # pages per group, stands in for STORAGE_BATCH_SIZE


class PoolName(str, Enum):
    KV = "kv"
    INDEXER = "indexer"


class PoolHitPolicy(str, Enum):
    ALL_PAGES = "all_pages"


@dataclass
class PoolTransfer:
    name: PoolName
    host_indices: Optional[list] = None
    keys: Optional[List[str]] = None
    indices_from_pool: Optional[PoolName] = PoolName.KV
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


class HiCacheStorageExtraInfo:
    def __init__(self, prefix_keys=None):
        self.prefix_keys = list(prefix_keys) if prefix_keys is not None else None


class _Op:
    """Mirrors PrefetchOperation's publish/terminate contract."""

    def __init__(self, pages, prefix_keys=None):
        self.id = 7
        self.request_id = "r"
        self.hash_value = [f"h{i}" for i in range(pages)]
        self.host_indices = list(range(pages * PAGE))
        self.prefix_keys = prefix_keys
        self.pool_transfers = [PoolTransfer(PoolName.INDEXER, keys=None)]
        self.completed_tokens = 0
        self.published = []
        self.pool_transfers_done = False
        self._lock = threading.Lock()
        self._terminated = False

    def publish_complete_group(self, num_tokens, page_size):
        with self._lock:
            if self._terminated:
                return False
            self.completed_tokens += num_tokens
            self.published.append(self.completed_tokens)
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated = True

    def is_terminated(self):
        return self._terminated


class _Storage:
    """Records concurrency; ``short`` maps group start page -> hits."""

    def __init__(self, delay=0.05, short_kv=None, short_sidecar=None, raise_at=None,
                 sidecar_delay_at=None, on_kv=None):
        self.delay = delay
        self.sidecar_delay_at = sidecar_delay_at or {}
        self.on_kv = on_kv
        self.finished = []
        self.short_kv = short_kv or {}
        self.short_sidecar = short_sidecar or {}
        self.raise_at = raise_at
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []
        self.prefixes = {}

    def _enter(self, kind, first, delay=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((kind, first))
        time.sleep(self.delay if delay is None else delay)
        with self.lock:
            self.active -= 1
            self.finished.append((kind, first))

    def batch_get_v1(self, hashes, indices, extra_info):
        first = int(hashes[0][1:])
        self.prefixes[first] = extra_info.prefix_keys
        if self.on_kv:
            self.on_kv(first)
        if self.raise_at == first:
            raise RuntimeError("io")
        self._enter("kv", first)
        hits = self.short_kv.get(first, len(hashes))
        return [i < hits for i in range(len(hashes))]

    def batch_get_v2(self, transfers):
        first = int(transfers[0].keys[0][1:])
        self._enter("sidecar", first, self.sidecar_delay_at.get(first))
        hits = self.short_sidecar.get(first, len(transfers[0].keys))
        return {t.name: [i < hits for i in range(len(t.keys))] for t in transfers}


def _controller_class():
    path = SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py"
    wanted = {
        "_successful_prefix",
        "_restore_executor",
        "_page_transfer_complete_groups",
        "_page_transfer_complete_groups_pipelined",
    }
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "HybridCacheController":
            node.body = [
                c for c in node.body
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                and c.name in wanted
            ]
            if {c.name for c in node.body} != wanted:
                raise AssertionError("missing production methods")
            node.bases = []
            node.decorator_list = []
            break
    else:
        raise AssertionError("HybridCacheController not found")
    scope = {
        "replace": replace,
        "STORAGE_BATCH_SIZE": BATCH,
        "HiCacheStorageExtraInfo": HiCacheStorageExtraInfo,
        "trace_event": lambda *a, **k: None,
        "logger": __import__("logging").getLogger("test_restore_pipeline"),
    }
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(code, str(path), "exec"), scope)
    cls = scope["HybridCacheController"]
    for name in ("_successful_prefix",):
        setattr(cls, name, staticmethod(getattr(cls, name)))
    return cls


Controller = _controller_class()


def _make(storage, depth):
    c = Controller.__new__(Controller)
    c.page_size = PAGE
    c.storage_backend = storage
    c.prefetch_pipeline_depth = depth
    c._restore_pool = None
    c.has_draft = False
    c._resolve_sidecar_derived_pool_transfers = lambda op: None
    return c


class PipelinedRestoreTest(unittest.TestCase):
    def run_both(self, pages, depth=3, **storage_kwargs):
        serial_op, piped_op = _Op(pages), _Op(pages)
        _make(_Storage(delay=0, **storage_kwargs), 1)._page_transfer_complete_groups(serial_op)
        storage = _Storage(**storage_kwargs)
        _make(storage, depth)._page_transfer_complete_groups_pipelined(piped_op)
        return serial_op, piped_op, storage

    def test_reads_overlap_and_publish_in_order(self):
        serial, piped, storage = self.run_both(pages=8, depth=3)
        self.assertEqual(piped.published, serial.published)
        self.assertEqual(piped.completed_tokens, 8 * PAGE)
        self.assertTrue(piped.pool_transfers_done)
        # KV+sidecar of up to 3 groups in flight, versus 1 call serially.
        self.assertGreaterEqual(storage.peak, 4)
        self.assertLessEqual(storage.peak, 6)

    def test_short_kv_group_publishes_common_prefix_then_stops(self):
        serial, piped, storage = self.run_both(pages=8, short_kv={2: 1})
        self.assertEqual(piped.published, serial.published)
        self.assertEqual(piped.completed_tokens, 3 * PAGE)
        self.assertTrue(piped.is_terminated())
        # No in-flight read survives the call.
        self.assertEqual(storage.active, 0)

    def test_short_sidecar_clamps_like_serial(self):
        serial, piped, _ = self.run_both(pages=6, short_sidecar={0: 1})
        self.assertEqual(piped.published, serial.published)
        self.assertEqual(piped.completed_tokens, 1 * PAGE)

    def test_exception_terminates_and_drains(self):
        storage = _Storage(raise_at=4)
        op = _Op(8)
        _make(storage, 3)._page_transfer_complete_groups_pipelined(op)
        self.assertTrue(op.is_terminated())
        self.assertEqual(op.completed_tokens, 4 * PAGE)
        self.assertEqual(storage.active, 0)
        self.assertTrue(op.pool_transfers_done)

    def test_prefix_keys_match_serial_snapshots(self):
        serial_store, piped_store = _Storage(delay=0), _Storage()
        _make(serial_store, 1)._page_transfer_complete_groups(_Op(6, prefix_keys=["p"]))
        _make(piped_store, 3)._page_transfer_complete_groups_pipelined(
            _Op(6, prefix_keys=["p"])
        )
        self.assertEqual(piped_store.prefixes, serial_store.prefixes)


class _FlakyPool:
    """Executor whose n-th submit raises, as on shutdown or thread exhaustion."""

    def __init__(self, fail_on):
        from concurrent.futures import ThreadPoolExecutor

        self.inner = ThreadPoolExecutor(max_workers=8)
        self.fail_on = fail_on
        self.count = 0

    def submit(self, fn, *args):
        self.count += 1
        if self.count == self.fail_on:
            raise RuntimeError("submit failed")
        return self.inner.submit(fn, *args)


class PipelinedRestoreDrainTest(unittest.TestCase):
    """Nothing this operation submitted may still run once it returns."""

    def assert_drained(self, storage, op):
        self.assertTrue(op.pool_transfers_done)
        self.assertEqual(storage.active, 0, "a read outlived the operation")
        self.assertEqual(sorted(storage.calls), sorted(storage.finished))

    def test_kv_error_waits_for_slow_sidecar_of_same_group(self):
        # The failing group's sidecar is far slower than everything else, so
        # only an explicit wait on it (not on later groups) can drain it.
        storage = _Storage(delay=0.01, raise_at=2, sidecar_delay_at={2: 0.3})
        op = _Op(8)
        _make(storage, 3)._page_transfer_complete_groups_pipelined(op)
        self.assertTrue(op.is_terminated())
        self.assertEqual(op.completed_tokens, 2 * PAGE)
        self.assertIn(("sidecar", 2), storage.finished)
        self.assert_drained(storage, op)

    def test_submit_failure_half_way_drains_submitted_reads(self):
        storage = _Storage(delay=0.2)
        op = _Op(8)
        c = _make(storage, 3)
        c._restore_pool = _FlakyPool(fail_on=4)  # second group's sidecar
        c._page_transfer_complete_groups_pipelined(op)
        self.assertTrue(op.is_terminated())
        self.assertEqual(op.completed_tokens, 0)
        self.assert_drained(storage, op)

    def test_cancel_mid_flight_publishes_nothing_more_and_drains(self):
        op = _Op(8)
        storage = _Storage(delay=0.05,
                           on_kv=lambda first: first == 2 and op.mark_terminate())
        _make(storage, 3)._page_transfer_complete_groups_pipelined(op)
        self.assertLessEqual(op.completed_tokens, 2 * PAGE)
        self.assertEqual(op.published, sorted(op.published))
        self.assert_drained(storage, op)

    def test_draft_reads_run_inside_the_drained_group_read(self):
        storage = _Storage(delay=0.01, raise_at=2)
        draft_done = []
        op = _Op(8)
        c = _make(storage, 3)
        c.has_draft = True

        def draft(hashes, indices):
            time.sleep(0.1)
            draft_done.append(hashes[0])

        c._draft_page_get = draft
        c._page_transfer_complete_groups_pipelined(op)
        # Every group whose KV read started also finished its draft read
        # (the draft read runs first inside the same task) before return.
        self.assertEqual({int(h[1:]) for h in draft_done}, set(storage.prefixes))
        self.assert_drained(storage, op)


if __name__ == "__main__":
    unittest.main()
