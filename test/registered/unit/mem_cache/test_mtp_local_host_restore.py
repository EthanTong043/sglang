"""Dependency-free regressions for packed MTP host restore layer indices.

Execute the actual HybridCacheController._l2_load_transfers method via AST
without importing Torch, SGLang, or native modules. Only transfer descriptors,
host/device pools, and the assembler's virtual-layer mapping are stand-ins.
This checks controller slot selection, not GPU copies or generated tokens.
"""

from __future__ import annotations

import ast
import os
import queue
import runpy
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[4]
CONTROLLER = Path(
    os.environ.get(
        "HYBRID_CONTROLLER_PATH",
        str(
            REPO
            / "python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
        ),
    )
)

register_cpu_ci = runpy.run_path(
    str(REPO / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Transfer(NamedTuple):
    host_pool: object
    device_pool: object
    host_indices: object
    device_indices: object
    layer_mapper: object = None
    is_draft: bool = False


def _load_method(source: str, trace=None):
    """Compile only the production method; never import its runtime module."""
    tree = ast.parse(source, filename=str(CONTROLLER))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HybridCacheController"
    ]
    if len(classes) != 1:
        raise AssertionError("Expected exactly one HybridCacheController class")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_l2_load_transfers"
    ]
    if len(methods) != 1:
        raise AssertionError("Expected exactly one production restore method")
    future = ast.parse("from __future__ import annotations\n").body
    code = ast.fix_missing_locations(
        ast.Module(body=future + methods, type_ignores=[])
    )
    scope = {"L2Transfer": _Transfer, "trace_event": trace or (lambda *args, **kwargs: None)}
    exec(compile(code, str(CONTROLLER), "exec"), scope)
    return scope["_l2_load_transfers"]


_MISSING = object()
_CP_RANGES = ((0, 20), (20, 40), (40, 59), (59, 78))


def _fixture(
    *,
    target_layer_num=20,
    host_layer_num=21,
    draft_count=1,
    mapping=None,
    cp_rank=None,
):
    """Model the existing assembler's 78+depth virtual MTP layer mapping."""
    if mapping is None:
        mapping = {78 + depth: 78 + depth for depth in range(draft_count)}
    transfers = []
    entries = {}
    mapper_calls = {}
    draft_device_pools = tuple(object() for _ in range(draft_count))
    for name in ("kv", "indexer"):
        pool_attrs = {"layer_num": host_layer_num}
        if target_layer_num is not _MISSING:
            pool_attrs["target_layer_num"] = target_layer_num
        if cp_rank is not None:
            start, end = _CP_RANGES[cp_rank]
            pool_attrs.update(
                owned_layer_start=start,
                owned_layer_end=end,
                owned_layer_count=end - start,
            )
        host_pool = SimpleNamespace(**pool_attrs)
        device_pool = SimpleNamespace(
            layer_num=78, layer_shard_enabled=cp_rank is not None
        )
        calls = []
        mapper_calls[name] = calls

        def mapper(layer_id, *, calls=calls):
            calls.append(layer_id)
            return mapping.get(layer_id)

        transfer = _Transfer(
            host_pool=host_pool,
            device_pool=device_pool,
            host_indices=object(),
            device_indices=object(),
            layer_mapper=mapper,
        )
        transfers.append(transfer)
        entries[name] = SimpleNamespace(
            host_pool=host_pool,
            device_pool=device_pool,
            packed_draft_device_pools=draft_device_pools,
        )
    controller = SimpleNamespace(
        layer_num=78,
        mem_pool_host=SimpleNamespace(entry_map=entries),
        # The accepted runtime uses these controller-wide fields; the source
        # candidate uses each entry's packed_draft_device_pools instead.
        has_mtp_draft=bool(draft_device_pools),
        mtp_draft_device_pools=draft_device_pools,
        _l2_transfers=lambda *_args: list(transfers),
    )
    return SimpleNamespace(
        controller=controller,
        transfers=transfers,
        entries=entries,
        mapper_calls=mapper_calls,
    )


class TestMtpLocalHostRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.load_transfers = staticmethod(_load_method(CONTROLLER.read_text()))

    def restore(self, fixture):
        return self.load_transfers(fixture.controller, object(), object(), None)

    def assert_draft_slots(self, fixture, restored, expected_slots):
        # Target transfers remain intact; additional entries describe draft
        # restores without mutating the caller's host/device pool ownership.
        self.assertEqual(len(restored), 2 + 2 * len(expected_slots))
        for actual, original in zip(restored[:2], fixture.transfers):
            self.assertIs(actual, original)
            self.assertFalse(actual.is_draft)
        for original, entry in zip(fixture.transfers, fixture.entries.values()):
            drafts = [
                transfer
                for transfer in restored
                if transfer.is_draft and transfer.host_pool is original.host_pool
            ]
            self.assertEqual(len(drafts), len(expected_slots))
            for depth, (transfer, slot) in enumerate(zip(drafts, expected_slots)):
                self.assertIs(
                    transfer.device_pool, entry.packed_draft_device_pools[depth]
                )
                self.assertIs(transfer.host_indices, original.host_indices)
                self.assertIs(transfer.device_indices, original.device_indices)
                self.assertEqual(transfer.layer_mapper(depth), slot)
                self.assertIsNone(transfer.layer_mapper(-1))
                self.assertIsNone(transfer.layer_mapper(len(expected_slots)))
                for other_depth in range(len(expected_slots)):
                    if other_depth != depth:
                        self.assertIsNone(transfer.layer_mapper(other_depth))

    def test_all_four_cp_ranks_restore_kv_and_indexer_to_local_slot_20(self):
        for cp_rank in range(4):
            with self.subTest(cp_rank=cp_rank):
                fixture = _fixture(cp_rank=cp_rank)
                restored = self.restore(fixture)
                self.assert_draft_slots(fixture, restored, [20])
                self.assertEqual(fixture.mapper_calls, {"kv": [78], "indexer": [78]})

    def test_cp2_cp3_padding_slot_does_not_become_the_draft_slot(self):
        for cp_rank in (2, 3):
            with self.subTest(cp_rank=cp_rank):
                fixture = _fixture(cp_rank=cp_rank)
                restored = self.restore(fixture)
                for transfer in restored:
                    if transfer.is_draft:
                        self.assertEqual(transfer.host_pool.owned_layer_count, 19)
                        self.assertEqual(transfer.host_pool.target_layer_num, 20)
                        self.assertEqual(transfer.layer_mapper(0), 20)
                        self.assertNotEqual(transfer.layer_mapper(0), 19)

    def test_two_draft_layers_use_distinct_local_slots_20_and_21(self):
        fixture = _fixture(draft_count=2, host_layer_num=22, cp_rank=3)
        self.assert_draft_slots(fixture, self.restore(fixture), [20, 21])
        self.assertEqual(fixture.mapper_calls, {"kv": [78, 79], "indexer": [78, 79]})

    def test_nonsharded_pool_preserves_host_slot_78(self):
        fixture = _fixture(target_layer_num=78, host_layer_num=79)
        self.assert_draft_slots(fixture, self.restore(fixture), [78])

    def test_none_virtual_mtp_mapping_skips_draft(self):
        fixture = _fixture(mapping={78: None}, cp_rank=2)
        self.assertEqual(self.restore(fixture), fixture.transfers)
        self.assertEqual(fixture.mapper_calls, {"kv": [78], "indexer": [78]})

    def test_missing_target_layer_count_retains_original_mapper_result(self):
        fixture = _fixture(
            target_layer_num=_MISSING, host_layer_num=64, mapping={78: 63}
        )
        self.assert_draft_slots(fixture, self.restore(fixture), [63])

    def test_none_target_layer_count_retains_original_mapper_result(self):
        fixture = _fixture(
            target_layer_num=None, host_layer_num=64, mapping={78: 63}
        )
        self.assert_draft_slots(fixture, self.restore(fixture), [63])

    def test_missing_layer_mapper_skips_packed_entry(self):
        fixture = _fixture()
        fixture.transfers[:] = [
            transfer._replace(layer_mapper=None) for transfer in fixture.transfers
        ]
        self.assertEqual(self.restore(fixture), fixture.transfers)
        self.assertEqual(fixture.mapper_calls, {"kv": [], "indexer": []})

    def test_no_draft_pools_leave_target_transfers_unchanged(self):
        fixture = _fixture(draft_count=0)
        self.assertEqual(self.restore(fixture), fixture.transfers)
        self.assertEqual(fixture.mapper_calls, {"kv": [], "indexer": []})

    def test_negative_local_slot_is_rejected(self):
        fixture = _fixture(target_layer_num=-1, cp_rank=0)
        with self.assertRaises(ValueError):
            self.restore(fixture)

    def test_local_slot_at_host_capacity_is_rejected(self):
        fixture = _fixture(target_layer_num=20, host_layer_num=20, cp_rank=2)
        with self.assertRaises(ValueError):
            self.restore(fixture)

    def test_second_draft_beyond_host_capacity_is_rejected(self):
        fixture = _fixture(draft_count=2, host_layer_num=21, cp_rank=3)
        with self.assertRaises(ValueError):
            self.restore(fixture)


class TestMtpDescriptorTrace(unittest.TestCase):
    def setUp(self):
        # Real emitter with an in-memory bounded queue: no writer thread or file.
        helper = runpy.run_path(str(
            REPO / "python/sglang/srt/mem_cache/prefetch_trace_runtime.py"
        ))
        cls = helper["_PrefetchTrace"]
        self.trace = cls.__new__(cls)
        self.trace._queue = queue.Queue(maxsize=2)
        self.trace._lock = threading.Lock()
        self.trace._closed = False
        self.trace._accepted = self.trace._dropped = 0
        self.trace.max_events = 4

    def test_real_controller_trace_preserves_local_slot_evidence(self):
        fixture = _fixture(cp_rank=2)
        load = _load_method(CONTROLLER.read_text(), self.trace.emit)
        load(fixture.controller, object(), object(), None)
        records = [self.trace._queue.get_nowait() for _ in range(2)]
        for record in records:
            self.assertEqual(record["event"], "mtp_load_descriptor")
            self.assertEqual(
                [record[key] for key in (
                    "draft_depth", "virtual_layer", "host_layer", "host_layer_count"
                )], [0, 78, 20, 21]
            )
        self.assertTrue(self.trace._queue.empty())

    def test_emitter_keeps_queue_and_event_limits(self):
        for _ in range(5):
            self.trace.emit("mtp_load_descriptor", host_layer=20)
        self.assertEqual(self.trace._queue.qsize(), 2)
        self.assertEqual(self.trace._accepted, 4)
        self.assertEqual(self.trace._dropped, 3)

    def test_emitter_still_filters_unknown_and_unsafe_values(self):
        self.trace.emit("mtp_load_descriptor", host_layer=1 << 80,
                        host_layer_count=21, draft_depth=float("nan"),
                        virtual_layer=object(), private_payload="secret")
        record = self.trace._queue.get_nowait()
        self.assertEqual(record["host_layer_count"], 21)
        for key in ("host_layer", "draft_depth", "virtual_layer", "private_payload"):
            self.assertNotIn(key, record)


class TestTraceRequestJoin(unittest.TestCase):
    """rid_hash joins one request across CP ranks only with a shared key."""

    def _trace(self, key, path):
        cls = runpy.run_path(str(
            REPO / "python/sglang/srt/mem_cache/prefetch_trace_runtime.py"
        ))["_PrefetchTrace"]
        saved = {k: os.environ.get(k) for k in (
            "SGLANG_PREFETCH_TRACE_RID_KEY", "SGLANG_PREFETCH_TRACE_PATH")}
        try:
            os.environ.pop("SGLANG_PREFETCH_TRACE_RID_KEY", None)
            if key is not None:
                os.environ["SGLANG_PREFETCH_TRACE_RID_KEY"] = key
            os.environ["SGLANG_PREFETCH_TRACE_PATH"] = path
            trace = cls()  # production __init__, including its writer thread
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(trace.close)
        return trace

    def setUp(self):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def test_shared_key_gives_equal_hash_across_ranks(self):
        a = self._trace("run-1", self.path("a"))
        b = self._trace("run-1", self.path("b"))
        c = self._trace("run-2", self.path("c"))
        self.assertEqual(a._rid_hash("req-7"), b._rid_hash("req-7"))
        self.assertNotEqual(a._rid_hash("req-7"), c._rid_hash("req-7"))

    def test_without_key_hashes_differ_per_process(self):
        a = self._trace(None, self.path("a"))
        b = self._trace(None, self.path("b"))
        self.assertNotEqual(a._rid_hash("req-7"), b._rid_hash("req-7"))

    def test_records_report_events_dropped_before_them(self):
        t = self._trace("k", self.path("t"))
        t.max_events = 2
        for op in (1, 2, 3):  # the third is over the limit and dropped
            t.emit("prefetch_aux_start", operation_id=op)
        t.close()
        with open(self.path("t")) as f:
            records = [__import__("json").loads(line) for line in f]
        self.assertEqual([r["operation_id"] for r in records], [1, 2])
        self.assertEqual([r["dropped_before"] for r in records], [0, 0])
        self.assertEqual(t.close(), {"accepted": 2, "dropped": 1})

if __name__ == "__main__":
    unittest.main(verbosity=2)
