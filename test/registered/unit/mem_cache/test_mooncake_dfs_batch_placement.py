"""Dependency-free v5 writer binding and application-accounting regressions.

Execute production AST bodies with independent native/buffer doubles. These
tests cover Python admission and observability, not allocator placement,
physical I/O, real collectives, or GPFS correctness (native/service gates do).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import runpy
import sys
import threading
import time
import unittest
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"
STORE = SRT / "mem_cache/storage/mooncake_store/mooncake_store.py"
METRICS = SRT / "observability/metrics_collector.py"
CONTROLLER = SRT / "managers/cache_controller.py"
HYBRID = SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py"

register_cpu_ci = runpy.run_path(str(REPO / "python/sglang/test/ci/ci_register.py"))["register_cpu_ci"]
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_nodes(path, names, scope):
    selected = []
    for node in ast.parse(path.read_text()).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name in names:
            selected.append(node)
    module = ast.Module(
        body=ast.parse("from __future__ import annotations\n").body + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)


def method(path, class_name, name, scope):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=ast.parse("from __future__ import annotations\n").body + [node], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def scope_for_store():
    scope = {
        "__name__": __name__, "os": os, "time": time, "threading": threading,
        "json": json, "uuid": uuid, "OrderedDict": OrderedDict, "Sequence": Sequence,
        "dataclass": dataclass, "field": field, "Enum": Enum,
        "logger": logging.getLogger("v5-tests"), "DEFAULT_LOCAL_BUFFER_SIZE": 16 << 20,
        "DEFAULT_TENANT_ID": "default", "SETUP_TIMEOUT": 1,
        "HiCacheStorage": type("HiCacheStorage", (), {}),
        "MLATokenToKVPoolHost": type("MLATokenToKVPoolHost", (), {}),
    }
    load_nodes(SRT / "mem_cache/hicache_storage.py", {"PoolName"}, scope)
    load_nodes(METRICS, {"STORAGE_WRITE_LATENCY_BUCKETS", "StorageMetrics"}, scope)
    load_nodes(STORE, {
        "_parse_bool", "_resolve_dfs_writer_slot", "_DfsWriteMetrics",
        "_DfsRepairState", "MooncakeBaseStore", "MooncakeStore", "MooncakeStoreConfig",
    }, scope)
    return scope


SCOPE = scope_for_store()


def rank_config(rank=0, **overrides):
    values = dict(tp_rank=rank, tp_size=4, pp_rank=0, pp_size=1,
                  attn_cp_rank=rank, attn_cp_size=4, layer_sharded=True,
                  owned_layer_start=(0, 20, 40, 59)[rank],
                  owned_layer_end=(20, 40, 59, 78)[rank],
                  is_mla_model=True, enable_storage_metrics=True,
                  should_split_heads=False, model_name="glm-5.3", extra_config=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class Native:
    def __init__(self):
        self.events = []
        self.objects = {}
        self.register_result = 0
        self.setup_result = 0
        self.unsupported_metrics = False

    def setup(self, *args, **kwargs):
        self.events.append(("setup", kwargs))
        if self.unsupported_metrics:
            raise TypeError("unsupported client_http_port")
        return self.setup_result

    def register_dfs_writer(self, slot):
        self.events.append(("register", slot))
        return self.register_result

    def put(self, key, value, *args):
        self.events.append(("put", key))
        self.objects[key] = value
        return 0

    def is_exist(self, key):
        return int(key in self.objects)

    def get(self, key):
        self.events.append(("get", key))
        return self.objects[key]

    def remove(self, key, force):
        self.events.append(("remove", key))
        del self.objects[key]
        return 0


class ReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.dfs_replica_num = 0
        self.nof_replica_num = 0
        self.group_ids = []


class TestWriterBinding(unittest.TestCase):
    def slot(self, config=None, index="0", role="prefill", enabled=True):
        return SCOPE["_resolve_dfs_writer_slot"](
            rank_config() if config is None else config,
            enabled=enabled, prefill_index=index, role=role,
        )

    def create(self, native=None, config=None, *, role="prefill", index="0", enabled="1", port="21300"):
        native = Native() if native is None else native
        config = rank_config() if config is None else config
        cls = SCOPE["MooncakeStore"]
        store_config = SCOPE["MooncakeStoreConfig"](
            local_hostname="localhost", metadata_server="P2PHANDSHAKE",
            global_segment_size=0, protocol="tcp", device_name="",
            master_server_address="master:50051", master_metrics_port=21000,
            check_server=False, standalone_storage=False, client_server_address="",
            enable_ssd_offload=True, ssd_offload_path="/independent-root",
            dfs_only=True, replica_num=0, dfs_replica_num=1,
        )
        context = ModuleType("sglang.srt.runtime_context")
        context.get_disagg = lambda: SimpleNamespace(disaggregation_mode=role)
        environment = {"MOONCAKE_DFS_BATCH_PLACEMENT": enabled}
        if index is not None:
            environment["MOONCAKE_DFS_PREFILL_INDEX"] = index
        if port is not None:
            environment["MOONCAKE_DFS_CLIENT_METRICS_PORT_BASE"] = port
        with patch.dict(os.environ, environment, clear=True), patch.dict(
            sys.modules, {"sglang.srt.runtime_context": context}
        ), patch.object(cls, "_import_mooncake_store", lambda self: lambda: native), patch.object(
            cls, "_import_mooncake_group_semantics", lambda self: (ReplicateConfig, True)
        ), patch.object(cls, "_load_config", lambda self, _: store_config):
            return cls(config, SimpleNamespace()), native

    def test_all_forty_slots_are_explicit_unique_and_stable(self):
        slots = [self.slot(rank_config(rank), str(index)) for index in range(10) for rank in range(4)]
        self.assertEqual(slots, list(range(40)))
        self.assertEqual(self.slot(rank_config(3), "9"), 39)

    def test_reader_and_default_do_not_claim_writer_slots(self):
        self.assertIsNone(self.slot(index=None, role="decode"))
        self.assertIsNone(self.slot(index=None, role="null"))
        self.assertIsNone(self.slot(index="9", role="decode", enabled=False))
        with self.assertRaisesRegex(ValueError, "only for a Prefill"):
            self.slot(index="0", role="decode")
        with self.assertRaisesRegex(ValueError, "requires MOONCAKE"):
            self.slot(index=None)

    def test_unsupported_or_ambiguous_topologies_fail_closed(self):
        for values in (
            {"attn_cp_size": 1}, {"attn_cp_rank": 4}, {"attn_cp_rank": -1},
            {"tp_size": 8}, {"tp_rank": 1}, {"pp_size": 2}, {"pp_rank": 1},
            {"layer_sharded": False},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.slot(rank_config(**values))
        for index in ("-1", "10", "x", "1.5"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.slot(index=index)

    def test_setup_then_register_then_warmup_uses_rank_metrics_port(self):
        store, native = self.create(config=rank_config(3), index="9")
        self.assertEqual([event[0] for event in native.events], ["setup", "register", "put", "get", "remove"])
        self.assertEqual(native.events[1], ("register", 39))
        self.assertEqual(native.events[0][1]["client_http_port"], 21303)
        self.assertTrue(native.events[0][1]["enable_client_http_server"])
        self.assertEqual(store.attn_cp_rank, 3)
        self.assertEqual(native.objects, {}, "warmup must not leave a persistent object")

    def test_failed_setup_or_registration_never_writes_warmup(self):
        for setup, register in ((-1, 0), (0, -1), (0, None), (0, False)):
            native = Native()
            native.setup_result, native.register_result = setup, register
            with self.subTest(setup=setup, register=register), self.assertRaises(RuntimeError):
                self.create(native)
            self.assertNotIn("put", [event[0] for event in native.events])

    def test_missing_binding_and_requested_metrics_do_not_silently_degrade(self):
        native = Native()
        native.register_dfs_writer = None
        with self.assertRaisesRegex(RuntimeError, "register_dfs_writer"):
            self.create(native)
        native = Native()
        native.unsupported_metrics = True
        with self.assertRaisesRegex(RuntimeError, "compatible native HTTP"):
            self.create(native)
        self.assertEqual(len(native.events), 1, "must not retry setup without requested metrics")

    def test_reader_has_no_registration_or_warmup_and_rejects_write(self):
        store, native = self.create(role="decode", index=None)
        self.assertEqual([event[0] for event in native.events], ["setup"])
        self.assertNotIn("client_http_port", native.events[0][1])
        with self.assertRaisesRegex(RuntimeError, "reader cannot submit"):
            store._write_batch_with_repair(["key"], [123], [64])

    def test_default_off_preserves_old_binding_and_warmup(self):
        native = Native()
        native.register_dfs_writer = None
        store, native = self.create(native, enabled="0", role="decode", index=None)
        self.assertEqual([event[0] for event in native.events], ["setup", "put", "get", "remove"])
        self.assertIsNone(store._dfs_writer_slot)

    def test_instrumented_baseline_keeps_metrics_without_writer_registration(self):
        native = Native()
        native.register_dfs_writer = None
        store, native = self.create(native, config=rank_config(3), index="9", enabled="0")
        self.assertEqual([event[0] for event in native.events], ["setup", "put", "get", "remove"])
        self.assertEqual(native.events[0][1]["client_http_port"], 21303)
        self.assertTrue(native.events[0][1]["enable_client_http_server"])
        self.assertIsNone(store._dfs_writer_slot)
        self.assertFalse(store._dfs_batch_reader_only)

    def test_instrumented_baseline_rejects_wrong_role_and_topology(self):
        with self.assertRaisesRegex(ValueError, "only for a Prefill"):
            self.create(role="decode", enabled="0", index="0")
        with self.assertRaisesRegex(ValueError, "attn_cp_size=4"):
            self.create(config=rank_config(attn_cp_size=1), enabled="0")


class TestWriteAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heal = runpy.run_path(str(Path(__file__).with_name("test_mooncake_dfs_self_heal.py")))
        cls.heal_case = cls.heal["TestMooncakeDfsSelfHeal"]
        cls.heal_case.setUpClass()

    def fixture(self):
        helper = self.heal_case("runTest")
        self.addCleanup(helper.doCleanups)
        store, native, memory = helper.fixture()
        store._dfs_write_metrics = SCOPE["_DfsWriteMetrics"]()
        store._dfs_metric_context = threading.local()
        return helper, store, native, memory

    @staticmethod
    def counter(snapshot, pool, event, unit):
        return snapshot["counters"].get((pool, event, unit), 0)

    def test_native_batch_boundaries_keep_primary_128_and_indexer_actual_size(self):
        helper, store, native, _ = self.fixture()
        names = [f"page-{index}" for index in range(130)]
        scope = {"STORAGE_BATCH_SIZE": 128, "HiCacheStorageExtraInfo": lambda **kw: SimpleNamespace(**kw),
                 "time": time, "logger": logging.getLogger("v5-tests")}
        page_backup = method(CONTROLLER, "HiCacheController", "_page_backup", scope)
        operation = SimpleNamespace(prefix_keys=None, hash_value=names, host_indices=list(range(130 * 64)),
                                    completed_tokens=0, failed_tokens=0, failed_reason=None)
        draft_batches = []
        controller = SimpleNamespace(page_size=64, has_draft=True,
            _draft_page_set=lambda keys, indices: draft_batches.append(list(keys)),
            page_set_func=lambda keys, indices, extra: all(store.batch_set_v1(keys, indices, extra)))
        page_backup(controller, operation)
        self.assertEqual(operation.completed_tokens, 130 * 64)
        self.assertEqual([len(batch) for batch in draft_batches], [128, 2])
        self.assertEqual(helper.call(store, "v2", "set", names), [True] * 130)
        self.assertEqual([len(call["keys"]) for call in native.writes], [128, 2, 130])
        snapshot = store._dfs_write_metrics.snapshot()
        for pool, operations in (("kv", 2), ("indexer", 1)):
            self.assertEqual(self.counter(snapshot, pool, "batch_submitted", "operations"), operations)
            self.assertEqual(self.counter(snapshot, pool, "put_submitted", "objects"), 130)
            self.assertEqual(self.counter(snapshot, pool, "put_submitted", "payload_bytes"), 130 * 8)
            self.assertEqual(self.counter(snapshot, pool, "batch_completed", "tokens"), 130 * 64)

    def test_duplicate_ready_is_not_native_submitted_or_new_bytes(self):
        helper, store, native, _ = self.fixture()
        self.assertEqual(helper.call(store, "v1", "set"), [True])
        store._dfs_write_metrics.snapshot()
        self.assertEqual(helper.call(store, "v1", "set"), [True])
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(len(native.writes), 1)
        self.assertEqual(self.counter(snapshot, "kv", "duplicate_ready", "objects"), 1)
        self.assertEqual(self.counter(snapshot, "kv", "put_submitted", "payload_bytes"), 0)
        self.assertEqual(self.counter(snapshot, "kv", "batch_completed", "payload_bytes"), 8)

    def test_native_partial_failure_keeps_per_object_success_and_phase_time(self):
        helper, store, native, _ = self.fixture()
        native.write_override = lambda keys: [0, -1]
        self.assertEqual(helper.call(store, "v1", "set", ("a", "b")), [True, False])
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(self.counter(snapshot, "kv", "put_submitted", "payload_bytes"), 16)
        self.assertEqual(self.counter(snapshot, "kv", "put_returned_success", "payload_bytes"), 8)
        self.assertEqual(self.counter(snapshot, "kv", "put_failed", "objects"), 1)
        self.assertEqual(self.counter(snapshot, "kv", "batch_completed", "objects"), 1)
        self.assertEqual(snapshot["timings"][("kv", "native_put")][0], 1)
        self.assertEqual(list(native.objects), [helper.key("v1", "a")])

    def test_native_exception_is_counted_and_leaves_no_published_object(self):
        helper, store, native, _ = self.fixture()
        def fail(_):
            raise OSError("injected native failure")
        native.before_write = fail
        self.assertEqual(helper.call(store, "v1", "set"), [False])
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(self.counter(snapshot, "kv", "put_failed", "objects"), 1)
        self.assertEqual(snapshot["timings"][("kv", "native_put")][0], 1)
        self.assertEqual(native.objects, {})

    def test_repair_new_and_duplicate_payloads_remain_separate(self):
        helper, store, native, _ = self.fixture()
        broken = helper.key("v1", "broken")
        existing = helper.key("v1", "existing")
        native.seed(broken, b"badbytes")
        native.seed(existing, b"unchange")
        store._record_failed_page_reads([broken], [-801], [8])
        self.assertEqual(
            helper.call(store, "v1", "set", ("broken", "new", "existing")),
            [True, True, True],
        )
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(self.counter(snapshot, "kv", "put_submitted", "payload_bytes"), 8)
        self.assertEqual(self.counter(snapshot, "kv", "repair_submitted", "payload_bytes"), 8)
        self.assertEqual(self.counter(snapshot, "kv", "duplicate_ready", "payload_bytes"), 8)
        self.assertEqual([len(call["keys"]) for call in native.writes], [1, 1])
        self.assertEqual(native.objects[existing], b"unchange")
        self.assertNotEqual(native.objects[broken], b"badbytes")

    def test_pending_and_ack_pair_exactly_including_failed_sidecar(self):
        _, store, _, _ = self.fixture()
        operation = SimpleNamespace(hash_value=["a", "b"], completed_tokens=64,
            pool_storage_result=SimpleNamespace(extra_pool_hit_pages={"indexer": 0}))
        transfer = SimpleNamespace(name="indexer", keys=None, indices_from_pool="kv", host_indices=None)
        store.record_backup_queued(operation, 64, False, [transfer])
        queued = store._dfs_write_metrics.snapshot()
        self.assertEqual(queued["pending"][("kv", "queued", "tokens")], 128)
        self.assertEqual(queued["pending"][("indexer", "queued", "tokens")], 128)
        store.record_backup_started(operation)
        store.record_backup_finished(operation)
        store.record_backup_finished(operation)  # a defensive duplicate must not count twice
        finished = store._dfs_write_metrics.snapshot()
        self.assertEqual(self.counter(finished, "kv", "local_completed", "tokens"), 64)
        self.assertEqual(self.counter(finished, "kv", "local_failed", "tokens"), 64)
        self.assertEqual(self.counter(finished, "indexer", "local_failed", "tokens"), 128)
        self.assertEqual(finished["pending"][("kv", "awaiting_ack", "tokens")], 128)
        self.assertEqual(finished["pending"][("kv", "active", "tokens")], 0)
        store.record_backup_ack_drained(operation)
        store.record_backup_ack_drained(operation)
        acked = store._dfs_write_metrics.snapshot()
        self.assertTrue(all(value == 0 for value in acked["pending"].values()))
        self.assertEqual(self.counter(acked, "kv", "ack_drained", "operations"), 1)
        self.assertEqual(self.counter(acked, "indexer", "ack_drained", "operations"), 1)
        self.assertEqual(acked["timings"][("kv", "ack_wait")][0], 1)

    def test_readiness_timing_preserves_processing_timeout_without_writes(self):
        helper, store, native, _ = self.fixture()
        key = helper.key("v1")
        native.seed(key, b"abcdefgh")
        native.exists_override = lambda keys: [0] * len(keys)
        native.readiness_override = lambda keys: {}
        store._DFS_DUPLICATE_READY_TIMEOUT_S = 0.001
        self.assertEqual(helper.call(store, "v1", "set"), [False])
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(native.writes, [])
        self.assertEqual(native.objects[key], b"abcdefgh")
        self.assertEqual(self.counter(snapshot, "kv", "readiness_timeout", "objects"), 1)
        self.assertGreater(snapshot["timings"][("kv", "readiness_http")][0], 0)
        self.assertGreater(snapshot["timings"][("kv", "readiness_native_exist")][0], 0)
        self.assertGreater(snapshot["timings"][("kv", "readiness_sleep")][1], 0)

    def test_runtime_sidecar_bool_vector_is_only_normalized_for_metrics(self):
        _, store, _, _ = self.fixture()
        results = [True, False, True]
        operation = SimpleNamespace(hash_value=["a", "b", "c"], completed_tokens=192,
            pool_storage_result=SimpleNamespace(extra_pool_hit_pages={"indexer": results}))
        transfer = SimpleNamespace(name="indexer", keys=None, indices_from_pool="kv", host_indices=None)
        store.record_backup_queued(operation, 64, False, [transfer])
        store.record_backup_started(operation)
        store.record_backup_finished(operation)
        snapshot = store._dfs_write_metrics.snapshot()
        self.assertEqual(self.counter(snapshot, "indexer", "local_completed", "tokens"), 64)
        self.assertEqual(self.counter(snapshot, "indexer", "local_failed", "tokens"), 128)
        self.assertIs(operation.pool_storage_result.extra_pool_hit_pages["indexer"], results)
        self.assertEqual(results, [True, False, True])

    def test_snapshot_is_lossless_under_concurrent_updates_and_bounded(self):
        metrics = SCOPE["_DfsWriteMetrics"]()
        start = threading.Barrier(5)
        def worker():
            start.wait()
            for _ in range(500):
                metrics.record("kv", "put_submitted", operations=1, payload_bytes=17,
                               phase="native_put", duration=0.01)
        workers = [threading.Thread(target=worker) for _ in range(4)]
        for worker in workers:
            worker.start()
        start.wait()
        snapshots = []
        while any(worker.is_alive() for worker in workers):
            snapshots.append(metrics.snapshot())
            time.sleep(0.0001)
        for worker in workers:
            worker.join()
        snapshots.append(metrics.snapshot())
        self.assertEqual(sum(self.counter(s, "kv", "put_submitted", "operations") for s in snapshots), 2000)
        self.assertEqual(sum(self.counter(s, "kv", "put_submitted", "payload_bytes") for s in snapshots), 34000)
        self.assertEqual(sum(s["timings"].get(("kv", "native_put"), [0])[0] for s in snapshots), 2000)
        for snapshot in snapshots:
            self.assertLessEqual(len(snapshot["counters"]), 4)
            self.assertLessEqual(len(snapshot["timings"]), 1)

    def test_both_runtime_daemons_preserve_exception_no_ack_and_measure_stranded_work(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                _, store, _, _ = self.fixture()
                scope = {"time": time, "logger": logging.getLogger("v5-tests"),
                         "Empty": Empty, "PoolTransferResult": SimpleNamespace(
                             empty=lambda: SimpleNamespace(extra_pool_hit_pages={}))}
                load_nodes(CONTROLLER, {"StorageOperation"}, scope)
                if hybrid:
                    scope["BaseStorageOperation"] = scope["StorageOperation"]
                    load_nodes(HYBRID, {"StorageOperation"}, scope)
                controller = SimpleNamespace(
                    storage_backend=store, page_size=64, backup_skip=False,
                    backup_queue=Queue(), ack_backup_queue=Queue(), stopped=False,
                    should_backup=lambda transfer: True,
                )
                controller.storage_stop_event = SimpleNamespace(is_set=lambda: controller.stopped)
                controller._record_storage_backup_state = method(
                    CONTROLLER, "HiCacheController", "_record_storage_backup_state", scope
                ).__get__(controller)
                source = HYBRID if hybrid else CONTROLLER
                owner = "HybridCacheController" if hybrid else "HiCacheController"
                enqueue = method(source, owner, "write_storage", scope)
                daemon = method(source, owner, "backup_thread_func", scope)
                enqueue(controller, list(range(128)), [], hash_value=["a", "b"])
                operation = controller.backup_queue.queue[0]
                def work(operation):
                    operation.completed_tokens = 64
                    raise OSError("injected after first page")
                controller._page_backup = work
                original_get = controller.backup_queue.get
                def get(**kwargs):
                    if controller.backup_queue.empty():
                        controller.stopped = True
                        raise Empty
                    return original_get(**kwargs)
                controller.backup_queue.get = get
                with self.assertRaisesRegex(OSError, "injected after first page"):
                    daemon(controller)
                self.assertEqual(controller.ack_backup_queue.qsize(), 0)
                self.assertEqual(operation.completed_tokens, 64)
                self.assertFalse(hasattr(operation, "failed_tokens"))
                self.assertFalse(hasattr(operation, "failed_reason"))
                snapshot = store._dfs_write_metrics.snapshot()
                self.assertEqual(self.counter(snapshot, "kv", "queued", "tokens"), 128)
                self.assertEqual(self.counter(snapshot, "kv", "local_completed", "tokens"), 64)
                self.assertEqual(self.counter(snapshot, "kv", "local_failed", "tokens"), 64)
                self.assertEqual(snapshot["pending"][("kv", "active", "tokens")], 0)
                self.assertEqual(snapshot["pending"][("kv", "failed_no_ack", "tokens")], 128)
                self.assertEqual(self.counter(snapshot, "kv", "ack_drained", "operations"), 0)
                self.assertTrue(operation._dfs_write_tickets)

    def test_both_runtime_daemons_keep_success_ack_fifo_and_pending_until_drain(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                _, store, _, _ = self.fixture()
                scope = {"time": time, "logger": logging.getLogger("v5-tests"),
                         "Empty": Empty, "PoolTransferResult": SimpleNamespace(
                             empty=lambda: SimpleNamespace(extra_pool_hit_pages={}))}
                load_nodes(CONTROLLER, {"StorageOperation"}, scope)
                if hybrid:
                    scope["BaseStorageOperation"] = scope["StorageOperation"]
                    load_nodes(HYBRID, {"StorageOperation"}, scope)
                controller = SimpleNamespace(
                    storage_backend=store, page_size=64, backup_skip=False,
                    backup_queue=Queue(), ack_backup_queue=Queue(), stopped=False,
                    should_backup=lambda transfer: True,
                )
                controller.storage_stop_event = SimpleNamespace(is_set=lambda: controller.stopped)
                controller._record_storage_backup_state = method(
                    CONTROLLER, "HiCacheController", "_record_storage_backup_state", scope
                ).__get__(controller)
                source = HYBRID if hybrid else CONTROLLER
                owner = "HybridCacheController" if hybrid else "HiCacheController"
                enqueue = method(source, owner, "write_storage", scope)
                daemon = method(source, owner, "backup_thread_func", scope)
                for key in ("a", "b"):
                    enqueue(controller, list(range(64)), [], hash_value=[key])
                original_operations = list(controller.backup_queue.queue)
                def work(operation):
                    operation.completed_tokens = 64
                controller._page_backup = work
                original_get = controller.backup_queue.get
                def get(**kwargs):
                    if controller.backup_queue.empty():
                        controller.stopped = True
                        raise Empty
                    return original_get(**kwargs)
                controller.backup_queue.get = get
                daemon(controller)
                acked = list(controller.ack_backup_queue.queue)
                self.assertEqual(acked, original_operations)
                pending = store._dfs_write_metrics.snapshot()
                self.assertEqual(pending["pending"][("kv", "awaiting_ack", "tokens")], 128)
                self.assertEqual(self.counter(pending, "kv", "local_completed", "operations"), 2)
                for operation in acked:
                    store.record_backup_ack_drained(operation)
                drained = store._dfs_write_metrics.snapshot()
                self.assertTrue(all(value == 0 for value in drained["pending"].values()))


class _Value:
    def __init__(self):
        self.value = 0

    def inc(self, value=1):
        self.value += value


class _Metric:
    def __init__(self, *, buckets=(), **kwargs):
        self.children = {}
        self.bounds = tuple(buckets)
        self.value = 0
        self._sum = _Value()
        self._buckets = [_Value() for _ in (*self.bounds, float("inf"))]

    def labels(self, **labels):
        key = tuple(sorted(labels.items()))
        return self.children.setdefault(key, _Metric(buckets=self.bounds))

    def set(self, value):
        self.value = value

    def inc(self, value=1):
        self.value += value

    def observe(self, value):
        self._sum.inc(value)
        for bound, bucket in zip((*self.bounds, float("inf")), self._buckets):
            if value <= bound:
                bucket.inc()
                break


class TestMetricsExportAndConsensus(unittest.TestCase):
    def test_runtime_ack_observation_happens_after_existing_host_release(self):
        scope = {"Empty": Empty}
        drain = method(SRT / "mem_cache/unified_radix_cache.py", "UnifiedRadixCache",
                       "_drain_storage_control_queues_impl", scope)
        events = []
        operation = SimpleNamespace(id=7, completed_tokens=64)
        controller = SimpleNamespace(prefetch_hit_queue=Queue(), ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(), storage_backend=SimpleNamespace(
                record_backup_ack_drained=lambda op: events.append(("observe", op.id))))
        controller.ack_backup_queue.put(operation)
        cache = SimpleNamespace(cache_controller=controller, ongoing_backup={7: ("node", "lease")},
            dec_host_lock_ref=lambda node, lease: events.append(("release", node, lease)))
        drain(cache, n_storage_hit=0, n_backup=1, n_release=0,
              extra_release_counts=None, log_metrics=False)
        self.assertEqual(events, [("release", "node", "lease"), ("observe", 7)])
        self.assertEqual(cache.ongoing_backup, {})
        self.assertEqual(controller.ack_backup_queue.qsize(), 0)

    def test_export_uses_delta_counters_absolute_pending_and_histogram_buckets(self):
        scope = dict(SCOPE)
        scope["_StatLoggerDIMixin"] = type("Mixin", (), {
            "_counter_cls": None, "_gauge_cls": None, "_histogram_cls": None,
        })
        load_nodes(METRICS, {"StorageMetricsCollector"}, scope)
        prometheus = ModuleType("prometheus_client")
        prometheus.Counter = prometheus.Gauge = prometheus.Histogram = _Metric
        with patch.dict(sys.modules, {"prometheus_client": prometheus}):
            collector = scope["StorageMetricsCollector"]({"tp_rank": "3"})
        tracker = scope["_DfsWriteMetrics"]()
        for delay in (0.001, 0.1, 4000):
            tracker.record("kv", "queued", operations=1, tokens=64,
                           phase="queue", duration=delay,
                           pending=(("queued", 1, 64),))
        metrics = scope["StorageMetrics"](dfs_write=tracker.snapshot(), dfs_writer_slot=39)
        collector.log_storage_metrics(metrics)
        collector.log_storage_metrics(scope["StorageMetrics"](dfs_write=tracker.snapshot()))
        self.assertEqual(collector.dfs_app_counters["tokens"].labels(tp_rank="3", pool="kv", event="queued").value, 192)
        self.assertEqual(collector.dfs_app_pending["tokens"].labels(tp_rank="3", pool="kv", state="queued").value, 192)
        histogram = collector.dfs_app_phase_seconds.labels(tp_rank="3", pool="kv", phase="queue")
        self.assertEqual(sum(bucket.value for bucket in histogram._buckets), 3)
        self.assertAlmostEqual(histogram._sum.value, 4000.101)
        self.assertEqual(histogram._buckets[-1].value, 1)
        self.assertEqual(collector.dfs_writer_slot.labels(tp_rank="3").value, 39)

    def test_ack_count_collective_stays_single_and_observation_is_after_it(self):
        events = []
        class Tensor:
            def __init__(self, values):
                self.values = values
            def tolist(self):
                return self.values
        torch = SimpleNamespace(int64="int64", tensor=lambda values, **kwargs: Tensor(values),
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(MIN="MIN")))
        scope = {"time": time, "torch": torch}
        sync = method(SRT / "mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "_sync_hicache_ready_counts", scope)
        controller = SimpleNamespace(ack_write_queue=[], ack_load_queue=[],
            prefetch_hit_queue=Queue(), ack_prefetch_queue=Queue(), ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(), storage_backend=SimpleNamespace(
                record_backup_ack_collective=lambda elapsed: events.append(("observe", elapsed))))
        controller.ack_backup_queue.put(object())
        cache = SimpleNamespace(cache_controller=controller, enable_storage=True,
            _count_ready_acks=len, tree_core=SimpleNamespace(write_back_duplicate_reclaim_digest=27),
            _all_reduce=lambda tensor, op: events.append(("reduce", op)))
        result = sync(cache)
        self.assertEqual([event[0] for event in events], ["reduce", "observe"])
        # This pinned runtime has no separate prefetch-ACK queue in the vector.
        self.assertEqual(result, (0, 0, (0, 1, 0), ()))
        self.assertGreaterEqual(events[-1][1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
