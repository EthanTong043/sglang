"""CPU regressions for storage-query exceptions before the prefetch rank MIN.

Execute actual controller, Mooncake query, and scheduler lifecycle methods from
this checkout without importing Torch or native/GPU modules. Only the native
existence boundary, scalar collective, and host allocation resources are models.
This does not validate data I/O, GPU transfers, or real distributed collectives.
"""

from __future__ import annotations

import ast
import logging
import runpy
import threading
import time
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"
LOGGER = "test_prefetch_query_exception"
TIMEOUT = 3.0

# CI registration itself is dependency-free. Importing sglang would load the
# optional runtime dependencies before these CPU-only tests could run.
register_cpu_ci = runpy.run_path(
    str(REPO / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Scalar:
    def __init__(self, value, **_kwargs):
        self.value = value

    def item(self):
        return self.value


class _Slots(list):
    def numel(self):
        return len(self)

    def split(self, width):
        return [
            _Slots(self[start : start + width])
            for start in range(0, len(self), width)
        ]


class _HostPool:
    """Resource oracle: a slot cannot be allocated twice or freed unowned."""

    page_size = 4

    def __init__(self):
        self.owned = set()
        self.allocations = []
        self.freed = []
        self.next_slot = 0
        self.entry_map = {}

    def alloc(self, count):
        slots = _Slots(range(self.next_slot, self.next_slot + count))
        self.next_slot += count
        self.owned.update(slots)
        self.allocations.append(slots)
        return slots

    def free(self, slots, **_kwargs):
        for slot in slots:
            if slot not in self.owned:
                raise AssertionError("host slot released twice or not owned")
            self.owned.remove(slot)
            self.freed.append(slot)


class _NativeExistence:
    def __init__(self, failure=None, fail_call=2):
        self.failure = failure
        self.fail_call = fail_call
        self.calls = 0

    def batch_is_exist(self, keys):
        self.calls += 1
        if self.failure is not None and self.calls == self.fail_call:
            raise self.failure
        return [1] * len(keys)


class _TwoRankMin:
    """Bounded rendezvous detects a rank missing or reordering a query MIN."""

    def __init__(self):
        self.groups = object()
        self.condition = threading.Condition()
        self.calls = [0, 0]
        self.rounds = {}
        self.completed = []

    def reduce(self, rank, tensor, operation, groups):
        if operation != "MIN" or groups is not self.groups:
            raise AssertionError("unexpected prefetch collective or groups")
        with self.condition:
            round_id = self.calls[rank]
            self.calls[rank] += 1
            state = self.rounds.setdefault(round_id, {"values": {}})
            state["values"][rank] = tensor.value
            if len(state["values"]) == 2:
                state["minimum"] = min(state["values"].values())
                self.completed.append((round_id, state["minimum"]))
                self.condition.notify_all()
            elif not self.condition.wait_for(lambda: "minimum" in state, TIMEOUT):
                raise TimeoutError("peer did not join metadata reduction")
            tensor.value = state["minimum"]


def _load_classes(path, wanted, scope, bases=None):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ClassDef) or node.name not in wanted:
            continue
        members = wanted[node.name]
        if members is not None:
            node.body = [
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name in members
            ]
            if {child.name for child in node.body} != set(members):
                raise AssertionError(f"Missing production methods in {path}")
        if bases is not None and node.name in bases:
            node.bases = [
                ast.Name(id=name, ctx=ast.Load()) for name in bases[node.name]
            ]
        nodes.append(node)
    if {node.name for node in nodes} != set(wanted):
        raise AssertionError(f"Missing production classes in {path}")
    future = ast.parse("from __future__ import annotations\n").body
    code = ast.fix_missing_locations(
        ast.Module(body=future + nodes, type_ignores=[])
    )
    exec(compile(code, str(path), "exec"), scope)


def _actual_types():
    scope = {
        "__name__": __name__,
        "dataclass": dataclass,
        "Enum": Enum,
        "NamedTuple": NamedTuple,
        "Queue": Queue,
        "Empty": Empty,
        "threading": threading,
        "time": time,
        "logger": logging.getLogger(LOGGER),
        "torch": SimpleNamespace(
            tensor=_Scalar,
            int="int",
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(MIN="MIN")),
            cat=lambda parts, **_kwargs: _Slots(
                slot for part in parts for slot in part
            ),
        ),
    }
    _load_classes(
        SRT / "mem_cache/hicache_storage.py",
        {
            name: None
            for name in (
                "PoolName",
                "PoolHitPolicy",
                "PoolTransfer",
                "PoolTransferResult",
                "HiCacheStorageExtraInfo",
            )
        },
        scope,
    )
    _load_classes(
        SRT / "managers/cache_controller.py",
        {
            "StorageOperation": None,
            "HiCacheController": {"prefetch_thread_func"},
        },
        scope,
        {"HiCacheController": []},
    )
    scope["BaseStorageOperation"] = scope["StorageOperation"]
    scope["BaseHiCacheController"] = scope["HiCacheController"]
    _load_classes(
        SRT / "mem_cache/hybrid_cache/hybrid_cache_controller.py",
        {
            "StorageOperation": None,
            "PrefetchOperation": None,
            "HybridCacheController": {
                "_storage_hit_query",
                "append_host_mem_release",
                "_append_host_mem_release_pages",
            },
        },
        scope,
        {"HybridCacheController": ["BaseHiCacheController"]},
    )
    _load_classes(
        SRT / "mem_cache/storage/mooncake_store/mooncake_store.py",
        {
            "MooncakeStore": {
                "batch_exists",
                "batch_exists_v2",
                "_tag_keys",
                "_mla_ownership_suffix",
                "_make_mla_physical_key",
                "_get_hybrid_page_component_keys",
                "_batch_exist",
            }
        },
        scope,
        {"MooncakeStore": []},
    )
    _load_classes(
        SRT / "mem_cache/unified_radix_cache.py",
        {
            "_OngoingPrefetch": None,
            "UnifiedRadixCache": {
                "revoke_pending_prefetch",
                "_prefetch_occupied_span",
                "_invalidate_absent_from_hit_query",
                "_account_prefetch_outcome",
                "_drain_storage_control_queues_impl",
            },
        },
        scope,
        {"UnifiedRadixCache": []},
    )
    return scope


class _RankFixture:
    def __init__(self, types, rank, reduction, groups, failure=None, fail_call=2):
        self.types = types
        self.cc = types["HybridCacheController"]()
        self.tree = types["UnifiedRadixCache"]()
        self.native = _NativeExistence(failure, fail_call)
        backend = types["MooncakeStore"]()
        backend.store = self.native
        backend.config_prefix = "owned-query-fixture"
        backend.is_mla_backend, backend.mla_suffix = True, "0"
        backend.mem_pool_host = SimpleNamespace(kv_buffer=object())
        backend.registered_pools = {types["PoolName"].INDEXER: object()}

        cc = self.cc
        cc.storage_backend = backend
        cc.storage_stop_event = threading.Event()
        cc.prefetch_queue, cc.prefetch_hit_queue = Queue(), Queue()
        cc.prefetch_buffer, cc.ack_prefetch_queue = Queue(), Queue()
        cc.ack_backup_queue, cc.host_mem_release_queue = Queue(), Queue()
        cc.extra_host_mem_release_queues = {}
        cc.prefetch_hits_sync_groups = groups
        cc.page_size, cc.prefetch_tokens_occupied = 4, 0
        cc.mem_pool_host = _HostPool()
        cc.get_hash_str = lambda tokens, _last, page_size: [
            f"page-{tokens[index]}" for index in range(0, len(tokens), page_size)
        ]
        cc._all_reduce = lambda tensor, op, passed_groups: reduction(
            rank, tensor, op, passed_groups
        )

        tree = self.tree
        tree.cache_controller = cc
        tree.prefetch_threshold, tree.page_size = 4, 4
        tree.host_memory_mode, tree.buffer_pipeline = "cache", None
        tree.ongoing_prefetch, tree.ongoing_backup = {}, {}
        tree._storage_prefetch_missed_rids = set()
        tree.enable_storage_metrics = False
        self.locks, self.releases, self.errors = set(), [], []

        def release(node, _params):
            if node not in self.locks:
                raise AssertionError("request anchor released twice")
            self.locks.remove(node)
            self.releases.append(node)

        tree.dec_host_lock_ref = release
        tree.evict_host = lambda _count: None
        self.worker = None

    def request(self, request_id):
        transfer = self.types["PoolTransfer"](
            self.types["PoolName"].INDEXER,
            indices_from_pool=self.types["PoolName"].KV,
        )
        operation = self.types["PrefetchOperation"](
            request_id, list(range(8)), pool_transfers=[transfer]
        )
        self.locks.add(request_id)
        self.tree.ongoing_prefetch[request_id] = self.types["_OngoingPrefetch"](
            request_id,
            list(range(8)),
            None,
            operation,
            object(),
            {"sidecar": [transfer]},
        )
        self.cc.prefetch_tokens_occupied += 8
        self.cc.prefetch_queue.put(operation)
        return operation

    def start(self):
        def run():
            try:
                self.cc.prefetch_thread_func()
            except BaseException as error:
                self.errors.append(error)

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def query_result(self):
        try:
            return self.cc.prefetch_hit_queue.get(timeout=TIMEOUT + 0.5)
        except Empty:
            errors = [type(error).__name__ for error in self.errors]
            raise AssertionError(
                f"query completion missing; worker errors={errors}"
            ) from None

    def drain(self, results=()):
        for result in results:
            self.cc.prefetch_hit_queue.put(result)
        self.tree._drain_storage_control_queues_impl(
            n_storage_hit=None,
            n_ack_prefetch=None,
            n_backup=None,
            n_release=None,
            extra_release_counts={},
            log_metrics=False,
        )

    def close(self):
        self.cc.storage_stop_event.set()
        self.cc.prefetch_queue.put(None)
        if self.worker is not None:
            self.worker.join(TIMEOUT + 0.5)
            if self.worker.is_alive():
                raise AssertionError("query worker did not stop")


class TestPrefetchQueryException(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types = _actual_types()

    def rank(self, rank, reduction, groups=None, **kwargs):
        fixture = _RankFixture(self.types, rank, reduction, groups, **kwargs)
        self.addCleanup(fixture.close)
        return fixture

    def test_rank_query_exception_releases_then_next_query_succeeds(self):
        collective = _TwoRankMin()
        ranks = [
            self.rank(
                rank,
                collective.reduce,
                collective.groups,
                failure=(
                    RuntimeError("DO_NOT_LOG_owned_keys_or_request_body")
                    if rank == 1
                    else None
                ),
            )
            for rank in range(2)
        ]
        first = [rank.request("failed-query") for rank in ranks]
        with self.assertLogs(LOGGER, "ERROR") as logs:
            for rank in ranks:
                rank.start()
            results = [rank.query_result() for rank in ranks]
        self.assertEqual(len(logs.output), 1)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("DO_NOT_LOG", logs.output[0])
        self.assertNotIn("failed-query", logs.output[0])
        self.assertEqual(collective.completed, [(0, 0)])
        self.assertEqual(collective.calls, [1, 1])
        for rank, operation, result in zip(ranks, first, results):
            self.assertIs(result, operation)
            self.assertEqual((result.hash_value, result.storage_hit_count), ([], 0))
            rank.drain([result])
            rank.drain()
            self.assertTrue(operation._prefetch_query_done)
            self.assertEqual(rank.locks, set())
            self.assertEqual(rank.releases, ["failed-query"])
            self.assertEqual(rank.cc.prefetch_tokens_occupied, 0)
            self.assertEqual(rank.cc.mem_pool_host.allocations, [])
            self.assertTrue(rank.cc.prefetch_buffer.empty())
            self.assertTrue(rank.cc.prefetch_hit_queue.empty())
            self.assertEqual(rank.errors, [])
            self.assertTrue(rank.worker.is_alive())

        second = [rank.request("healthy-query") for rank in ranks]
        results = [rank.query_result() for rank in ranks]
        self.assertEqual(collective.completed, [(0, 0), (1, 8)])
        self.assertEqual(collective.calls, [2, 2])
        for rank, operation, result in zip(ranks, second, results):
            self.assertIs(result, operation)
            self.assertEqual(result.storage_hit_count, 8)
            self.assertFalse(result.is_terminated())
            rank.drain([result])
            self.assertIs(rank.cc.prefetch_buffer.get_nowait(), operation)
            self.assertEqual(len(rank.cc.mem_pool_host.allocations), 1)
            self.assertEqual(len(operation.host_indices), 8)
            self.assertEqual(rank.errors, [])
            self.assertEqual(rank.releases, ["failed-query"])
            self.assertEqual(rank.native.calls, 4)

    def test_query_exception_after_revoke_does_not_release_twice(self):
        collective = _TwoRankMin()
        entered, resume = threading.Event(), threading.Event()
        ranks = [
            self.rank(rank, collective.reduce, collective.groups) for rank in range(2)
        ]
        original = ranks[1].native.batch_is_exist

        def native_query(keys):
            if ranks[1].native.calls == 1:
                entered.set()
                if not resume.wait(TIMEOUT):
                    raise TimeoutError("test did not release query barrier")
                raise RuntimeError("injected after cancellation")
            return original(keys)

        ranks[1].native.batch_is_exist = native_query
        first = [rank.request("cancelled-query") for rank in ranks]
        with self.assertLogs(LOGGER, "ERROR"):
            for rank in ranks:
                rank.start()
            try:
                self.assertTrue(entered.wait(TIMEOUT))
                for rank, operation in zip(ranks, first):
                    operation.mark_terminate()
                    rank.tree.revoke_pending_prefetch("cancelled-query")
            finally:
                resume.set()
            results = [rank.query_result() for rank in ranks]
        for rank, operation, result in zip(ranks, first, results):
            self.assertIs(result, operation)
            rank.drain([result])
            rank.drain()
            self.assertTrue(operation._prefetch_query_done)
            self.assertEqual(rank.releases, ["cancelled-query"])
            self.assertEqual(rank.cc.prefetch_tokens_occupied, 0)
            self.assertEqual(rank.cc.mem_pool_host.allocations, [])
            self.assertEqual(rank.errors, [])
            self.assertTrue(rank.cc.prefetch_hit_queue.empty())
        self.assertEqual(collective.calls, [1, 1])

    def test_collective_exception_propagates_without_false_completion(self):
        failure = RuntimeError("collective unavailable")

        def fail_collective(_rank, _tensor, _operation, _groups):
            raise failure

        rank = self.rank(0, fail_collective)
        operation = rank.request("collective-fault")
        with self.assertRaises(RuntimeError) as caught:
            rank.cc.prefetch_thread_func()
        self.assertIs(caught.exception, failure)
        self.assertTrue(rank.cc.prefetch_hit_queue.empty())
        self.assertFalse(operation.is_terminated())
        self.assertFalse(hasattr(operation, "_prefetch_query_done"))
        self.assertEqual(rank.releases, [])

    def test_baseexceptions_propagate_without_joining_collective(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__):
                calls = []
                failure = exception("do not swallow process interruption")
                rank = self.rank(
                    0,
                    lambda *args: calls.append(args),
                    failure=failure,
                    fail_call=1,
                )
                operation = rank.request("interrupted-query")
                with self.assertRaises(exception) as caught:
                    rank.cc.prefetch_thread_func()
                self.assertIs(caught.exception, failure)
                self.assertEqual(calls, [])
                self.assertTrue(rank.cc.prefetch_hit_queue.empty())
                self.assertFalse(operation.is_terminated())

    def test_cancelled_query_skips_native_and_still_publishes(self):
        calls = []
        rank = self.rank(
            0, lambda _rank, tensor, op, _groups: calls.append((tensor.item(), op))
        )
        operation = rank.request("already-cancelled")
        operation.mark_terminate()
        rank.start()
        result = rank.query_result()
        rank.drain([result])
        self.assertEqual(rank.native.calls, 0)
        self.assertEqual(calls, [(0, "MIN")])
        self.assertTrue(operation._prefetch_query_done)
        self.assertEqual(rank.releases, ["already-cancelled"])
        self.assertEqual(rank.cc.prefetch_tokens_occupied, 0)
        self.assertEqual(rank.cc.mem_pool_host.allocations, [])


if __name__ == "__main__":
    unittest.main()
