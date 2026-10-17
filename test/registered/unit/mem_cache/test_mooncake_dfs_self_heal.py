"""Dependency-free CPU regressions for DFS-only repair of rejected cache pages.

Execute the current worktree's MooncakeStore class ASTs with a byte-store model
instead of importing SGLang, torch, or a native Mooncake module. Object bytes
and host allocations are independent; the model's Put preserves an existing
object, while Upsert replaces it. These tests do not validate native pointers,
GPU transfers, controller slot ownership, or cross-process writer coordination.
"""

from __future__ import annotations

import ast
import logging
import runpy
import threading
import time
import unittest
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SOURCE = (
    REPO / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
)

# The registration module uses only the standard library. Loading that file
# directly avoids sglang.__init__, which imports optional runtime dependencies.
register_cpu_ci = runpy.run_path(
    str(REPO / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

POOL_NAMES = (
    "KV MAMBA SWA INDEXER DRAFT DRAFT_INDEXER DRAFT_SWA DEEPSEEK_V4_C4 "
    "DEEPSEEK_V4_C4_INDEXER DEEPSEEK_V4_C128 DEEPSEEK_V4_C4_STATE "
    "DEEPSEEK_V4_C4_INDEXER_STATE DEEPSEEK_V4_C128_STATE"
).split()
PoolName = SimpleNamespace(**{name: name.lower() for name in POOL_NAMES})


@dataclass
class _Transfer:
    name: str
    host_indices: list[int]
    keys: list[str]
    hit_policy: str = "all_pages"


class _Memory:
    def __init__(self):
        self.buffers = {}
        self.next_pointer = 10000

    def allocate(self, size):
        pointer = self.next_pointer
        self.next_pointer += size + 16
        self.buffers[pointer] = bytearray(b"?" * size)
        return pointer


class _HostPool:
    page_size = 64
    kv_buffer = object()

    def __init__(self, memory, parts=(8,)):
        self.memory = memory
        self.parts = parts
        self.pages = {}

    def page(self, index):
        if index not in self.pages:
            self.pages[index] = [self.memory.allocate(size) for size in self.parts]
        return self.pages[index]

    def get_page_buffer_meta(self, indices):
        pointers, sizes = [], []
        for page_index in indices[:: self.page_size]:
            pointers.extend(self.page(page_index))
            sizes.extend(self.parts)
        return pointers, sizes

    def fill(self, index, value):
        if len(value) != sum(self.parts):
            raise ValueError("Source bytes must fill exactly one host page")
        offset = 0
        for pointer, size in zip(self.page(index), self.parts):
            self.memory.buffers[pointer][:] = value[offset : offset + size]
            offset += size

    def read_bytes(self, index):
        return b"".join(bytes(self.memory.buffers[p]) for p in self.page(index))


class _ReplicateConfig:
    def __init__(self):
        self.replica_num = 1
        self.nof_replica_num = 0
        self.dfs_replica_num = 0
        self.group_ids = None


class _ByteStore:
    """A native-response model, not a native API or real pointer implementation."""

    def __init__(self, memory, objects=None, checksums=None):
        self.memory = memory
        self.objects = {} if objects is None else objects
        self.checksums = {} if checksums is None else checksums
        self.read_errors = {}
        self.writes = []
        self.read_calls = []
        self.read_override = None
        self.exists_override = None
        self.readiness_override = None
        self.write_override = None
        self.before_write = None

    def seed(self, key, value):
        self.objects[key] = value
        self.checksums[key] = zlib.crc32(value)

    def batch_is_exist(self, keys):
        if self.exists_override is not None:
            return self.exists_override(keys)
        return [int(key in self.objects) for key in keys]

    def batch_get_replica_desc(self, keys):
        if self.readiness_override is not None:
            return self.readiness_override(keys)
        return {
            key: [object()]
            for key in keys
            if key in self.objects
        }

    def _read(self, keys, pointers, sizes):
        self.read_calls.append(
            {"keys": list(keys), "pointers": [list(row) for row in pointers]}
        )
        if self.read_override is not None:
            return self.read_override(keys)
        results = []
        for key, row, widths in zip(keys, pointers, sizes):
            if key in self.read_errors:
                results.append(self.read_errors[key])
                continue
            value = self.objects.get(key)
            if value is None:
                results.append(-704)
                continue
            if len(value) > sum(widths):
                results.append(-600)
                continue
            if zlib.crc32(value) != self.checksums[key]:
                results.append(-801)
                continue
            offset = 0
            for pointer, width in zip(row, widths):
                copied = min(width, len(value) - offset)
                if copied > 0:
                    self.memory.buffers[pointer][:copied] = value[
                        offset : offset + copied
                    ]
                    offset += copied
            results.append(len(value))
        return results

    def batch_get_into(self, keys, pointers, sizes):
        return self._read(keys, [[p] for p in pointers], [[s] for s in sizes])

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        return self._read(keys, pointers, sizes)

    def _write(self, mode, keys, pointers, sizes, config):
        call = {
            "mode": mode,
            "keys": list(keys),
            "pointers": [list(row) for row in pointers],
            "config": vars(config).copy(),
        }
        self.writes.append(call)
        if self.before_write is not None:
            self.before_write(call)
        values = [
            b"".join(bytes(self.memory.buffers[p][:s]) for p, s in zip(row, widths))
            for row, widths in zip(pointers, sizes)
        ]
        call["values"] = values
        results = (
            self.write_override(keys)
            if self.write_override is not None
            else [0] * len(keys)
        )
        if isinstance(results, (list, tuple)) and len(results) == len(keys):
            for key, value, result in zip(keys, values, results):
                if type(result) is int and result == 0:
                    if mode.startswith("upsert") or key not in self.objects:
                        self.seed(key, value)
                        self.read_errors.pop(key, None)
        return results

    def batch_put_from(self, keys, pointers, sizes, config):
        return self._write(
            "put", keys, [[p] for p in pointers], [[s] for s in sizes], config
        )

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config):
        return self._write("put_multi", keys, pointers, sizes, config)

    def batch_upsert_from(self, keys, pointers, sizes, config):
        return self._write(
            "upsert", keys, [[p] for p in pointers], [[s] for s in sizes], config
        )

    def batch_upsert_from_multi_buffers(self, keys, pointers, sizes, config):
        return self._write("upsert_multi", keys, pointers, sizes, config)


class _AdminResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Requests:
    response = None
    calls = []

    @classmethod
    def get(cls, url, **kwargs):
        cls.calls.append((url, kwargs))
        if cls.response is None:
            raise RuntimeError("admin probe not configured")
        return cls.response


def _load_worktree_classes():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    names = {"_DfsRepairState", "MooncakeBaseStore", "MooncakeStore"}
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    if {node.name for node in classes} != names:
        raise AssertionError("Current MooncakeStore source lacks DFS repair classes")
    future = ast.parse("from __future__ import annotations\n").body
    module = ast.fix_missing_locations(
        ast.Module(body=future + classes, type_ignores=[])
    )
    logger = logging.getLogger("test-mooncake-dfs-self-heal")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    scope = {
        "time": time,
        "threading": threading,
        "OrderedDict": OrderedDict,
        "Sequence": Sequence,
        "PoolName": PoolName,
        "MLATokenToKVPoolHost": _HostPool,
        "HiCacheStorage": type("_HiCacheStorage", (), {}),
        "requests": _Requests,
        "logger": logger,
    }
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["MooncakeStore"], scope["_DfsRepairState"]


class TestMooncakeDfsSelfHeal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store_type, cls.state_type = _load_worktree_classes()

    def fixture(self, parts=(8,), *, dfs=True, mla=True):
        memory = _Memory()
        native = _ByteStore(memory)
        store = self.store_type.__new__(self.store_type)
        store.mem_pool_host = _HostPool(memory, parts)
        store.registered_pools = {PoolName.INDEXER: _HostPool(memory, parts)}
        store.store = native
        store._dfs_repair_state = self.state_type()
        store._dfs_only = dfs
        store._use_dfs_replica = dfs
        store._effective_replica_num = 0 if dfs else 1
        store._replicate_config_cls = _ReplicateConfig
        store._DFS_DUPLICATE_READY_TIMEOUT_S = 0.05
        store.config = SimpleNamespace(
            replica_num=0 if dfs else 1,
            dfs_replica_num=1 if dfs else 0,
            master_server_address="127.0.0.1:53203",
            master_metrics_port=21203,
        )
        store._use_group_semantics = True
        store.config_prefix = "owned-cache"
        store.is_mla_backend = mla
        store.should_split_heads = False
        store.split_factor = 0
        store.mla_suffix = ""
        store.mha_suffix = "0"
        store.enable_storage_metrics = False
        store.backup_pgs = []
        store.backup_bandwidth = []
        store.prefetch_pgs = []
        store.prefetch_bandwidth = []
        store.gb_per_page = sum(parts) / (1 << 30)
        return store, native, memory

    @staticmethod
    def key(kind, name="a"):
        suffix = "k" if kind == "v1" else "indexer"
        return f"owned-cache_{name}__{suffix}"

    def call(self, store, kind, mode, names=("a",)):
        indices = list(range(_HostPool.page_size * len(names)))
        if kind == "v1":
            return getattr(store, f"batch_{mode}_v1")(list(names), indices)
        return getattr(store, f"batch_{mode}_v2")(
            [_Transfer(PoolName.INDEXER, indices, list(names))]
        )[PoolName.INDEXER]

    @staticmethod
    def pool(store, kind):
        if kind == "v1":
            return store.mem_pool_host
        return store.registered_pools[PoolName.INDEXER]

    @staticmethod
    def replace_pool(store, kind, pool):
        if kind == "v1":
            store.mem_pool_host = pool
        else:
            store.registered_pools[PoolName.INDEXER] = pool

    def test_admin_readiness_probe_avoids_ordinary_failure_counter_path(self):
        store, _, _ = self.fixture()
        key = self.key("v1", "admin")
        _Requests.calls = []
        _Requests.response = _AdminResponse(
            {"success": True, "data": {key: {"ok": True}}}
        )
        try:
            result = store._batch_readiness_probe([key])
        finally:
            _Requests.response = None

        self.assertEqual(result, {key: [True]})
        self.assertEqual(len(_Requests.calls), 1)
        self.assertEqual(
            _Requests.calls[0][0], "http://127.0.0.1:21203/batch_query_keys"
        )

    def mark_short(self, store, native, kind, names=("a",)):
        for name in names:
            native.seed(self.key(kind, name), b"broken!")
        self.assertEqual(self.call(store, kind, "get", names), [False] * len(names))

    def assert_pending(self, store, count=1):
        status = store.get_repair_status()
        self.assertEqual(status["pending_objects"], count)
        self.assertEqual(status["inflight_objects"], 0)

    def assert_dfs_config(self, call, names=("a",)):
        config = call["config"]
        self.assertEqual(
            (
                config["replica_num"],
                config["nof_replica_num"],
                config["dfs_replica_num"],
            ),
            (0, 0, 1),
        )
        self.assertEqual(
            config["group_ids"],
            [f"sglang-hicache:owned-cache_{name}" for name in names],
        )

    def fill_tracking_limit(self, store):
        limit = store.get_repair_status()["tracking_limit"]
        store._record_failed_page_reads(
            [f"other-owned-key-{i}" for i in range(limit)], [7] * limit, [8] * limit
        )
        return limit

    def test_short_scalar_pages_repair_from_fresh_recomputed_host_bytes(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, memory = self.fixture()
                self.mark_short(store, native, kind)
                rejected_pointers = native.read_calls[-1]["pointers"]
                self.assertEqual(native.writes, [])

                recomputed = _HostPool(memory)
                recomputed.fill(0, b"correct!")
                self.replace_pool(store, kind, recomputed)
                self.assertEqual(self.call(store, kind, "set"), [True])
                write = native.writes[0]
                self.assertEqual(write["mode"], "upsert")
                self.assertNotEqual(write["pointers"], rejected_pointers)
                self.assertEqual(write["values"], [b"correct!"])
                self.assert_dfs_config(write)
                self.assert_pending(store, 0)

                # Fresh allocations and a second response-model client prevent
                # a successful result from merely rereading the write buffer.
                replacement = _HostPool(memory)
                self.replace_pool(store, kind, replacement)
                store.store = _ByteStore(memory, native.objects, native.checksums)
                self.assertEqual(self.call(store, kind, "get"), [True])
                self.assertEqual(replacement.read_bytes(0), b"correct!")

    def test_packed_v1_and_indexer_repair_preserve_all_slice_bytes(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, memory = self.fixture((3, 5))
                self.mark_short(store, native, kind)
                self.pool(store, kind).fill(0, b"abcDEFGH")
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.writes[0]["mode"], "upsert_multi")
                self.assertEqual(native.objects[self.key(kind)], b"abcDEFGH")
                self.assert_dfs_config(native.writes[0])
                replacement = _HostPool(memory, (3, 5))
                self.replace_pool(store, kind, replacement)
                self.assertEqual(self.call(store, kind, "get"), [True])
                self.assertEqual(replacement.read_bytes(0), b"abcDEFGH")

    def test_checksum_rejection_repairs_only_on_later_recomputed_backup(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                key = self.key(kind)
                native.seed(key, b"original")
                native.objects[key] = b"bad-data"  # Same size, retained old CRC.
                self.assertEqual(self.call(store, kind, "get"), [False])
                self.assertEqual(native.writes, [])
                self.assertEqual(native.objects[key], b"bad-data")
                self.assert_pending(store)
                self.pool(store, kind).fill(0, b"original")
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.objects[key], b"original")
                self.assertEqual(self.call(store, kind, "get"), [True])

    def test_healthy_existing_pages_skip_writes(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                native.seed(self.key(kind), b"healthy!")
                self.assertEqual(self.call(store, kind, "get"), [True])
                self.pool(store, kind).fill(0, b"new-data")
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.objects[self.key(kind)], b"healthy!")
                self.assertEqual(native.writes, [])
                self.assert_pending(store, 0)

    def test_missing_pages_use_ordinary_put_with_dfs_only_topology(self):
        for kind in ("v1", "v2"):
            for parts in ((8,), (3, 5)):
                with self.subTest(kind=kind, parts=parts):
                    store, native, _ = self.fixture(parts)
                    self.pool(store, kind).fill(0, b"new-data")
                    self.assertEqual(self.call(store, kind, "set"), [True])
                    expected = "put" if len(parts) == 1 else "put_multi"
                    self.assertEqual(native.writes[0]["mode"], expected)
                    self.assert_dfs_config(native.writes[0])

    def test_mixed_batch_preserves_object_identity_and_filtered_group_ids(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind, ("repair",))
                native.seed(self.key(kind, "healthy"), b"healthy!")
                for index, value in enumerate((b"correct!", b"ignored!", b"new-data")):
                    self.pool(store, kind).fill(index * _HostPool.page_size, value)
                self.assertEqual(
                    self.call(store, kind, "set", ("repair", "healthy", "new")),
                    [True, True, True],
                )
                calls = {call["mode"]: call for call in native.writes}
                self.assertEqual(set(calls), {"put", "upsert"})
                self.assertEqual(calls["upsert"]["keys"], [self.key(kind, "repair")])
                self.assertEqual(calls["put"]["keys"], [self.key(kind, "new")])
                self.assert_dfs_config(calls["upsert"], ("repair",))
                self.assert_dfs_config(calls["put"], ("new",))
                self.assertEqual(native.objects[self.key(kind, "healthy")], b"healthy!")

    def test_non_dfs_path_keeps_original_existing_object_behavior(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture(dfs=False)
                self.mark_short(store, native, kind)
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.writes, [])
                self.assert_pending(store, 0)

    def test_transient_zero_and_oversized_reads_do_not_trigger_upsert(self):
        for kind in ("v1", "v2"):
            for code in (-200, -600, -703, -704, -707, -900, -901, -1102, 0, 9):
                with self.subTest(kind=kind, code=code):
                    store, native, _ = self.fixture()
                    native.seed(self.key(kind), b"healthy!")
                    native.read_errors[self.key(kind)] = code
                    self.assertEqual(self.call(store, kind, "get"), [False])
                    self.assertEqual(self.call(store, kind, "set"), [True])
                    self.assertEqual(native.writes, [])
                    self.assert_pending(store, 0)

    def test_malformed_read_vectors_fail_closed_without_markers(self):
        responses = ([], [7], [7, 7, 7], [7, True], [7, 7.0], None, True, 7, {})
        for kind in ("v1", "v2"):
            for response in responses:
                with self.subTest(kind=kind, response=response):
                    store, native, _ = self.fixture()
                    native.read_override = lambda keys, r=response: r
                    self.assertEqual(
                        self.call(store, kind, "get", ("a", "b")), [False, False]
                    )
                    self.assert_pending(store, 0)
                    self.assertEqual(native.writes, [])

    def test_mismatched_recompute_size_does_not_claim_or_write(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, memory = self.fixture()
                self.mark_short(store, native, kind)
                replacement = _HostPool(memory, (9,))
                replacement.fill(0, b"different")
                self.replace_pool(store, kind, replacement)
                self.assertEqual(self.call(store, kind, "set"), [False])
                self.assertEqual(native.writes, [])
                self.assert_pending(store)

    def test_unknown_or_malformed_exists_never_acknowledges_backup(self):
        for kind in ("v1", "v2"):
            for marked in (False, True):
                for response in ([-900], [], [1, 1], [True], [0.0], None, 1):
                    with self.subTest(kind=kind, marked=marked, response=response):
                        store, native, _ = self.fixture()
                        if marked:
                            self.mark_short(store, native, kind)
                        native.exists_override = lambda keys, r=response: r
                        self.assertEqual(self.call(store, kind, "set"), [False])
                        self.assertEqual(native.writes, [])
                        self.assert_pending(store, int(marked))

    def test_missing_scalar_or_packed_upsert_keeps_repair_pending(self):
        for kind in ("v1", "v2"):
            for parts, method in (
                ((8,), "batch_upsert_from"),
                ((3, 5), "batch_upsert_from_multi_buffers"),
            ):
                with self.subTest(kind=kind, method=method):
                    store, native, _ = self.fixture(parts)
                    self.mark_short(store, native, kind)
                    setattr(native, method, None)
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assertEqual(native.writes, [])
                    self.assert_pending(store)

    def test_failed_upsert_keeps_marker_then_retries_recomputed_bytes(self):
        for kind in ("v1", "v2"):
            for parts in ((8,), (3, 5)):
                with self.subTest(kind=kind, parts=parts):
                    store, native, _ = self.fixture(parts)
                    self.mark_short(store, native, kind)
                    self.pool(store, kind).fill(0, b"correct!")
                    native.write_override = lambda keys: [-1103] * len(keys)
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assertEqual(native.objects[self.key(kind)], b"broken!")
                    self.assert_pending(store)
                    native.write_override = None
                    self.pool(store, kind).fill(0, b"retried!")
                    self.assertEqual(self.call(store, kind, "set"), [True])
                    self.assertEqual(native.objects[self.key(kind)], b"retried!")
                    self.assert_pending(store, 0)

    def test_native_exception_returns_failure_and_releases_claim(self):
        def raising(keys):
            raise RuntimeError("synthetic native failure")

        for kind in ("v1", "v2"):
            for phase in ("exists", "write"):
                with self.subTest(kind=kind, phase=phase):
                    store, native, _ = self.fixture()
                    self.mark_short(store, native, kind)
                    setattr(native, f"{phase}_override", raising)
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assert_pending(store)
                    setattr(native, f"{phase}_override", None)
                    self.pool(store, kind).fill(0, b"retried!")
                    self.assertEqual(self.call(store, kind, "set"), [True])

    def test_malformed_write_vectors_fail_closed_for_put_and_upsert(self):
        responses = ([], [0, 0], [False], [0.0], [-200], [None], None, True)
        for kind in ("v1", "v2"):
            for marked in (False, True):
                for response in responses:
                    with self.subTest(kind=kind, marked=marked, response=response):
                        store, native, _ = self.fixture()
                        if marked:
                            self.mark_short(store, native, kind)
                        native.write_override = lambda keys, r=response: r
                        self.assertEqual(self.call(store, kind, "set"), [False])
                        self.assert_pending(store, int(marked))

    def test_partial_upsert_failure_preserves_all_claims_for_retry(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind, ("a", "b"))
                self.pool(store, kind).fill(0, b"first-ok")
                self.pool(store, kind).fill(_HostPool.page_size, b"secondok")
                native.write_override = lambda keys: [0, -200]
                self.assertEqual(
                    self.call(store, kind, "set", ("a", "b")), [False, False]
                )
                self.assertEqual(native.objects[self.key(kind, "a")], b"first-ok")
                self.assertEqual(native.objects[self.key(kind, "b")], b"broken!")
                self.assert_pending(store, 2)
                native.write_override = None
                self.assertEqual(
                    self.call(store, kind, "set", ("a", "b")), [True, True]
                )
                self.assert_pending(store, 0)

    def test_new_failure_generation_survives_older_successful_upsert(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind)
                key = self.key(kind)
                old_generation = store._dfs_repair_state.markers[key][1]
                self.pool(store, kind).fill(0, b"correct!")
                # A previously started read reports a short page during repair.
                native.before_write = lambda call: store._record_failed_page_reads(
                    [key], [7], [8]
                )
                self.assertEqual(self.call(store, kind, "set"), [False])
                self.assertEqual(native.objects[key], b"correct!")
                self.assertGreater(
                    store._dfs_repair_state.markers[key][1], old_generation
                )
                self.assert_pending(store)
                native.before_write = None
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assert_pending(store, 0)

    def test_same_key_claim_is_exclusive_without_holding_lock_during_io(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind)
                self.pool(store, kind).fill(0, b"correct!")
                entered, released = threading.Event(), threading.Event()
                results, errors = [], []

                def blocked(call):
                    entered.set()
                    if not released.wait(3):
                        raise RuntimeError("test synchronization timed out")

                def write():
                    try:
                        results.append(self.call(store, kind, "set"))
                    except BaseException as error:
                        errors.append(error)

                native.before_write = blocked
                worker = threading.Thread(target=write, daemon=True)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    lock = store._dfs_repair_state.lock
                    acquired = lock.acquire(blocking=False)
                    self.assertTrue(acquired, "metadata lock held during native I/O")
                    if acquired:
                        lock.release()
                    self.assertEqual(store.get_repair_status()["inflight_objects"], 1)
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assertEqual(len(native.writes), 1)
                finally:
                    released.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(results, [[True]])
                self.assert_pending(store, 0)

    def test_cancellation_propagates_preserving_marker_and_releasing_claim(self):
        for kind in ("v1", "v2"):
            for exception in (KeyboardInterrupt, SystemExit):
                with self.subTest(kind=kind, exception=exception.__name__):
                    store, native, _ = self.fixture()
                    self.mark_short(store, native, kind)

                    def cancel(call):
                        raise exception("synthetic cancellation")

                    native.before_write = cancel
                    with self.assertRaises(exception):
                        self.call(store, kind, "set")
                    self.assertEqual(native.objects[self.key(kind)], b"broken!")
                    self.assert_pending(store)
                    native.before_write = None
                    self.pool(store, kind).fill(0, b"retried!")
                    self.assertEqual(self.call(store, kind, "set"), [True])

    def test_overflow_enters_recovery_write_mode_and_degraded_state_is_sticky(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind)
                limit = self.fill_tracking_limit(store)
                self.assertEqual(limit, 4096)
                status = store.get_repair_status()
                self.assertEqual(status["pending_objects"], limit)
                self.assertTrue(status["degraded"])
                self.assertEqual(status["overflow_events"], 1)
                self.pool(store, kind).fill(0, b"correct!")
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assert_pending(store, limit - 1)
                self.assertTrue(store.get_repair_status()["degraded"])
                native.seed(self.key(kind, "healthy"), b"healthy!")
                # Lost failure identity means even a reported miss must be
                # rewritten: Put can race with a duplicate and skip its bytes.
                self.assertEqual(self.call(store, kind, "set", ("healthy",)), [True])
                self.assertEqual(self.call(store, kind, "set", ("missing",)), [True])
                self.assertEqual(len(native.writes), 3)
                self.assertEqual(native.writes[1]["mode"], "upsert")
                self.assertEqual(native.writes[2]["mode"], "upsert")
                writes_before_repeat = len(native.writes)
                self.assertEqual(self.call(store, kind, "set", ("healthy",)), [True])
                self.assertEqual(len(native.writes), writes_before_repeat)
                self.assertGreaterEqual(store.get_repair_status()["recovered_objects"], 1)

    def test_degraded_untracked_failure_never_gets_acknowledged_as_duplicate(self):
        for kind in ("v1", "v2"):
            for exists in (-900, 0, 1):
                with self.subTest(kind=kind, exists=exists):
                    store, native, _ = self.fixture()
                    limit = self.fill_tracking_limit(store)
                    self.mark_short(store, native, kind)
                    native.exists_override = (
                        lambda keys, value=exists: [value] * len(keys)
                    )
                    self.pool(store, kind).fill(0, b"correct!")
                    expected = exists in (0, 1)
                    self.assertEqual(self.call(store, kind, "set"), [expected])
                    if exists == -900:
                        self.assertEqual(native.writes, [])
                        self.assertEqual(native.objects[self.key(kind)], b"broken!")
                    else:
                        self.assertEqual(native.writes[0]["mode"], "upsert")
                        self.assertEqual(native.objects[self.key(kind)], b"correct!")
                    self.assert_pending(store, limit)

    def test_failed_read_arriving_during_put_prevents_duplicate_noop_ack(self):
        for kind in ("v1", "v2"):
            for overflow in (False, True):
                with self.subTest(kind=kind, overflow=overflow):
                    store, native, _ = self.fixture()
                    if overflow:
                        self.fill_tracking_limit(store)
                    self.pool(store, kind).fill(0, b"correct!")

                    def racing_put(call):
                        key = call["keys"][0]
                        native.seed(key, b"broken!")
                        store._record_failed_page_reads([key], [7], [8])

                    native.before_write = racing_put
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assertEqual(native.writes[0]["mode"], "put")
                    self.assertEqual(native.objects[self.key(kind)], b"broken!")
                    self.assertEqual(store.get_repair_status()["degraded"], overflow)

    def test_degraded_recovery_upsert_race_requires_retry(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.fill_tracking_limit(store)
                key = self.key(kind, "healthy")
                store._record_failed_page_reads([key], [7], [8])
                native.seed(key, b"healthy!")
                self.pool(store, kind).fill(0, b"current!")

                def racing_recovery(_call):
                    store._record_failed_page_reads([key], [7], [8])

                native.before_write = racing_recovery
                self.assertEqual(self.call(store, kind, "set", ("healthy",)), [False])
                self.assertEqual(native.writes[0]["mode"], "upsert")
                native.before_write = None
                self.assertEqual(self.call(store, kind, "set", ("healthy",)), [True])
                self.assertEqual(len(native.writes), 2)

    def test_failure_after_classification_cannot_be_hidden_by_new_snapshot(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.fill_tracking_limit(store)
                key = self.key(kind)
                self.pool(store, kind).fill(0, b"current!")
                classify = store._dfs_repair_state.classify

                def classification_race(keys, sizes, exists):
                    result = classify(keys, sizes, exists)
                    native.seed(key, b"broken!")
                    store._record_failed_page_reads([key], [7], [8])
                    return result

                store._dfs_repair_state.classify = classification_race
                self.assertEqual(self.call(store, kind, "set"), [False])
                self.assertEqual(native.objects[key], b"broken!")
                store._dfs_repair_state.classify = classify
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.objects[key], b"current!")

    def test_degraded_same_key_claim_is_exclusive_and_bounded(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.fill_tracking_limit(store)
                self.mark_short(store, native, kind)
                self.pool(store, kind).fill(0, b"correct!")
                entered, released = threading.Event(), threading.Event()
                results, errors = [], []

                def blocked(_call):
                    entered.set()
                    if not released.wait(3):
                        raise RuntimeError("test synchronization timed out")

                def write():
                    try:
                        results.append(self.call(store, kind, "set"))
                    except BaseException as error:
                        errors.append(error)

                native.before_write = blocked
                worker = threading.Thread(target=write, daemon=True)
                worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    self.assertEqual(store.get_repair_status()["inflight_objects"], 1)
                    self.assertEqual(self.call(store, kind, "set"), [False])
                    self.assertEqual(len(native.writes), 1)
                    lock = store._dfs_repair_state.lock
                    acquired = lock.acquire(blocking=False)
                    self.assertTrue(acquired, "metadata lock held during native I/O")
                    if acquired:
                        lock.release()
                finally:
                    released.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(results, [[True]])
                self.assertEqual(native.objects[self.key(kind)], b"correct!")

    def test_stale_claim_cannot_finish_a_new_claim(self):
        state = self.state_type()
        state.record([("key", 8)])
        _, _, first, epoch = state.classify(["key"], [8], [1])
        first_claim = first[0][2]
        self.assertFalse(state.finish("key", first_claim, False))
        _, _, second, _ = state.classify(["key"], [8], [1])
        second_claim = second[0][2]
        self.assertNotEqual(first_claim, second_claim)
        self.assertFalse(state.finish("key", first_claim, True))
        self.assertEqual(len(state.markers), 1)
        self.assertFalse(state.finish("key", second_claim, False))

    def test_recovered_lru_eviction_reenters_upsert_claim(self):
        state = self.state_type()
        state.degraded = True
        with state.lock:
            for index in range(state.LIMIT + 1):
                state._remember_recovered_locked(f"key-{index}")
        self.assertEqual(len(state.recovered), state.LIMIT)
        existing, ordinary, claims, _ = state.classify(["key-0"], [8], [1])
        self.assertEqual((existing, ordinary), ([], []))
        self.assertEqual(len(claims), 1)
        self.assertFalse(state.finish("key-0", claims[0][2], False))

    def test_marker_slot_is_reused_after_successful_repair(self):
        state = self.state_type()
        failures = [(f"key-{index}", 8) for index in range(state.LIMIT)]
        state.record(failures)
        _, _, claims, _ = state.classify(["key-0"], [8], [1])
        self.assertTrue(state.finish("key-0", claims[0][2], True))
        state.record([("new-key", 8)])
        self.assertIn("new-key", state.markers)
        self.assertEqual(len(state.markers), state.LIMIT)

    def test_mha_repairs_only_rejected_component(self):
        store, native, _ = self.fixture((4, 4), mla=False)
        native.seed("owned-cache_a_0_k", b"bad")
        native.seed("owned-cache_a_0_v", b"okay")
        self.assertEqual(self.call(store, "v1", "get"), [False])
        store.mem_pool_host.fill(0, b"goodokay")
        self.assertEqual(self.call(store, "v1", "set"), [True])
        self.assertEqual(native.writes[0]["keys"], ["owned-cache_a_0_k"])
        self.assertEqual(native.objects["owned-cache_a_0_k"], b"good")
        self.assertEqual(native.objects["owned-cache_a_0_v"], b"okay")
        self.assert_dfs_config(native.writes[0])

    def test_existing_marker_can_be_repaired_after_metadata_disappears(self):
        for kind in ("v1", "v2"):
            with self.subTest(kind=kind):
                store, native, _ = self.fixture()
                self.mark_short(store, native, kind)
                native.objects.pop(self.key(kind))
                self.pool(store, kind).fill(0, b"restored")
                self.assertEqual(self.call(store, kind, "set"), [True])
                self.assertEqual(native.writes[0]["mode"], "upsert")
                self.assertEqual(native.objects[self.key(kind)], b"restored")
                self.assert_pending(store, 0)

    def test_generic_set_repairs_a_checksum_failure(self):
        for parts in ((8,), (3, 5)):
            with self.subTest(parts=parts):
                store, native, _ = self.fixture(parts)
                key = self.key("v1")
                self.mark_short(store, native, "v1")
                pointers, sizes = self.pool(store, "v1").get_page_buffer_meta([0])
                self.assert_pending(store)
                self.pool(store, "v1").fill(0, b"restored")
                self.assertTrue(store.set(key, target_location=pointers, target_sizes=sizes))
                self.assertEqual(native.writes[0]["mode"], "upsert_multi")
                self.assertEqual(native.objects[key], b"restored")
                self.assert_pending(store, 0)

    def test_generic_batch_set_repairs_and_puts_in_one_call(self):
        store, native, _ = self.fixture((8,))
        broken = self.key("v1", "broken")
        fresh = self.key("v1", "fresh")
        self.mark_short(store, native, "v1", ("broken",))
        pointers, sizes = self.pool(store, "v1").get_page_buffer_meta(
            list(range(128))
        )
        self.assert_pending(store)
        self.pool(store, "v1").fill(0, b"restored")
        self.pool(store, "v1").fill(64, b"new-data")
        self.assertTrue(
            store.batch_set(
                [broken, fresh],
                target_locations=[[pointers[0]], [pointers[1]]],
                target_sizes=[[sizes[0]], [sizes[1]]],
            )
        )
        self.assertEqual(
            [write["mode"] for write in native.writes], ["put_multi", "upsert_multi"]
        )
        self.assertEqual(native.objects[broken], b"restored")
        self.assertEqual(native.objects[fresh], b"new-data")
        self.assert_pending(store, 0)

    def test_generic_get_keeps_capacity_semantics_and_does_not_track_short_copy(self):
        store, native, memory = self.fixture((4,))
        key = self.key("v1", "capacity")
        native.seed(key, b"four")
        pointer = memory.allocate(8)
        memory.buffers[pointer][:] = b"????????"
        self.assertTrue(store.get(key, [pointer], [8]))
        self.assertEqual(bytes(memory.buffers[pointer]), b"four????")
        self.assert_pending(store, 0)

    def test_generic_mha_batch_get_keeps_page_count_and_metrics(self):
        store, native, memory = self.fixture((4,), mla=False)
        store.enable_storage_metrics = True
        native.seed("owned-cache-a", b"four")
        native.seed("owned-cache-b", b"four")
        pointers = [memory.allocate(8), memory.allocate(8)]
        self.assertEqual(
            store.batch_get(
                ["owned-cache-a", "owned-cache-b"],
                [[pointers[0]], [pointers[1]]],
                [[8], [8]],
            ),
            1,
        )
        self.assertEqual(len(store.prefetch_pgs), 1)
        self.assertEqual(len(store.prefetch_bandwidth), 1)

    def test_dfs_existing_processing_key_is_not_acknowledged(self):
        store, native, _ = self.fixture()
        key = self.key("v1", "processing")
        native.seed(key, b"remote")
        native.readiness_override = lambda keys: {}
        pointers, sizes = self.pool(store, "v1").get_page_buffer_meta([0])
        self.pool(store, "v1").fill(0, b"current!")

        self.assertFalse(store.set(key, target_location=pointers, target_sizes=sizes))
        self.assertEqual(native.writes, [])

    def test_dfs_existing_complete_key_keeps_idempotent_success(self):
        store, native, _ = self.fixture()
        key = self.key("v1", "complete")
        native.seed(key, b"remote")
        pointers, sizes = self.pool(store, "v1").get_page_buffer_meta([0])
        self.pool(store, "v1").fill(0, b"current!")

        self.assertTrue(store.set(key, target_location=pointers, target_sizes=sizes))
        self.assertEqual(native.writes, [])

    def test_dfs_existing_processing_key_waits_for_complete(self):
        store, native, _ = self.fixture()
        key = self.key("v1", "eventual")
        native.seed(key, b"remote")
        calls = []

        def becomes_complete(keys):
            calls.append(list(keys))
            return {} if len(calls) == 1 else {key: [object()]}

        native.readiness_override = becomes_complete
        pointers, sizes = self.pool(store, "v1").get_page_buffer_meta([0])
        self.pool(store, "v1").fill(0, b"current!")

        self.assertTrue(store.set(key, target_location=pointers, target_sizes=sizes))
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(native.writes, [])

    def test_dfs_revoked_duplicate_restarts_put(self):
        store, native, _ = self.fixture()
        key = self.key("v1", "revoked")
        native.seed(key, b"remote")
        exists_calls = []
        native.readiness_override = lambda keys: {}

        def revoked(keys):
            exists_calls.append(list(keys))
            return [1] * len(keys) if len(exists_calls) == 1 else [0] * len(keys)

        native.exists_override = revoked
        pointers, sizes = self.pool(store, "v1").get_page_buffer_meta([0])
        self.pool(store, "v1").fill(0, b"current!")

        self.assertTrue(store.set(key, target_location=pointers, target_sizes=sizes))
        self.assertEqual([write["mode"] for write in native.writes], ["put_multi"])


if __name__ == "__main__":
    unittest.main()
