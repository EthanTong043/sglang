"""DFS 副本透传的单元测试。

背景：SGLang 的 KV cache 写入走 batch_put_from。若不显式传
ReplicateConfig，Mooncake C++ 侧用默认构造的配置，其 dfs_replica_num 为 0
（replica.h:107）—— 意味着挂载的并行文件系统（GPFS）**永远不会被写入**。

这一组用例验证 dfs_replica_num 能正确地从配置传到 ReplicateConfig。
"""

import importlib
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# 被测的 ReplicateConfig 替身
#
# 用三种形态覆盖能力探测的分支：
#   - 有 dfs_replica_num：正常路径
#   - 完全没有该属性（__slots__ 阻止动态设置）：旧版 Mooncake
#   - 实例属性只在构造后可见：pybind 某些版本的行为
# ---------------------------------------------------------------------------


class ReplicateConfigWithDfs:
    def __init__(self):
        self.replica_num = 1
        self.dfs_replica_num = 0
        self.group_ids = None


class ReplicateConfigWithoutDfs:
    """模拟旧版 Mooncake：没有 dfs_replica_num 字段。

    用 __slots__ 让动态 setattr 失败，与 pybind 对象的行为一致 ——
    这正是"盲目 setattr 不可靠"所以要显式探测的原因。
    """

    __slots__ = ("replica_num",)

    def __init__(self):
        self.replica_num = 1


class ReplicateConfigWithBoth:
    """同时支持 group_ids 与 dfs_replica_num，验证两者能共存于一个 config。"""

    def __init__(self):
        self.group_ids = None
        self.replica_num = 1
        self.dfs_replica_num = 0


def _fake_mooncake_modules(fake_store_cls, replicate_config_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    mooncake_store.ReplicateConfig = replicate_config_cls
    return {"mooncake": mooncake, "mooncake.store": mooncake_store}


@contextmanager
def _mock_modules(modules):
    # Restore only the explicitly mocked entries. patch.dict(sys.modules)
    # also removes dependencies imported by the real backend during the test;
    # importing Torch compiler modules again can re-register native globals.
    missing = object()
    backend_name = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
    parent_name, _, attribute = backend_name.rpartition(".")
    old_backend = sys.modules.pop(backend_name, missing)
    old_parent = sys.modules.get(parent_name)
    old_attribute = getattr(old_parent, attribute, missing)
    if old_parent is not None:
        old_parent.__dict__.pop(attribute, None)
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        # A backend imported against fake host-pool classes must not leak into
        # later tests; restore its module and parent-package binding together.
        if old_backend is missing:
            sys.modules.pop(backend_name, None)
        else:
            sys.modules[backend_name] = old_backend
        parent = sys.modules.get(parent_name)
        if parent is not None:
            if old_attribute is missing:
                parent.__dict__.pop(attribute, None)
            else:
                setattr(parent, attribute, old_attribute)
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _fake_pool_host_mla_module():
    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class MLATokenToKVPoolHost:
        pass

    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return pool_host_mla


def _fake_pool_host_module():
    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator
    return pool_host


def _fake_host_pool_modules():
    return {
        "sglang.srt.mem_cache.pool_host": _fake_pool_host_module(),
        "sglang.srt.mem_cache.pool_host.mla": _fake_pool_host_mla_module(),
    }


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.batch_put_calls = []
            self.put_calls = []
            self.remove_calls = []
            self.setup_calls = []
            self.existing_keys = set()
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            self.setup_calls.append((args, kwargs))
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        # warmup() 会用 put / is_exist / get 做一次探针读写，
        # 缺少任何一个都会在构造阶段抛 AttributeError
        def put(self, key, value, *args, **kwargs):
            self.put_calls.append((key, value, args, kwargs))
            self.objects[key] = value
            return 0

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def get(self, key):
            return self.objects.get(key)

        def remove(self, key, force=False):
            self.remove_calls.append((key, force))
            self.objects.pop(key, None)
            return 0

        def batch_is_exist(self, keys):
            return [1 if key in self.existing_keys else 0 for key in keys]

        def batch_put_from(self, keys, ptrs, sizes, *args):
            self.batch_put_calls.append(
                {
                    "method": "batch_put_from",
                    "keys": list(keys),
                    "args": args,
                }
            )
            self.existing_keys.update(keys)
            return [0] * len(keys)

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
            self.batch_put_calls.append(
                {
                    "method": "batch_put_from_multi_buffers",
                    "keys": list(keys),
                    "args": args,
                }
            )
            self.existing_keys.update(keys)
            return [0] * len(keys)

    return FakeMooncakeDistributedStore


class FakeHostKVCache:
    def __init__(self):
        self.kv_buffer = torch.empty((1024,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        ptrs = [1000 + i for i in range(len(indices))]
        sizes = [8] * len(indices)
        return ptrs, sizes


def _make_config(
    *,
    dfs_replica_num=None,
    enable_group_semantics=False,
    dfs_only=False,
    replica_num=None,
    enable_ssd_offload=False,
    standalone_storage=False,
    global_segment_size=1024 * 1024,
    check_server=False,
):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": check_server,
        "global_segment_size": global_segment_size,
        "enable_group_semantics": enable_group_semantics,
        "dfs_only": dfs_only,
        "enable_ssd_offload": enable_ssd_offload,
        "standalone_storage": standalone_storage,
    }
    if replica_num is not None:
        extra_config["replica_num"] = replica_num
    if dfs_replica_num is not None:
        extra_config["dfs_replica_num"] = dfs_replica_num

    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config=extra_config,
    )


def _make_store(
    *,
    dfs_replica_num=None,
    replicate_config_cls=ReplicateConfigWithDfs,
    enable_group_semantics=False,
    dfs_only=False,
    replica_num=None,
    enable_ssd_offload=False,
    standalone_storage=False,
    global_segment_size=1024 * 1024,
    check_server=False,
):
    fake_store_cls = _fake_store_class()
    modules = _fake_mooncake_modules(fake_store_cls, replicate_config_cls)
    modules.update(_fake_host_pool_modules())

    with _mock_modules(modules):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(
            _make_config(
                dfs_replica_num=dfs_replica_num,
                enable_group_semantics=enable_group_semantics,
                dfs_only=dfs_only,
                replica_num=replica_num,
                enable_ssd_offload=enable_ssd_offload,
                standalone_storage=standalone_storage,
                global_segment_size=global_segment_size,
                check_server=check_server,
            ),
            FakeHostKVCache(),
        )
    return store, fake_store_cls


class TestMooncakeDfsReplicaNum(unittest.TestCase):
    def test_disabled_by_default(self):
        """不配置时不传 ReplicateConfig —— 保持改动前的行为。"""
        store, fake_store_cls = _make_store()

        self.assertFalse(store._use_dfs_replica)

        store._put_batch_zero_copy_impl(["k0"], [1000], [8])
        call = fake_store_cls.instances[0].batch_put_calls[-1]
        self.assertEqual(call["method"], "batch_put_from")
        # 三参重载：没有 ReplicateConfig 位置参数
        self.assertEqual(len(call["args"]), 0)

    def test_explicit_zero_keeps_legacy_path(self):
        """显式配 0 与不配等价。"""
        store, fake_store_cls = _make_store(dfs_replica_num=0)

        self.assertFalse(store._use_dfs_replica)

        store._put_batch_zero_copy_impl(["k0"], [1000], [8])
        call = fake_store_cls.instances[0].batch_put_calls[-1]
        self.assertEqual(len(call["args"]), 0)

    def test_dfs_replica_num_is_forwarded(self):
        """★ 核心：配了 1 就必须把 dfs_replica_num 设进 ReplicateConfig。

        不做这一步，GPFS 永远不会被写入 —— 这正是本改动存在的理由。
        """
        store, fake_store_cls = _make_store(dfs_replica_num=1)

        self.assertTrue(store._use_dfs_replica)

        store._put_batch_zero_copy_impl(["k0"], [1000], [8])
        call = fake_store_cls.instances[0].batch_put_calls[-1]
        self.assertEqual(call["method"], "batch_put_from")
        self.assertEqual(len(call["args"]), 1, "must pass a ReplicateConfig")

        config = call["args"][0]
        self.assertEqual(config.dfs_replica_num, 1)

    def test_unsupported_package_degrades_without_crashing(self):
        """旧版 Mooncake 没有该字段：不崩溃、不启用，只是写不了 DFS。"""
        store, fake_store_cls = _make_store(
            dfs_replica_num=1, replicate_config_cls=ReplicateConfigWithoutDfs
        )

        self.assertFalse(
            store._use_dfs_replica,
            "must not claim DFS support when the field does not exist",
        )

        # 仍然能正常写入（只是没有 DFS 副本）
        store._put_batch_zero_copy_impl(["k0"], [1000], [8])
        call = fake_store_cls.instances[0].batch_put_calls[-1]
        self.assertEqual(len(call["args"]), 0)

    def test_coexists_with_group_ids(self):
        """group_ids 与 dfs_replica_num 共用一个 config 对象，互不覆盖。"""
        store, fake_store_cls = _make_store(
            dfs_replica_num=1,
            replicate_config_cls=ReplicateConfigWithBoth,
            enable_group_semantics=True,
        )

        self.assertTrue(store._use_dfs_replica)
        self.assertTrue(store._use_group_semantics)

        store._put_batch_zero_copy_impl(
            ["k0", "k1"], [1000, 1001], [8, 8], group_ids=["g0", "g1"]
        )
        config = fake_store_cls.instances[0].batch_put_calls[-1]["args"][0]
        self.assertEqual(config.dfs_replica_num, 1)
        self.assertEqual(config.group_ids, ["g0", "g1"])


class TestMooncakeModuleIsolation(unittest.TestCase):
    def _check_constructor_isolation(self, *, existing_backend, raise_after_init):
        backend_name = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
        parent_name, _, attribute = backend_name.rpartition(".")
        parent = importlib.import_module(parent_name)
        dependency_name = "_sglang_mooncake_test_new_dependency"
        dependency = types.ModuleType(dependency_name)
        prior_backend = types.ModuleType(backend_name)
        missing = object()
        saved_backend = sys.modules.get(backend_name, missing)
        saved_attribute = parent.__dict__.get(attribute, missing)
        saved_dependency = sys.modules.get(dependency_name, missing)
        fake_store_cls = _fake_store_class()
        mocks = _fake_mooncake_modules(fake_store_cls, ReplicateConfigWithDfs)
        mocks.update(_fake_host_pool_modules())
        saved_mocks = {name: sys.modules.get(name, missing) for name in mocks}
        try:
            if existing_backend:
                sys.modules[backend_name] = prior_backend
                setattr(parent, attribute, prior_backend)
            else:
                sys.modules.pop(backend_name, None)
                parent.__dict__.pop(attribute, None)
            sys.modules.pop(dependency_name, None)

            outcome = (
                self.assertRaisesRegex(RuntimeError, "after real constructor")
                if raise_after_init
                else nullcontext()
            )
            with outcome:
                with _mock_modules(mocks):
                    self.assertNotIn(backend_name, sys.modules)
                    self.assertNotIn(attribute, parent.__dict__)
                    from sglang.srt.mem_cache.storage.mooncake_store import (
                        mooncake_store as backend,
                    )

                    # Exercise the complete module and constructor, with only
                    # the native client and host-pool dependencies replaced.
                    store = backend.MooncakeStore(
                        _make_config(
                            dfs_replica_num=1,
                            dfs_only=True,
                            replica_num=0,
                            enable_ssd_offload=True,
                        ),
                        FakeHostKVCache(),
                    )
                    self.assertIs(type(store), backend.MooncakeStore)
                    self.assertIs(sys.modules[backend_name], backend)
                    self.assertIs(getattr(parent, attribute), backend)
                    self.assertIsNot(backend, prior_backend)
                    self.assertEqual(store.get_repair_status()["pending_objects"], 0)
                    self.assertEqual(len(fake_store_cls.instances[0].setup_calls), 1)
                    # Model a Torch dependency first imported by the backend:
                    # it must survive scope exit to avoid re-registration.
                    sys.modules[dependency_name] = dependency
                    if raise_after_init:
                        raise RuntimeError("after real constructor")

            if existing_backend:
                self.assertIs(sys.modules[backend_name], prior_backend)
                self.assertIs(getattr(parent, attribute), prior_backend)
            else:
                self.assertNotIn(backend_name, sys.modules)
                self.assertNotIn(attribute, parent.__dict__)
            self.assertIs(sys.modules[dependency_name], dependency)
            for name, previous in saved_mocks.items():
                self.assertIs(sys.modules.get(name, missing), previous)
        finally:
            for name, previous in (
                (backend_name, saved_backend),
                (dependency_name, saved_dependency),
            ):
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
            if saved_attribute is missing:
                parent.__dict__.pop(attribute, None)
            else:
                setattr(parent, attribute, saved_attribute)

    def test_existing_backend_and_parent_binding_are_restored(self):
        self._check_constructor_isolation(
            existing_backend=True, raise_after_init=False
        )

    def test_new_backend_is_removed_after_constructor_scope_raises(self):
        self._check_constructor_isolation(
            existing_backend=False, raise_after_init=True
        )


class TestMooncakeDfsReplicaConfigLoading(unittest.TestCase):
    """三个配置入口都要能读到 dfs_replica_num。"""

    def test_load_from_extra_config(self):
        with _mock_modules(_fake_host_pool_modules()):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStoreConfig,
            )

        config = MooncakeStoreConfig.load_from_extra_config(
            {"master_server_address": "127.0.0.1:50051", "dfs_replica_num": 1}
        )
        self.assertEqual(config.dfs_replica_num, 1)

    def test_load_from_extra_config_defaults_to_zero(self):
        with _mock_modules(_fake_host_pool_modules()):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStoreConfig,
            )

        config = MooncakeStoreConfig.load_from_extra_config(
            {"master_server_address": "127.0.0.1:50051"}
        )
        self.assertEqual(config.dfs_replica_num, 0)


class TestMooncakeDfsOnly(unittest.TestCase):
    def test_dfs_only_initializes_empty_bounded_repair_state(self):
        store, _ = _make_store(
            dfs_replica_num=1,
            dfs_only=True,
            replica_num=0,
            enable_ssd_offload=True,
        )
        self.assertEqual(
            store.get_repair_status(),
            {
                "pending_objects": 0,
                "inflight_objects": 0,
                "tracking_limit": 4096,
                "degraded": False,
                "overflow_events": 0,
                "recovered_objects": 0,
            },
        )

    def test_explicit_dfs_only_uses_zero_memory_and_zero_global_segment(self):
        store, fake_store_cls = _make_store(
            dfs_replica_num=1,
            dfs_only=True,
            replica_num=0,
            enable_ssd_offload=True,
            global_segment_size=4 * 1024 * 1024,
        )
        self.assertTrue(store._dfs_only)
        self.assertEqual(store._effective_replica_num, 0)
        self.assertEqual(store.config.effective_global_segment_size(), 0)

        fake = fake_store_cls.instances[-1]
        self.assertEqual(fake.setup_calls[0][0][2], 0)
        call = fake.put_calls[0]
        config = call[2][0]
        self.assertEqual(config.replica_num, 0)
        self.assertEqual(config.dfs_replica_num, 1)

    def test_dfs_only_uses_separate_namespace(self):
        store, _ = _make_store(
            dfs_replica_num=1,
            dfs_only=True,
            replica_num=0,
            enable_ssd_offload=True,
        )
        self.assertEqual(store._tag_keys(["page-0"]), ["dfs-only_page-0"])

        ordinary, _ = _make_store(dfs_replica_num=1)
        self.assertEqual(ordinary._tag_keys(["page-0"]), ["page-0"])

    def test_dfs_only_rejects_missing_backend_enablement(self):
        with self.assertRaisesRegex(RuntimeError, "enable_ssd_offload"):
            _make_store(
                dfs_replica_num=1,
                dfs_only=True,
                replica_num=0,
                enable_ssd_offload=False,
            )

    def test_dfs_only_rejects_standalone_storage(self):
        with self.assertRaisesRegex(RuntimeError, "standalone_storage"):
            _make_store(
                dfs_replica_num=1,
                dfs_only=True,
                replica_num=0,
                enable_ssd_offload=True,
                standalone_storage=True,
            )

    def test_dfs_only_rejects_old_binding_without_dfs_field(self):
        with self.assertRaisesRegex(RuntimeError, "dfs_replica_num"):
            _make_store(
                dfs_replica_num=1,
                dfs_only=True,
                replica_num=0,
                enable_ssd_offload=True,
                replicate_config_cls=ReplicateConfigWithoutDfs,
            )

    def test_dfs_only_warmup_is_removed(self):
        store, fake_store_cls = _make_store(
            dfs_replica_num=1,
            dfs_only=True,
            replica_num=0,
            enable_ssd_offload=True,
        )
        fake = fake_store_cls.instances[-1]
        self.assertEqual(len(fake.remove_calls), 1)
        warmup_key, force = fake.remove_calls[0]
        self.assertTrue(force)
        self.assertNotIn(warmup_key, fake.objects)

    def test_normal_empty_readiness_still_waits_and_fails_bounded(self):
        with _mock_modules(_fake_host_pool_modules()):
            from sglang.srt.mem_cache.storage.mooncake_store import mooncake_store

        store = object.__new__(mooncake_store.MooncakeStore)
        store._dfs_only = False
        store.config = SimpleNamespace(
            master_server_address="127.0.0.1:50051", master_metrics_port=9003
        )

        class EmptyResponse:
            status_code = 200
            text = ""

        with patch.object(
            mooncake_store.requests, "get", return_value=EmptyResponse()
        ), patch.object(
            mooncake_store.time,
            "perf_counter",
            side_effect=[0.0, 0.0, 1000.0, 1000.0],
        ), patch.object(mooncake_store.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "timeout"):
                store.check_server()


if __name__ == "__main__":
    unittest.main()
