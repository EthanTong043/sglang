"""DFS 副本透传的单元测试。

背景：SGLang 的 KV cache 写入走 batch_put_from。若不显式传
ReplicateConfig，Mooncake C++ 侧用默认构造的配置，其 dfs_replica_num 为 0
（replica.h:107）—— 意味着挂载的并行文件系统（GPFS）**永远不会被写入**。

这一组用例验证 dfs_replica_num 能正确地从配置传到 ReplicateConfig。
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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
        self.dfs_replica_num = 0


def _fake_mooncake_modules(fake_store_cls, replicate_config_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    mooncake_store.ReplicateConfig = replicate_config_cls
    return {"mooncake": mooncake, "mooncake.store": mooncake_store}


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
            self.existing_keys = set()
            self.objects = {}
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        # warmup() 会用 put / is_exist / get 做一次探针读写，
        # 缺少任何一个都会在构造阶段抛 AttributeError
        def put(self, key, value, *args, **kwargs):
            self.objects[key] = value
            return 0

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def get(self, key):
            return self.objects.get(key)

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


def _make_config(*, dfs_replica_num=None, enable_group_semantics=False):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
        "enable_group_semantics": enable_group_semantics,
    }
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


def _make_store(*, dfs_replica_num=None, replicate_config_cls=ReplicateConfigWithDfs,
                enable_group_semantics=False):
    fake_store_cls = _fake_store_class()
    modules = _fake_mooncake_modules(fake_store_cls, replicate_config_cls)
    modules.update(_fake_host_pool_modules())

    with patch.dict("sys.modules", modules):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(
            _make_config(
                dfs_replica_num=dfs_replica_num,
                enable_group_semantics=enable_group_semantics,
            ),
            FakeHostKVCache(),
        )
    return store, fake_store_cls


class TestMooncakeDfsReplicaNum(CustomTestCase):
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


class TestMooncakeDfsReplicaConfigLoading(CustomTestCase):
    """三个配置入口都要能读到 dfs_replica_num。"""

    def test_load_from_extra_config(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.load_from_extra_config(
            {"master_server_address": "127.0.0.1:50051", "dfs_replica_num": 1}
        )
        self.assertEqual(config.dfs_replica_num, 1)

    def test_load_from_extra_config_defaults_to_zero(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.load_from_extra_config(
            {"master_server_address": "127.0.0.1:50051"}
        )
        self.assertEqual(config.dfs_replica_num, 0)


if __name__ == "__main__":
    unittest.main()
