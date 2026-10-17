import unittest

from sglang.srt.managers.cache_controller import _should_skip_primary_backup
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config(*, tp_rank: int, layer_sharded: bool) -> HiCacheStorageConfig:
    ranges = ((0, 20), (20, 40), (40, 59), (59, 78))
    owned_layer_start, owned_layer_end = ranges[tp_rank]
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=4,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=tp_rank,
        attn_cp_size=4,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="glm53",
        layer_sharded=layer_sharded,
        owned_layer_start=owned_layer_start,
        owned_layer_end=owned_layer_end,
    )


class TestLayerSplitBackupOwnership(unittest.TestCase):
    def test_replicated_mla_keeps_single_primary_writer(self):
        self.assertTrue(
            _should_skip_primary_backup(_config(tp_rank=1, layer_sharded=False))
        )
        self.assertFalse(
            _should_skip_primary_backup(_config(tp_rank=0, layer_sharded=False))
        )

    def test_layer_split_mla_writes_every_cp_owned_slice(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                self.assertFalse(
                    _should_skip_primary_backup(
                        _config(tp_rank=rank, layer_sharded=True)
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
