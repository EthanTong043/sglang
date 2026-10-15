import unittest
from unittest import mock


from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host import mla
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


class _FakeDevicePool:
    def __init__(self, buffers):
        self.kv_buffer = buffers
        self.index_k_with_scale_buffer = buffers


class LayerShardedDirectBackupTests(unittest.TestCase):
    def test_mla_page_first_direct_uses_single_layer_lf_pf_kernel(self):
        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.layout = "page_first_direct"
        host.page_size = 64
        host.kv_buffer = object()
        host._host_layer_index = mock.MagicMock(return_value=3)
        device_pool = _FakeDevicePool([object() for _ in range(8)])
        device_indices = mock.sentinel.device_indices
        host_indices = mock.sentinel.host_indices

        with mock.patch.object(
            mla, "transfer_kv_all_layer_direct_lf_pf"
        ) as transfer:
            host._backup_from_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=5,
                io_backend="direct",
            )

        host._host_layer_index.assert_called_once_with(5)
        transfer.assert_called_once_with(
            src_ptrs=[device_pool.kv_buffer[5]],
            dst_ptrs=[host.kv_buffer],
            src_indices=device_indices,
            dst_indices=host_indices,
            page_size=host.page_size,
        )

    def test_mla_unsupported_layout_fails(self):
        host = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        host.layout = "invalid"
        host.page_size = 64
        host.kv_buffer = object()
        host._host_layer_index = mock.MagicMock(return_value=3)
        device_pool = _FakeDevicePool([object() for _ in range(8)])

        with self.assertRaisesRegex(ValueError, "layer_first or page_first_direct"):
            host._backup_from_device_per_layer(
                device_pool,
                mock.sentinel.host_indices,
                mock.sentinel.device_indices,
                layer_id=5,
                io_backend="direct",
            )

    def test_dsa_page_first_direct_uses_single_layer_lf_pf_kernel(self):
        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.layout = "page_first_direct"
        host.index_k_with_scale_buffer = object()
        host._host_layer_index = mock.MagicMock(return_value=2)
        host._get_indexer_page_indices = mock.MagicMock(
            return_value=(mock.sentinel.host_pages, mock.sentinel.device_pages)
        )
        host.page_size = 1
        device_pool = _FakeDevicePool([object() for _ in range(8)])

        with mock.patch(
                "sglang.srt.mem_cache.memory_pool_host.transfer_kv_all_layer_direct_lf_pf"
            ) as transfer:
            host._backup_from_device_per_layer(
                device_pool,
                mock.sentinel.host_indices,
                mock.sentinel.device_indices,
                layer_id=4,
                io_backend="direct",
            )

        host._host_layer_index.assert_called_once_with(4)
        host._get_indexer_page_indices.assert_called_once_with(
            mock.sentinel.host_indices, mock.sentinel.device_indices
        )
        transfer.assert_called_once_with(
            src_ptrs=[device_pool.index_k_with_scale_buffer[4]],
            dst_ptrs=[host.index_k_with_scale_buffer],
            src_indices=mock.sentinel.device_pages,
            dst_indices=mock.sentinel.host_pages,
            page_size=1,
        )

    def test_dsa_unsupported_layout_fails(self):
        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.layout = "invalid"
        host._host_layer_index = mock.MagicMock(return_value=2)
        host._get_indexer_page_indices = mock.MagicMock(return_value=(object(), object()))
        device_pool = _FakeDevicePool([object() for _ in range(8)])

        with self.assertRaisesRegex(ValueError, "layer_first or page_first_direct"):
            host._backup_from_device_per_layer(
                device_pool,
                mock.sentinel.host_indices,
                mock.sentinel.device_indices,
                layer_id=4,
                io_backend="direct",
            )


if __name__ == "__main__":
    unittest.main()
