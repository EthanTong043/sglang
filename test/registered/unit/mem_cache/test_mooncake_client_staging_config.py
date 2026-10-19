"""Dependency-free checks of real config loading and native setup arguments.

The production config/base/store AST bodies run against an independent native
double. This checks fail-closed parsing and setup/setup_dummy arguments without
CUDA or native allocation; actual registration/read wave behavior needs a new
service lifecycle. It does not claim native bandwidth or memory safety coverage.
"""

from __future__ import annotations

import json
import os
import runpy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_mooncake_dfs_batch_placement as placement

REPO = Path(__file__).resolve().parents[4]
ENV_NAME = "MOONCAKE_CLIENT_LOCAL_BUFFER_SIZE_BYTES"
DEFAULT_BYTES = 16777216
CANDIDATE_BYTES = 134217728
MISSING = object()

placement.register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SCOPE = placement.scope_for_store()
SCOPE["envs"] = runpy.run_path(str(REPO / "python/sglang/srt/environ.py"))["envs"]
placement.load_nodes(
    placement.STORE, {"_parse_global_segment_size", "_normalize_tenant_id"}, SCOPE
)
CONFIG = SCOPE["MooncakeStoreConfig"]
STORE = SCOPE["MooncakeStore"]


class HostAllocator:
    pass


class Native(placement.Native):
    def __init__(self):
        super().__init__()
        self.normal_calls = []
        self.dummy_calls = []

    def setup(self, *args, **kwargs):
        self.normal_calls.append((args, kwargs))
        return super().setup(*args, **kwargs)

    def setup_dummy(self, *args):
        self.dummy_calls.append(args)
        return self.setup_result


SCOPE["MooncakeHostTensorAllocator"] = HostAllocator


class TestClientStagingConfig(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    @contextmanager
    def configuration(self, source, value=MISSING, *, dummy=False, extra=None):
        data = {
            "master_server_address": "127.0.0.1:50051",
            "client_server_address": "127.0.0.1:50052",
            "protocol": "tcp",
            "check_server": False,
            "standalone_storage": dummy,
        }
        if value is not MISSING:
            data["client_local_buffer_size_bytes"] = value
        rank = placement.rank_config(
            model_name="staging-test", enable_storage_metrics=False, extra_config=extra
        )
        if source == "extra":
            rank.extra_config = data
            yield rank
        elif source == "file":
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mooncake.json"
                path.write_text(json.dumps(data))
                with patch.dict(
                    os.environ, {"SGLANG_HICACHE_MOONCAKE_CONFIG_PATH": str(path)}
                ):
                    yield rank
        elif source == "env":
            environment = {
                "MOONCAKE_MASTER": data["master_server_address"],
                "MOONCAKE_CLIENT": data["client_server_address"],
                "MOONCAKE_PROTOCOL": "tcp",
                "MOONCAKE_CHECK_SERVER": "false",
                "MOONCAKE_STANDALONE_STORAGE": str(dummy),
            }
            if value is not MISSING:
                environment[ENV_NAME] = str(value)
            with patch.dict(os.environ, environment):
                yield rank
        else:
            raise AssertionError(source)

    def construct(self, rank, native=None):
        native = Native() if native is None else native
        # Exercise real dummy sizing with a distinct host allocation, so the
        # staging value cannot accidentally replace required shared-map bytes.
        tensor = SimpleNamespace(
            data_ptr=lambda: 123456, numel=lambda: 3072, element_size=lambda: 2
        )
        host_pool = SimpleNamespace(allocator=HostAllocator(), kv_buffer=tensor)
        with patch.object(
            STORE, "_import_mooncake_store", lambda self: lambda: native
        ), patch.object(
            STORE,
            "_import_mooncake_group_semantics",
            lambda self: (placement.ReplicateConfig, True),
        ):
            store = STORE(rank, host_pool)
        return store, native

    def assert_setup_size(self, native, expected, dummy):
        if dummy:
            self.assertEqual(native.normal_calls, [])
            self.assertEqual(
                native.dummy_calls, [(6144, expected, "127.0.0.1:50052")]
            )
        else:
            self.assertEqual(native.dummy_calls, [])
            self.assertEqual(len(native.normal_calls), 1)
            args, _ = native.normal_calls[0]
            self.assertEqual(args[3], expected)
            self.assertEqual(args[4], "tcp")
            self.assertEqual(args[6], "127.0.0.1:50051")

    def test_default_setup_remains_16_mib_from_all_sources(self):
        for source in ("extra", "file", "env"):
            for dummy in (False, True):
                with self.subTest(source=source, dummy=dummy), self.configuration(
                    source, dummy=dummy
                ) as rank:
                    store, native = self.construct(rank)
                    self.assertEqual(
                        store.config.client_local_buffer_size_bytes, DEFAULT_BYTES
                    )
                    self.assert_setup_size(native, DEFAULT_BYTES, dummy)

    def test_128_mib_reaches_normal_and_dummy_setup_from_all_sources(self):
        for source in ("extra", "file", "env"):
            for dummy in (False, True):
                with self.subTest(source=source, dummy=dummy), self.configuration(
                    source, CANDIDATE_BYTES, dummy=dummy
                ) as rank:
                    store, native = self.construct(rank)
                    self.assertEqual(
                        store.config.client_local_buffer_size_bytes, CANDIDATE_BYTES
                    )
                    self.assert_setup_size(native, CANDIDATE_BYTES, dummy)

    def test_size_is_per_rank_not_divided_by_tp_or_cp(self):
        for rank_id in range(4):
            with self.subTest(rank=rank_id), self.configuration(
                "extra", CANDIDATE_BYTES
            ) as rank:
                rank.tp_rank = rank.attn_cp_rank = rank_id
                _, native = self.construct(rank)
                self.assert_setup_size(native, CANDIDATE_BYTES, False)

    def test_json_inputs_reject_invalid_values_before_native_setup(self):
        invalid = (
            True, False, None, 0, -4096, 1, 4095, 4097,
            1073745920, 134217728.0, 134217728.5,
            "134217728", "128MiB", "", [], {},
        )
        for source in ("extra", "file"):
            for value in invalid:
                with self.subTest(source=source, value=value), self.configuration(
                    source, value
                ) as rank:
                    native = Native()
                    with self.assertRaisesRegex(
                        ValueError, "client_local_buffer_size_bytes"
                    ):
                        self.construct(rank, native)
                    self.assertEqual(native.normal_calls, [])
                    self.assertEqual(native.dummy_calls, [])

    def test_env_inputs_fail_closed_instead_of_defaulting(self):
        invalid = (
            "", "true", "False", "None", "134217728.0", "128MiB",
            "+134217728", " 134217728 ", "１３４２１７７２８",
            "-4096", "0", "4097", "1073745920",
        )
        for value in invalid:
            with self.subTest(value=value), self.configuration("env", value) as rank:
                native = Native()
                with self.assertRaisesRegex(
                    ValueError, "CLIENT_LOCAL_BUFFER_SIZE_BYTES|client_local_buffer_size_bytes"
                ):
                    self.construct(rank, native)
                self.assertEqual(native.normal_calls, [])
                self.assertEqual(native.dummy_calls, [])

    def test_aligned_boundary_sizes_are_not_clamped(self):
        for source in ("extra", "file", "env"):
            for value in (4096, DEFAULT_BYTES, CANDIDATE_BYTES, 1073741824):
                with self.subTest(source=source, value=value), self.configuration(
                    source, value
                ) as rank:
                    config = STORE.__new__(STORE)._load_config(rank)
                    self.assertEqual(config.client_local_buffer_size_bytes, value)

    def test_direct_config_construction_uses_same_validator(self):
        base = CONFIG.load_from_extra_config({"master_server_address": "master:1"})
        fields = vars(base).copy()
        for value in (True, None, "134217728", 134217728.0, 0, 4097, 1073745920):
            fields["client_local_buffer_size_bytes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                CONFIG(**fields)

    def test_size_only_extra_config_overrides_file_or_environment(self):
        for source in ("file", "env"):
            with self.subTest(source=source), self.configuration(
                source,
                DEFAULT_BYTES,
                extra={"client_local_buffer_size_bytes": CANDIDATE_BYTES},
            ) as rank:
                _, native = self.construct(rank)
                self.assert_setup_size(native, CANDIDATE_BYTES, False)

    def test_invalid_size_only_extra_config_is_not_silently_ignored(self):
        for source in ("file", "env"):
            with self.subTest(source=source), self.configuration(
                source, extra={"client_local_buffer_size_bytes": True}
            ) as rank:
                with self.assertRaisesRegex(ValueError, "client_local_buffer_size_bytes"):
                    self.construct(rank)

    def test_full_json_configuration_keeps_existing_environment_precedence(self):
        for source in ("extra", "file"):
            with self.subTest(source=source), patch.dict(
                os.environ, {ENV_NAME: "invalid-unused-environment"}
            ), self.configuration(source) as rank:
                _, native = self.construct(rank)
                self.assert_setup_size(native, DEFAULT_BYTES, False)

    def test_offload_backend_buffer_does_not_change_client_staging(self):
        with patch.dict(
            os.environ, {"MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(CANDIDATE_BYTES)}
        ), self.configuration("env") as rank:
            _, native = self.construct(rank)
            self.assert_setup_size(native, DEFAULT_BYTES, False)

    def test_setup_failure_remains_failure_for_both_entries(self):
        for dummy in (False, True):
            with self.subTest(dummy=dummy), self.configuration(
                "extra", CANDIDATE_BYTES, dummy=dummy
            ) as rank:
                native = Native()
                native.setup_result = -1
                with self.assertRaisesRegex(RuntimeError, "Failed to setup Mooncake"):
                    self.construct(rank, native)
                self.assert_setup_size(native, CANDIDATE_BYTES, dummy)
                self.assertEqual(native.objects, {})

    def test_effective_configuration_is_logged_after_success(self):
        with self.configuration("extra", CANDIDATE_BYTES) as rank, self.assertLogs(
            "v5-tests", level="INFO"
        ) as logs:
            self.construct(rank)
        self.assertTrue(any(
            "client_local_buffer_size_bytes=134217728 standalone_storage=False" in entry
            for entry in logs.output
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
