import ctypes
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import requests
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache, HostTensorAllocator
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.observability.metrics_collector import StorageMetrics

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min
DEFAULT_TENANT_ID = "default"

logger = logging.getLogger(__name__)


class MooncakeHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        from mooncake.store import MooncakeHostMemAllocator

        self.allocator = MooncakeHostMemAllocator()
        self.ptr = None

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Allocates memory using MooncakeHostMemAllocator and wraps it in a PyTorch tensor.
        """
        self.dims = dims
        self.dtype = dtype
        size = 1
        for d in dims:
            size *= d
        size *= torch.tensor([], dtype=self.dtype).element_size()
        ptr_int = self.allocator.alloc(size)
        self.ptr = ptr_int
        c_type = ctypes.c_byte * size
        c_array = c_type.from_address(ptr_int)

        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size)

        if dtype != torch.uint8:
            element_size = torch.tensor([], dtype=dtype).element_size()
            assert size % element_size == 0, "Size must be divisible by element size"
            tensor = tensor.view(dtype)

        return tensor.view(dims)


def _parse_global_segment_size(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.endswith("gb"):
            num = s[:-2].strip()
            if not num:
                raise ValueError(
                    "Invalid global_segment_size: missing number before 'gb'"
                )
            return int(num) * 1024 * 1024 * 1024
        return int(s)
    return int(value)


def _normalize_tenant_id(value) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    tenant_id = str(value).strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def _parse_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y"):
            return True
        if normalized in ("0", "false", "no", "n"):
            return False
    raise ValueError(f"{name} must be a boolean")


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    protocol: str
    device_name: str
    master_server_address: str
    master_metrics_port: int
    check_server: bool
    standalone_storage: bool
    client_server_address: str
    enable_ssd_offload: bool = False
    ssd_offload_path: Optional[str] = None
    tenant_id: str = DEFAULT_TENANT_ID
    # Number of MEMORY replicas. The default preserves historical behavior;
    # zero is valid only when dfs_only is explicitly enabled.
    replica_num: int = 1
    # Explicitly select the no-distributed-L3 topology. Do not infer this from
    # dfs_replica_num, since ordinary MEMORY+DFS remains supported.
    dfs_only: bool = False
    # 每个对象额外写入的 DFS（分布式文件系统）副本数。
    #
    # 0 = 不写 DFS，与改动前行为一致，也是 Mooncake C++ 侧
    # ReplicateConfig 的默认值（replica.h:107）。
    #
    # 设为 1 时 KV 会同时写入 Mooncake 的 DRAM 段和挂载的并行文件系统
    # （如 GPFS），后者作为跨实例共享的容量层。**不设则 DFS 永远不会
    # 被写入** —— 因为不传 ReplicateConfig 时 C++ 侧用的是默认值 0。
    #
    # Mooncake 目前只支持 0 或 1（master_service.cpp 对 >1 返回
    # INVALID_PARAMS）；DFS-only 需要显式 dfs_only=true。
    dfs_replica_num: int = 0

    def effective_replica_num(self) -> int:
        """Return the explicit MEMORY replica count for writes.

        ``dfs_replica_num`` alone never changes this value. A zero-MEMORY
        topology must opt in with ``dfs_only`` so an old config cannot silently
        lose its DRAM replica.
        """
        if self.dfs_only:
            if self.replica_num not in (0, 1):
                raise ValueError(
                    "dfs_only requires replica_num to be 0 or the default 1"
                )
            return 0
        if self.replica_num < 1:
            raise ValueError("replica_num=0 requires explicit dfs_only=true")
        return self.replica_num

    def effective_global_segment_size(self) -> int:
        """Return the global MEMORY segment capacity for this topology."""
        return 0 if self.dfs_only else self.global_segment_size

    @staticmethod
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        if not envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            raise RuntimeError(
                f"Config file path not set. Please set {envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.name}"
            )
        file_path = envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get()
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if (
            "master_server_address" not in config
            and "client_server_address" not in config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in config file"
            )

        return MooncakeStoreConfig(
            local_hostname=config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=config.get("check_server", envs.MOONCAKE_CHECK_SERVER.default),
            standalone_storage=config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
            replica_num=int(
                config.get("replica_num", envs.MOONCAKE_REPLICA_NUM.default)
            ),
            dfs_only=_parse_bool(
                config.get("dfs_only", envs.MOONCAKE_DFS_ONLY.default),
                "dfs_only",
            ),
            dfs_replica_num=int(
                config.get("dfs_replica_num", envs.MOONCAKE_DFS_REPLICA_NUM.default)
            ),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE=""
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...
        if not envs.MOONCAKE_MASTER.is_set() and not envs.MOONCAKE_CLIENT.is_set():
            raise ValueError(
                "Either the environment variable 'MOONCAKE_MASTER' or 'MOONCAKE_CLIENT' is not set."
            )

        # Special handling for local_hostname: try MOONCAKE_LOCAL_HOSTNAME first,
        # then fall back to LOCAL_HOSTNAME if not set.
        # This is for forward compatibility with the legacy LOCAL_HOSTNAME environment variable.
        if envs.MOONCAKE_LOCAL_HOSTNAME.is_set():
            local_hostname = envs.MOONCAKE_LOCAL_HOSTNAME.get()
        else:
            local_hostname = os.getenv(
                "LOCAL_HOSTNAME", envs.MOONCAKE_LOCAL_HOSTNAME.default
            )

        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=envs.MOONCAKE_TE_META_DATA_SERVER.get(),
            global_segment_size=_parse_global_segment_size(
                envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.get()
            ),
            protocol=envs.MOONCAKE_PROTOCOL.get(),
            device_name=envs.MOONCAKE_DEVICE.get(),
            master_server_address=envs.MOONCAKE_MASTER.get(),
            master_metrics_port=envs.MOONCAKE_MASTER_METRICS_PORT.get(),
            check_server=envs.MOONCAKE_CHECK_SERVER.get(),
            standalone_storage=envs.MOONCAKE_STANDALONE_STORAGE.get(),
            client_server_address=envs.MOONCAKE_CLIENT.get(),
            enable_ssd_offload=envs.MOONCAKE_ENABLE_SSD_OFFLOAD.get(),
            ssd_offload_path=envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.get(),
            tenant_id=_normalize_tenant_id(envs.MOONCAKE_TENANT_ID.get()),
            replica_num=envs.MOONCAKE_REPLICA_NUM.get(),
            dfs_only=envs.MOONCAKE_DFS_ONLY.get(),
            dfs_replica_num=envs.MOONCAKE_DFS_REPLICA_NUM.get(),
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "MooncakeStoreConfig":
        """Load config from extra_config dictionary."""
        if (
            "master_server_address" not in extra_config
            and "client_server_address" not in extra_config
        ):
            raise ValueError(
                "Either master_server_address or client_server_address is required in extra_config"
            )

        return MooncakeStoreConfig(
            local_hostname=extra_config.get(
                "local_hostname", envs.MOONCAKE_LOCAL_HOSTNAME.default
            ),
            metadata_server=extra_config.get(
                "metadata_server", envs.MOONCAKE_TE_META_DATA_SERVER.default
            ),
            global_segment_size=_parse_global_segment_size(
                extra_config.get(
                    "global_segment_size", envs.MOONCAKE_GLOBAL_SEGMENT_SIZE.default
                )
            ),
            protocol=extra_config.get("protocol", envs.MOONCAKE_PROTOCOL.default),
            device_name=extra_config.get("device_name", envs.MOONCAKE_DEVICE.default),
            master_server_address=extra_config.get(
                "master_server_address", envs.MOONCAKE_MASTER.default
            ),
            master_metrics_port=extra_config.get(
                "master_metrics_port", envs.MOONCAKE_MASTER_METRICS_PORT.default
            ),
            check_server=extra_config.get(
                "check_server", envs.MOONCAKE_CHECK_SERVER.default
            ),
            standalone_storage=extra_config.get(
                "standalone_storage", envs.MOONCAKE_STANDALONE_STORAGE.default
            ),
            client_server_address=extra_config.get(
                "client_server_address", envs.MOONCAKE_CLIENT.default
            ),
            enable_ssd_offload=extra_config.get(
                "enable_ssd_offload", envs.MOONCAKE_ENABLE_SSD_OFFLOAD.default
            ),
            ssd_offload_path=extra_config.get(
                "ssd_offload_path", envs.MOONCAKE_OFFLOAD_FILE_STORAGE_PATH.default
            ),
            tenant_id=_normalize_tenant_id(
                extra_config.get("tenant_id", envs.MOONCAKE_TENANT_ID.default)
            ),
            replica_num=int(
                extra_config.get(
                    "replica_num", envs.MOONCAKE_REPLICA_NUM.default
                )
            ),
            dfs_only=_parse_bool(
                extra_config.get("dfs_only", envs.MOONCAKE_DFS_ONLY.default),
                "dfs_only",
            ),
            dfs_replica_num=int(
                extra_config.get(
                    "dfs_replica_num", envs.MOONCAKE_DFS_REPLICA_NUM.default
                )
            ),
        )


class MooncakeBaseStore:
    def __init__(self):
        self.store = None
        self.config = None

    def _import_mooncake_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore

            return MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeConnector."
            ) from e

    def _import_mooncake_group_semantics(self):
        try:
            from mooncake.store import ReplicateConfig
        except ImportError:
            return None, False

        supports_group_ids = hasattr(ReplicateConfig, "group_ids")
        if not supports_group_ids:
            try:
                supports_group_ids = hasattr(ReplicateConfig(), "group_ids")
            except Exception:
                supports_group_ids = False
        return ReplicateConfig, supports_group_ids

    @staticmethod
    def _supports_replicate_config_field(replicate_config_cls, field: str) -> bool:
        """Detect whether the binding exposes a ReplicateConfig field.

        与 group_ids 用同一套判断方式：先看类属性，pybind 的
        def_readwrite 在某些版本上只在实例上可见，所以再退一步试实例。
        """
        if replicate_config_cls is None:
            return False
        if hasattr(replicate_config_cls, field):
            return True
        try:
            return hasattr(replicate_config_cls(), field)
        except Exception:
            return False

    @staticmethod
    def _supports_dfs_replica_num(replicate_config_cls) -> bool:
        return MooncakeStore._supports_replicate_config_field(
            replicate_config_cls, "dfs_replica_num"
        )

    def _needs_explicit_replicate_config(self) -> bool:
        return (
            self._dfs_only
            or self._use_dfs_replica
            or self.config.replica_num != 1
            or self._use_group_semantics
        )

    def _make_replicate_config(self):
        """Build the write config without inferring DFS-only from DFS count."""
        if self._replicate_config_cls is None:
            raise RuntimeError(
                "Mooncake binding does not expose ReplicateConfig; "
                "cannot configure the requested write topology"
            )
        if not self._supports_replicate_config_field(
            self._replicate_config_cls, "replica_num"
        ):
            raise RuntimeError(
                "Mooncake binding does not expose ReplicateConfig.replica_num"
            )
        config = self._replicate_config_cls()
        config.replica_num = self._effective_replica_num
        if self._use_dfs_replica:
            config.dfs_replica_num = self.config.dfs_replica_num
        return config

    def _load_config(self, storage_config: Any = None):
        extra_config = (
            getattr(storage_config, "extra_config", None) if storage_config else None
        )

        if extra_config and (
            extra_config.get("master_server_address") is not None
            or extra_config.get("client_server_address") is not None
        ):
            config = MooncakeStoreConfig.load_from_extra_config(extra_config)
            logger.info("Mooncake Configuration loaded from extra_config successfully.")

        elif envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set():
            config = MooncakeStoreConfig.from_file()
            logger.info("Mooncake Configuration loaded from file successfully.")

        else:
            config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded from env successfully.")

        return config

    def register_buffer(self, tensor: torch.Tensor):
        if self.store is None:
            raise RuntimeError("Mooncake store is not initialized.")
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        if size == 0:
            # conv-only models have a 0-element ssm state; nothing to register
            return
        ret_code = self.store.register_buffer(ptr, size)
        if ret_code != 0:
            logger.error(f"Failed to register buffer, error code: {ret_code}")
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )


class _DfsRepairState:
    """Bounded metadata only; never owns a tensor, pointer, or native call."""

    LIMIT = 4096

    def __init__(self):
        self.lock = threading.Lock()
        self.markers = {}  # physical key -> (expected total size, generation)
        # physical key -> (claim id, marker generation or None, failure epoch)
        self.inflight = {}
        self.claim_id = 0
        # Keys successfully rewritten during degraded recovery can return to
        # the normal idempotent path. This bounded LRU prevents permanent
        # write amplification while keeping recovery metadata finite.
        self.recovered = OrderedDict()
        self.generation = 0
        self.degraded = False
        self.overflow_events = 0

    def record(self, failures):
        with self.lock:
            first_overflow = False
            for key, expected_size in failures:
                self.generation += 1
                self.recovered.pop(key, None)
                if key in self.markers or len(self.markers) < self.LIMIT:
                    self.markers[key] = (expected_size, self.generation)
                else:
                    first_overflow |= not self.degraded
                    self.degraded = True
                    self.overflow_events += 1
            return first_overflow

    def classify(self, keys, sizes, exists):
        """Classify writes while keeping degraded mode fail-closed.

        Lost failure identities cannot be recovered from an existence query:
        even a miss can race with an idempotent Put of an existing bad object.
        Use a claimed Upsert for untrusted keys in degraded mode, including
        misses. Never hold the metadata lock across native I/O.
        """
        existing, ordinary, repairs = [], [], []
        with self.lock:
            for index, (key, size, present) in enumerate(zip(keys, sizes, exists)):
                if present not in (0, 1) or key in self.inflight:
                    continue
                marker = self.markers.get(key)
                if marker is not None:
                    expected_size, marker_generation = marker
                    if size != expected_size:
                        continue
                elif self.degraded:
                    if present == 1 and key in self.recovered:
                        self.recovered.move_to_end(key)
                        existing.append(index)
                        continue
                    marker_generation = None
                else:
                    (existing if present == 1 else ordinary).append(index)
                    continue
                if len(self.inflight) >= self.LIMIT:
                    continue
                self.claim_id += 1
                self.inflight[key] = (
                    self.claim_id, marker_generation, self.generation
                )
                repairs.append((index, key, self.claim_id))
            # Bind the acknowledgement epoch to this classification, not to
            # a later snapshot that could hide a failure between the two.
            return existing, ordinary, repairs, self.generation

    def finish(self, key, claim_id, success):
        with self.lock:
            claim = self.inflight.get(key)
            if claim is None or claim[0] != claim_id:
                return False
            del self.inflight[key]
            _, marker_generation, failure_epoch = claim
            if not success:
                return False
            current = self.markers.get(key)
            if marker_generation is not None:
                if current is None or current[1] != marker_generation:
                    return False
                del self.markers[key]
            elif current is not None or failure_epoch != self.generation:
                # The full marker table may have dropped a newer failure.
                return False
            if self.degraded:
                self._remember_recovered_locked(key)
            return True

    def can_ack_existing(self, key, failure_epoch):
        """An existence/no-op Put acknowledgement cannot recover lost data."""
        with self.lock:
            return (
                failure_epoch == self.generation
                and key not in self.markers
                and key not in self.inflight
                and (not self.degraded or key in self.recovered)
            )

    def _remember_recovered_locked(self, key):
        self.recovered[key] = None
        self.recovered.move_to_end(key)
        while len(self.recovered) > self.LIMIT:
            self.recovered.popitem(last=False)

    def status(self):
        with self.lock:
            return {
                "pending_objects": len(self.markers),
                "inflight_objects": len(self.inflight),
                "tracking_limit": self.LIMIT,
                "degraded": self.degraded,
                "overflow_events": self.overflow_events,
                "recovered_objects": len(self.recovered),
            }


class MooncakeStore(HiCacheStorage, MooncakeBaseStore):

    # A DFS-only duplicate write is idempotent only after a readable COMPLETE
    # replica is visible. Keep this gate bounded; a PROCESSING key must not be
    # converted into a successful cache write indefinitely.
    _DFS_DUPLICATE_READY_TIMEOUT_S = 5.0
    _DFS_DUPLICATE_READY_BACKOFF_S = (0.01, 0.02, 0.05, 0.1, 0.2)

    @staticmethod
    def _standalone_required_bytes(mem_pool: Any) -> int:
        """Compute total bytes of host buffers that must be visible to the real client.

        In standalone (dummy client) mode, the real mooncake_client process needs
        to map any host buffers we will later pass by pointer via register_buffer().
        For hybrid models, that includes KV + sidecar pools (e.g. Mamba temporal/conv).
        """
        # Prefer a generic "hybrid pool" accessor when present.
        total = 0
        seen_ptrs: set[int] = set()

        def _add_tensor(t: Optional[torch.Tensor]):
            nonlocal total
            if t is None:
                return
            try:
                ptr = int(t.data_ptr())
            except Exception:
                return
            if ptr in seen_ptrs:
                return
            seen_ptrs.add(ptr)
            total += int(t.numel() * t.element_size())

        # Always include the anchor KV buffer if present.
        _add_tensor(getattr(mem_pool, "kv_buffer", None))

        # HostPoolGroup: include each pool's hybrid buffers when available.
        entries = getattr(mem_pool, "entries", None)
        if entries:
            for entry in entries:
                host_pool = getattr(entry, "host_pool", None)
                if host_pool is None:
                    continue
                # KV pool anchor memory is already covered, but harmless if added twice.
                _add_tensor(getattr(host_pool, "kv_buffer", None))
                for buf in getattr(host_pool, "get_hybrid_pool_buffer", lambda: [])():
                    _add_tensor(buf)
            return total

        # Single HostKVCache-like pool: add its sidecar buffers if any.
        for buf in getattr(mem_pool, "get_hybrid_pool_buffer", lambda: [])():
            _add_tensor(buf)
        return total

    def __init__(
        self, storage_config: HiCacheStorageConfig = None, mem_pool: HostKVCache = None
    ):
        MooncakeBaseStore.__init__(self)
        MooncakeDistributedStore = self._import_mooncake_store()
        self._replicate_config_cls, self._supports_group_ids = (
            self._import_mooncake_group_semantics()
        )
        # 默认关闭。真正的判定在下面的 try 块里（需要先加载配置），
        # 这里先给出确定值，避免构造中途抛异常时留下未定义属性。
        self._use_dfs_replica = False
        self._dfs_repair_state = _DfsRepairState()
        try:
            self.store = MooncakeDistributedStore()

            self.config = self._load_config(storage_config)
            self._dfs_only = bool(self.config.dfs_only)
            self._effective_replica_num = self.config.effective_replica_num()
            if self._dfs_only:
                if self.config.dfs_replica_num != 1:
                    raise RuntimeError(
                        "dfs_only=true requires dfs_replica_num=1"
                    )
                if self.config.standalone_storage:
                    raise RuntimeError(
                        "dfs_only=true is not supported with "
                        "standalone_storage=true"
                    )
                if not self.config.enable_ssd_offload:
                    raise RuntimeError(
                        "dfs_only=true requires enable_ssd_offload=true so "
                        "the DFS backend is initialized"
                    )
            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            self.enable_group_semantics = bool(
                extra_config.get("enable_group_semantics", False)
                if extra_config
                else False
            )
            self._use_group_semantics = (
                self.enable_group_semantics
                and self._supports_group_ids
                and self._replicate_config_cls is not None
            )
            if self.enable_group_semantics and not self._supports_group_ids:
                logger.warning(
                    "Mooncake group semantics is enabled, but the installed "
                    "Mooncake package does not support ReplicateConfig.group_ids. "
                    "Falling back to the existing batch_put_from path."
                )
            if (
                (
                    self.config.replica_num != 1
                    or self.config.dfs_replica_num > 0
                    or self._dfs_only
                )
                and not self._supports_replicate_config_field(
                    self._replicate_config_cls, "replica_num"
                )
            ):
                raise RuntimeError(
                    "The installed Mooncake binding does not expose "
                    "ReplicateConfig.replica_num"
                )
            if self._dfs_only and not self._supports_replicate_config_field(
                self._replicate_config_cls, "dfs_replica_num"
            ):
                raise RuntimeError(
                    "dfs_only=true requires a Mooncake binding exposing "
                    "ReplicateConfig.dfs_replica_num"
                )

            # DFS 副本：配置要求写、且安装的 Mooncake 支持该字段时才启用。
            #
            # 与 group semantics 同样做能力探测 —— dfs_replica_num 是较新的
            # pybind 绑定（store_py.cpp:2003），旧版本的 wheel 上不存在。
            # 盲目 setattr 在旧版本上要么静默无效（pybind 对象通常禁止
            # 动态属性，但不同版本行为不一），要么抛 AttributeError，
            # 两种都不如显式探测 + 告警。
            self._use_dfs_replica = False
            if self.config.dfs_replica_num > 0:
                if self._supports_dfs_replica_num(self._replicate_config_cls):
                    self._use_dfs_replica = True
                    logger.info(
                        "Mooncake DFS replica enabled, dfs_replica_num=%d; "
                        "KV pages will also be written to the mounted "
                        "distributed filesystem",
                        self.config.dfs_replica_num,
                    )
                else:
                    if self._dfs_only:
                        raise RuntimeError(
                            "dfs_only=true requires a Mooncake binding exposing "
                            "ReplicateConfig.dfs_replica_num"
                        )
                    # 不静默降级：用户显式要求写 DFS 却写不了，属于配置
                    # 与运行环境不匹配。继续跑（只是没有 L4），但必须让
                    # 人看到 —— 否则表现为"GPFS 目录一直是空的"而无从查起。
                    logger.warning(
                        "dfs_replica_num=%d was requested, but the installed "
                        "Mooncake package does not expose "
                        "ReplicateConfig.dfs_replica_num. KV will NOT be "
                        "written to the distributed filesystem. Upgrade "
                        "Mooncake to use the DFS capacity tier.",
                        self.config.dfs_replica_num,
                    )
            rank_scale_factor = (
                1
                if storage_config is None
                else (storage_config.tp_size * storage_config.pp_size)
            )

            per_rank_global_segment_size = (
                self.config.effective_global_segment_size() // rank_scale_factor
            )

            # Use the backend tag and model name as a prefix to isolate tenants
            # and models sharing one store.
            self.config_prefix = None
            config_prefix_parts = []
            if self._dfs_only:
                # Never probe or overwrite keys written by the historical
                # MEMORY+DFS/ MEMORY-only topology in the same namespace.
                config_prefix_parts.append("dfs-only")
            if extra_config and extra_config.get("extra_backend_tag") is not None:
                config_prefix_parts.append(str(extra_config["extra_backend_tag"]))
            if storage_config is not None and storage_config.model_name:
                model_name = "-".join(storage_config.model_name.split("/"))
                config_prefix_parts.append(model_name)
            if config_prefix_parts:
                self.config_prefix = "_".join(config_prefix_parts)
                logger.info(f"Using Mooncake config prefix: {self.config_prefix}")

            # Check server status
            if self.config.check_server:
                self.check_server()

            # Handle JSON device_name configuration
            device_name = self.config.device_name
            if device_name and device_name.strip().startswith("{"):
                try:
                    device_config = json.loads(device_name)
                    if storage_config and hasattr(storage_config, "tp_rank"):
                        tp_rank = storage_config.tp_rank
                        # Try both integer and string keys since JSON parsing may convert keys
                        device_name = device_config.get(tp_rank, "")
                        if not device_name:
                            device_name = device_config.get(str(tp_rank), "")
                    else:
                        device_name = ""
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"Failed to parse device_name as JSON: {device_name}"
                    )
                    device_name = ""
            if self.config.standalone_storage:
                if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
                    raise RuntimeError(
                        "MooncakeStore with standalone_storage=True requires MooncakeHostTensorAllocator. "
                        "Please set standalone_storage=False "
                        "or upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'."
                    )
                required_bytes = self._standalone_required_bytes(mem_pool)
                ret_code = self.store.setup_dummy(
                    required_bytes,
                    DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                    self.config.client_server_address,
                )
            else:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_mooncake_transfer_engine,
                    )

                    self._shared_mooncake_transfer_engine = (
                        get_mooncake_transfer_engine()
                    )
                except Exception:
                    self._shared_mooncake_transfer_engine = None
                    logger.debug("Failed to reuse initialized mooncake transfer engine")

                # Only reuse the shared MooncakeTransferEngine when its
                # configuration matches the one used by MooncakeStore.
                if (
                    self._shared_mooncake_transfer_engine is not None
                    and device_name
                    == self._shared_mooncake_transfer_engine.get_ib_device()
                    and self.config.metadata_server == "P2PHANDSHAKE"
                    and self.config.protocol == "rdma"
                ):
                    client_hostname = (
                        self._shared_mooncake_transfer_engine.get_session_id()
                    )
                    transfer_engine = self._shared_mooncake_transfer_engine.get_engine()
                    logger.info(
                        f"Reuse initialized mooncake transfer engine: {self._shared_mooncake_transfer_engine}"
                    )
                else:
                    client_hostname = self.config.local_hostname
                    transfer_engine = None

                setup_kwargs = {}
                if self.config.enable_ssd_offload:
                    setup_kwargs["enable_ssd_offload"] = True
                if self.config.ssd_offload_path is not None:
                    setup_kwargs["ssd_offload_path"] = self.config.ssd_offload_path
                if self.config.tenant_id != DEFAULT_TENANT_ID:
                    setup_kwargs["tenant_id"] = self.config.tenant_id

                while True:
                    try:
                        ret_code = self.store.setup(
                            client_hostname,
                            self.config.metadata_server,
                            per_rank_global_segment_size,
                            DEFAULT_LOCAL_BUFFER_SIZE,  # Zero copy interface does not need local buffer
                            self.config.protocol,
                            device_name,
                            self.config.master_server_address,
                            transfer_engine,
                            **setup_kwargs,
                        )
                        break
                    except TypeError as e:
                        unsupported_kwargs = [
                            key for key in list(setup_kwargs) if key in str(e)
                        ]
                        if not unsupported_kwargs:
                            raise
                        if self._dfs_only and any(
                            key in ("enable_ssd_offload", "ssd_offload_path")
                            for key in unsupported_kwargs
                        ):
                            raise RuntimeError(
                                "dfs_only=true requires the installed Mooncake "
                                "binding to support DFS backend setup"
                            ) from e
                        if "tenant_id" in unsupported_kwargs:
                            raise RuntimeError(
                                "The installed Mooncake version does not support "
                                "tenant_id in MooncakeDistributedStore.setup(). "
                                "Please upgrade Mooncake to use non-default "
                                "Mooncake tenants with SGLang."
                            ) from e
                        logger.warning(
                            "The installed Mooncake version does not support the "
                            f"{', '.join(unsupported_kwargs)} parameter(s) in setup(). "
                            f"Retrying without {', '.join(unsupported_kwargs)}. "
                            "Please upgrade Mooncake to enable SSD offload support."
                        )
                        for key in unsupported_kwargs:
                            setup_kwargs.pop(key, None)
            if ret_code:
                raise RuntimeError(
                    f"Failed to setup Mooncake store, error code: {ret_code}"
                )
            logger.info("Mooncake store setup successfully.")

            self.local_rank = (
                storage_config.tp_rank if storage_config is not None else 0
            )
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

            self.enable_storage_metrics = False
            if storage_config is not None:
                self.is_mla_backend = storage_config.is_mla_model
                self.pp_rank = storage_config.pp_rank
                self.pp_size = storage_config.pp_size
                self.attn_cp_rank = storage_config.attn_cp_rank
                self.attn_cp_size = storage_config.attn_cp_size
                self.layer_sharded = bool(
                    getattr(storage_config, "layer_sharded", False)
                )
                self.owned_layer_start = getattr(
                    storage_config, "owned_layer_start", None
                )
                self.owned_layer_end = getattr(
                    storage_config, "owned_layer_end", None
                )
                self.enable_storage_metrics = storage_config.enable_storage_metrics
            else:
                self.is_mla_backend = False
                self.local_rank = 0
                self.pp_rank = 0
                self.pp_size = 1
                self.attn_cp_rank = 0
                self.attn_cp_size = 1
                self.layer_sharded = False
                self.owned_layer_start = None
                self.owned_layer_end = None

            self.enable_pp = self.pp_size > 1
            if self.enable_pp:
                self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
                self.mla_suffix = f"{self.pp_rank}"
            else:
                self.mha_suffix = f"{self.local_rank}"
                self.mla_suffix = ""

            self.storage_config = storage_config
            self.should_split_heads = storage_config.should_split_heads
            self.split_factor = 0
            if self.should_split_heads:
                self.split_factor = (
                    self.storage_config.tp_lcm_size // self.storage_config.tp_size
                )
                base_rank = self.local_rank * self.split_factor
                target_ranks = [base_rank + i for i in range(self.split_factor)]
                if self.enable_pp:
                    self.mha_suffix = [
                        f"{rank}_{self.pp_rank}" for rank in target_ranks
                    ]
                else:
                    self.mha_suffix = [f"{rank}" for rank in target_ranks]

            self.registered_pools = {}

            self.gb_per_page = None
            self.prefetch_pgs = []
            self.backup_pgs = []
            self.prefetch_bandwidth = []
            self.backup_bandwidth = []

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    @staticmethod
    def _iter_host_pool_buffers(host_pool: HostKVCache):
        get_buffers = getattr(
            host_pool,
            "get_hybrid_pool_buffer",
            lambda: [getattr(host_pool, "kv_buffer", None)],
        )
        for buf in get_buffers():
            if buf is not None:
                yield buf

    def check_server(self):
        master_server_ip = self.config.master_server_address.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self.config.master_metrics_port}/get_all_segments"
        start_time = time.perf_counter()

        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                logger.info(
                    "waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            # An empty segment list is valid in DFS-only mode: readiness means
            # the admin endpoint is serving, not that a MEMORY segment exists.
            if (
                getattr(check_segments_resp, "status_code", 200) != 200
                or (
                    not getattr(self, "_dfs_only", False)
                    and getattr(check_segments_resp, "text", "") == ""
                )
            ):
                logger.info(
                    "waiting mooncake store admin endpoint, status=%s, "
                    "cost_time: %.2f seconds.",
                    getattr(check_segments_resp, "status_code", None),
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            logger.info("Mooncake store server started successfully.")
            check_result = True
            break

        if not check_result:
            logger.error("Launch mooncake store server timeout")
            raise ValueError("Launch mooncake store server timeout")

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB
        warmup_config = (
            self._make_replicate_config()
            if self._dfs_only or self.config.replica_num != 1
            else None
        )

        # Retry logic to handle Transfer Engine startup race condition
        max_retries = 10
        retry_delay = 1.0  # seconds
        published = False
        try:
            for attempt in range(max_retries):
                if warmup_config is None:
                    ret = self.store.put(warmup_key, warmup_value)
                else:
                    ret = self.store.put(warmup_key, warmup_value, warmup_config)
                if ret == 0:
                    published = True
                    break
                logger.warning(
                    f"[TP{getattr(self, 'local_rank', 0)}] Warmup put failed "
                    f"(attempt {attempt + 1}/{max_retries}), "
                    f"ret={ret}, retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"[TP{getattr(self, 'local_rank', 0)}] Warmup put failed after "
                    f"{max_retries} attempts, "
                    "Transfer Engine might not be ready"
                )

            assert self.store.is_exist(warmup_key) == 1
            assert self.store.get(warmup_key) == warmup_value
        finally:
            if published:
                remove_result = self.store.remove(warmup_key, True)
                if remove_result != 0:
                    raise RuntimeError(
                        f"Failed to remove Mooncake warmup key: {remove_result}"
                    )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        if getattr(self.mem_pool_host, "kv_buffer", None) is None:
            # Hybrid logical anchors only own allocation indices. Their physical
            # tensors are registered through register_mem_host_pool_v2().
            return
        try:
            for buffer in self._iter_host_pool_buffers(self.mem_pool_host):
                super().register_buffer(buffer)
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

        bytes_per_page = mem_pool_host.get_ksize_per_token() * mem_pool_host.page_size
        self.gb_per_page = bytes_per_page / (1 << 30)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        # KV anchor memory is already registered via register_mem_pool_host().
        # v2 here only registers additional hybrid pools.
        if host_pool_name == PoolName.KV:
            return
        if host_pool_name == PoolName.DRAFT:
            self.registered_pools[host_pool_name] = host_pool
            super().register_buffer(host_pool.kv_buffer)
            return

        # Keep a name->pool mapping so batch v2 can resolve PoolTransfer.name to
        # the corresponding host pool implementation at runtime.
        self.registered_pools[host_pool_name] = host_pool

        # Non-anchor pools are either sidecar-specific pools with their own
        # accessor, or ordinary KV-like host pools used as SWA side pools.
        for buf in self._iter_host_pool_buffers(host_pool):
            super().register_buffer(buf)

    def _tag_keys(self, keys: List[str]) -> List[str]:
        if self.config_prefix is None:
            return keys
        return [f"{self.config_prefix}_{key}" for key in keys]

    def _mla_ownership_suffix(self, owner_pool: Optional[Any] = None) -> str:
        """Return the physical ownership portion of an MLA key.

        Replicated MLA keeps its historical suffix for cache compatibility.
        Layer-split MLA must identify both the CP slice and PP partition; CP4
        therefore produces ``cp{rank}_{size}_pp{pp_rank}`` even when PP=1.
        """
        layer_sharded = getattr(self, "layer_sharded", False)
        attn_cp_rank = getattr(self, "attn_cp_rank", 0)
        attn_cp_size = getattr(self, "attn_cp_size", 1)
        if owner_pool is not None:
            pool_device = getattr(owner_pool, "device_pool", None)
            if pool_device is not None:
                layer_sharded = bool(
                    getattr(pool_device, "layer_shard_enabled", False)
                )
                attn_cp_rank = getattr(
                    pool_device, "layer_shard_rank", attn_cp_rank
                )
                attn_cp_size = getattr(
                    pool_device, "layer_shard_size", attn_cp_size
                )
        if layer_sharded:
            if attn_cp_size <= 1:
                raise RuntimeError(
                    "layer_sharded MLA storage requires attn_cp_size > 1"
                )
            return (
                f"cp{attn_cp_rank}_{attn_cp_size}_pp"
                f"{getattr(self, 'pp_rank', 0)}"
            )
        return getattr(self, "mla_suffix", "")

    def _make_mla_physical_key(
        self,
        logical_key: str,
        component: str,
        owner_pool: Optional[Any] = None,
    ) -> str:
        """Build one MLA physical object key from a tagged logical key."""
        return (
            f"{logical_key}_{self._mla_ownership_suffix(owner_pool)}_{component}"
        )

    def _can_use_group_semantics(self) -> bool:
        return self._use_group_semantics

    def _make_group_id(self, logical_key: str) -> str:
        return f"sglang-hicache:{logical_key}"

    def _expand_group_ids(
        self, logical_keys: List[str], key_multiplier: int
    ) -> List[str]:
        group_ids = []
        for key in logical_keys:
            group_ids.extend([self._make_group_id(key)] * key_multiplier)
        return group_ids

    def _filter_group_ids(
        self, group_ids: Optional[List[str]], indices: List[int]
    ) -> Optional[List[str]]:
        if group_ids is None:
            return None
        return [group_ids[i] for i in indices]

    def _get_hybrid_page_component_keys(
        self, page_keys: List[str], transfer: PoolTransfer
    ) -> Tuple[List[str], int]:
        host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
        if host_pool is None:
            raise ValueError(f"Unregistered Mooncake hybrid pool: {transfer.name}")

        # Suffix order must match get_page_buffer_meta() for one page, because
        # Mooncake zips object keys with registered buffer pointers.
        pool_name = transfer.name

        def mla_component_keys(component: str) -> Tuple[List[str], int]:
            return (
                [
                    self._make_mla_physical_key(page_key, component, host_pool)
                    for page_key in page_keys
                ],
                1,
            )

        suffixes = []
        if pool_name == PoolName.KV:
            return mla_component_keys("k")
        elif pool_name == PoolName.MAMBA:
            # Mamba stores one temporal object plus one object per conv state.
            # conv-only models have no ssm state; drop the 0-element temporal
            # object (mooncake rejects 0-size puts). get_page_buffer_meta drops
            # its temporal pointer under the same condition to stay aligned.
            conv_num = len(getattr(host_pool, "conv_buffer", None) or [])
            suffixes = [f"_{self.mha_suffix}_conv_{i}" for i in range(conv_num)]
            if getattr(host_pool, "temporal_state_elem_size", 1) > 0:
                suffixes = [f"_{self.mha_suffix}_temporal"] + suffixes
        elif pool_name == PoolName.DRAFT:
            # Draft pool's MLA/MHA layout is independent from the target
            # (e.g. EAGLE-MHA draft on top of an MLA target), so pick the
            # suffix scheme from the draft pool's own class. The `_draft`
            # tag is what keeps these keys from colliding with target's
            # `{rank}_k` / `{rank}_k` + `{rank}_v` keys.
            draft_pool = self.registered_pools.get(PoolName.DRAFT)
            if isinstance(draft_pool, MLATokenToKVPoolHost):
                return mla_component_keys(f"{PoolName.DRAFT}_k")
            else:
                suffixes = [
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_k",
                    f"_{self.mha_suffix}_{PoolName.DRAFT}_v",
                ]
        elif pool_name == PoolName.DRAFT_SWA:
            from sglang.srt.mem_cache.memory_pool_host import (
                DeepSeekV4PagedHostPool,
            )
            from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

            if isinstance(
                host_pool,
                (DeepSeekV4PagedHostPool, MLATokenToKVPoolHost),
            ):
                return mla_component_keys(str(pool_name))
            elif isinstance(host_pool, MHATokenToKVPoolHost):
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
        elif pool_name in (
            PoolName.INDEXER,
            PoolName.DRAFT_INDEXER,
            PoolName.DEEPSEEK_V4_C4,
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.DEEPSEEK_V4_C128_STATE,
        ):
            # DSA indexer and DeepSeek V4 side pools are page-packed
            # single-object pools.
            return mla_component_keys(str(pool_name))
        elif pool_name == PoolName.SWA:
            if not self.is_mla_backend and hasattr(host_pool, "v_buffer"):
                # Ordinary MHA SWA mirrors a K/V pool.
                suffixes = [
                    f"_{self.mha_suffix}_{pool_name}_k",
                    f"_{self.mha_suffix}_{pool_name}_v",
                ]
            elif self.is_mla_backend:
                return mla_component_keys(str(pool_name))

        if not suffixes:
            raise ValueError(
                f"Unsupported Mooncake hybrid pool name: {pool_name}, "
                f"host_pool={type(host_pool)}"
            )
        key_multiplier = len(suffixes)
        component_keys = [
            f"{page_key}{suffix}" for page_key in page_keys for suffix in suffixes
        ]
        return component_keys, key_multiplier

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        if self.mem_pool_host.kv_buffer is None:
            # Logical anchor: no physical KV object exists in Mooncake, so the
            # usable prefix is determined entirely by required sidecar objects.
            kv_pages = len(keys)
        else:
            kv_pages = self.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        # Start from every KV prefix and let each pool remove the stop points it
        # cannot serve. Collect the whole set, not just its maximum: a
        # TRAILING_PAGES pool leaves holes (see PoolTransferResult), and the
        # caller has to intersect these sets across ranks.
        restorable = list(range(1, kv_pages + 1))

        for transfer in pool_transfers or []:
            if not restorable:
                break
            component_keys, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            component_keys = self._tag_keys(component_keys)
            ex = self._batch_exist(component_keys)
            # The request covers every input key, even when the KV hit
            # prefix is shorter. Check the full response before slicing it.
            expected_components = (
                len(keys) * key_multiplier
                if isinstance(key_multiplier, int)
                and not isinstance(key_multiplier, bool)
                else -1
            )
            if (
                expected_components <= 0
                or not isinstance(component_keys, (list, tuple))
                or len(component_keys) != expected_components
                or not isinstance(ex, (list, tuple))
                or len(ex) != expected_components
            ):
                # ``all([])`` is true.  A missing, truncated, or oversized
                # native existence vector must therefore remove the whole
                # sidecar prefix instead of being treated as a hit.
                page_exists = [False] * kv_pages
            else:
                page_exists = [
                    all(
                        r == 1
                        for r in ex[i * key_multiplier : (i + 1) * key_multiplier]
                    )
                    for i in range(kv_pages)
                ]
            boundary = 0
            pool_restorable = []
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                try:
                    boundary = page_exists.index(False)
                except ValueError:
                    boundary = kv_pages
                pool_restorable = list(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                # A stop point works when the window ending there is complete,
                # so scan every one instead of stopping at the longest.
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        page_exists[i]
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        pool_restorable.append(prefix_len)
                        if boundary == 0:
                            boundary = prefix_len
            else:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")
            if boundary:
                hit_count[transfer.name] = boundary
            pool_restorable_set = set(pool_restorable)
            restorable = [p for p in restorable if p in pool_restorable_set]

        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_count, restorable)

    def _batch_io_v2(self, transfers: List[PoolTransfer], is_set: bool):
        # Unified v2 I/O path: each PoolTransfer can expand to one or more
        # storage objects per logical page, but API still reports page-level result.
        results: dict = {}
        for transfer in transfers:
            host_pool = getattr(self, "registered_pools", {}).get(transfer.name)
            keys = transfer.keys
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices
            assert len(keys) > 0
            assert len(keys) == len(host_indices) // page_size

            tagged_keys = self._tag_keys(keys)
            key_strs, key_multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            key_strs = self._tag_keys(key_strs)
            ptr_list, element_size_list = host_pool.get_page_buffer_meta(host_indices)
            if len(ptr_list) != len(key_strs):
                ptr_list, element_size_list = self._pack_multi_buffer_meta(
                    key_strs, ptr_list, element_size_list
                )

            if is_set:
                group_ids = (
                    self._expand_group_ids(tagged_keys, key_multiplier)
                    if self._can_use_group_semantics()
                    else None
                )
                if self._dfs_only:
                    io_results = self._write_batch_with_repair(
                        key_strs, ptr_list, element_size_list, group_ids
                    )
                else:
                    exist_result = self._batch_exist(key_strs)
                    io_results = [0 if state == 1 else -1 for state in exist_result]
                    missing_idx = [i for i, state in enumerate(exist_result) if state != 1]
                    if missing_idx:
                        put_results = self._put_batch_zero_copy_impl(
                            [key_strs[i] for i in missing_idx],
                            [ptr_list[i] for i in missing_idx],
                            [element_size_list[i] for i in missing_idx],
                            self._filter_group_ids(group_ids, missing_idx),
                        )
                        for i, res in zip(missing_idx, put_results):
                            io_results[i] = res
            else:
                io_results = self._get_batch_zero_copy_impl(
                    key_strs, ptr_list, element_size_list
                )
                self._record_failed_page_reads(key_strs, io_results, element_size_list)
            results[transfer.name] = self._batch_postprocess(
                io_results,
                is_set_operate=is_set,
                key_multiplier=key_multiplier,
                expected_sizes=None if is_set else element_size_list,
                expected_group_count=len(keys),
            )
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict:
        return self._batch_io_v2(transfers, is_set=True)

    def _get_mha_split_heads_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = (
            self.mem_pool_host.get_split_heads_page_buffer_meta(
                indices, self.split_factor
            )
        )
        key_list = []
        for key_ in keys:
            for suffix in self.mha_suffix:
                key_list.append(f"{key_}_{suffix}_k")
                key_list.append(f"{key_}_{suffix}_v")
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    @staticmethod
    def _uses_multi_buffer(buffer_ptrs: List[Any]) -> bool:
        return bool(buffer_ptrs) and isinstance(buffer_ptrs[0], Sequence)

    @staticmethod
    def _pack_multi_buffer_meta(
        key_strs: List[str],
        ptr_list: List[int],
        element_size_list: List[int],
    ) -> Tuple[List[Any], List[Any]]:
        if len(ptr_list) == len(key_strs):
            return ptr_list, element_size_list

        assert len(key_strs) > 0
        assert len(ptr_list) == len(element_size_list)
        assert len(ptr_list) % len(key_strs) == 0

        nbuf = len(ptr_list) // len(key_strs)
        return [ptr_list[i : i + nbuf] for i in range(0, len(ptr_list), nbuf)], [
            element_size_list[i : i + nbuf]
            for i in range(0, len(element_size_list), nbuf)
        ]

    def _get_mha_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(f"{key_}_{self.mha_suffix}_k")
            key_list.append(f"{key_}_{self.mha_suffix}_v")
        if len(key_list) != len(ptr_list):
            raise RuntimeError(
                "Mooncake layer_first multi-buffer is only supported for MLA "
                "host KV pool. Use page_first/page_first_direct for MHA."
            )
        return key_list, ptr_list, element_size_list

    def _get_mla_buffer_meta(self, keys, indices):
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(indices)
        key_list = []
        for key_ in keys:
            key_list.append(self._make_mla_physical_key(key_, "k"))
        ptr_list, element_size_list = self._pack_multi_buffer_meta(
            key_list, ptr_list, element_size_list
        )
        assert len(key_list) == len(ptr_list)
        return key_list, ptr_list, element_size_list

    def _batch_preprocess(self, keys, host_indices):
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        if self.is_mla_backend:
            return self._get_mla_buffer_meta(keys, host_indices)
        else:
            if self.should_split_heads:
                return self._get_mha_split_heads_buffer_meta(keys, host_indices)
            else:
                return self._get_mha_buffer_meta(keys, host_indices)

    def _batch_postprocess(
        self,
        results: List[int],
        is_set_operate=False,
        key_multiplier=None,
        *,
        expected_sizes=None,
        expected_group_count=None,
    ):
        """Validate complete fixed-size HiCache pages returned by native reads.

        Generic get/range APIs retain their capacity-based contract. Page reads
        require exact object bytes and a complete set of components for each page.
        Writes retain the existing zero-is-success semantics.
        """
        if key_multiplier is None:
            if self.is_mla_backend:
                key_multiplier = 1
            else:
                key_multiplier = 2
                if self.should_split_heads:
                    key_multiplier *= self.split_factor

        if is_set_operate:
            # Keep the existing write-result behavior independent of read checks.
            return [
                all(res == 0 for res in results[i : i + key_multiplier])
                for i in range(0, len(results), key_multiplier)
            ]

        if (
            not isinstance(expected_group_count, int)
            or isinstance(expected_group_count, bool)
            or expected_group_count < 0
        ):
            raise ValueError("page reads require an explicit logical-page count")
        failed = [False] * expected_group_count
        if (
            not isinstance(key_multiplier, int)
            or isinstance(key_multiplier, bool)
            or key_multiplier <= 0
            or not isinstance(results, (list, tuple))
            or not isinstance(expected_sizes, (list, tuple))
        ):
            return failed
        component_count = expected_group_count * key_multiplier
        if len(results) != component_count or len(expected_sizes) != component_count:
            # Reject an incomplete final component group, even when the returned
            # vector and expected_sizes happen to have the same truncated length.
            return failed

        def exact_object_size(value):
            # The common page-first path returns one integer per object. Keep
            # that path allocation-free; packed MTP/MLA paths still validate
            # every nonnegative slice and sum them exactly.
            if isinstance(value, int) and not isinstance(value, bool):
                return value if value > 0 else None
            if not isinstance(value, (list, tuple)) or not value:
                return None
            total = 0
            for size in value:
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    return None
                total += size
            return total if total > 0 else None

        component_ok = []
        for result, sizes in zip(results, expected_sizes):
            expected = exact_object_size(sizes)
            component_ok.append(
                isinstance(result, int)
                and not isinstance(result, bool)
                and result > 0
                and expected is not None
                and result == expected
            )
        return [
            all(component_ok[i : i + key_multiplier])
            for i in range(0, component_count, key_multiplier)
        ]

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Empty batches are valid at the storage boundary (for example after
        # a fully aligned prefix is trimmed). Avoid dividing by the logical
        # page count below and keep the caller's result shape deterministic.
        if not keys:
            return []
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4's KV anchor is logical only; v2 side pools carry data.
            return [True] * len(keys)

        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)

        start_time = time.perf_counter()
        get_results = self._get_batch_zero_copy_impl(
            key_strs, buffer_ptrs, buffer_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        self._record_failed_page_reads(key_strs, get_results, buffer_sizes)

        return self._batch_postprocess(
            get_results,
            is_set_operate=False,
            key_multiplier=len(key_strs) // len(keys),
            expected_sizes=buffer_sizes,
            expected_group_count=len(keys),
        )

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        # Match batch_get_v1: an empty write is a no-op, not a malformed
        # replication request. In particular this prevents a zero-length
        # `key_strs // len(keys)` calculation from escaping into the worker.
        if not keys:
            return []
        if self.mem_pool_host.kv_buffer is None:
            # DeepSeek V4's KV anchor is logical only; v2 side pools carry data.
            return [True] * len(keys)

        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(keys, host_indices)
        key_multiplier = len(key_strs) // len(keys)
        group_ids = (
            self._expand_group_ids(keys, key_multiplier)
            if self._can_use_group_semantics()
            else None
        )
        if self._dfs_only:
            results = self._write_batch_with_repair(
                key_strs, buffer_ptrs, buffer_sizes, group_ids, track_metrics=True
            )
            return self._batch_postprocess(results, is_set_operate=True)
        exist_result = self._batch_exist(key_strs)

        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        for i in range(len(key_strs)):
            if exist_result[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
            else:
                set_results[i] = 0

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            start_time = time.perf_counter()
            put_results = self._put_batch_zero_copy_impl(
                set_keys,
                set_buffer_ptrs,
                set_buffer_sizes,
                self._filter_group_ids(group_ids, set_indices),
            )
            end_time = time.perf_counter()

            if self.enable_storage_metrics:
                self.backup_pgs.append(len(set_keys))
                self.backup_bandwidth.append(
                    len(set_keys) / (end_time - start_time) * self.gb_per_page
                )

            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._batch_postprocess(set_results, is_set_operate=True)

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_location is not None and target_sizes is not None
        if getattr(self, "_dfs_only", False):
            results = self._write_batch_with_repair(
                [key], [target_location], [target_sizes]
            )
            return bool(results) and results[0] == 0
        exist_result = self._batch_exist([key])
        if exist_result[0] == 1:
            return True
        put_result = self._put_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return put_result[0] == 0

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        # Only support zero copy set for now
        assert target_locations is not None and target_sizes is not None
        assert len(keys) == len(target_locations) == len(target_sizes)

        if len(keys) == 0:
            return False

        for i in range(len(keys)):
            if (
                keys[i] is None
                or target_locations[i] is None
                or target_sizes[i] is None
            ):
                return False

        if getattr(self, "_dfs_only", False):
            results = self._write_batch_with_repair(
                keys, target_locations, target_sizes
            )
            return bool(results) and all(result == 0 for result in results)

        exist_result = self._batch_exist(keys)
        set_keys = []
        set_target_locations = []
        set_target_sizes = []
        set_indices = []
        for i in range(len(keys)):
            if exist_result[i] != 1:
                set_keys.append(keys[i])
                set_target_locations.append(target_locations[i])
                set_target_sizes.append(target_sizes[i])
                set_indices.append(i)
        # Only set non-existing keys to storage
        start_time = time.perf_counter()
        put_result = self._put_batch_zero_copy_impl(
            set_keys, set_target_locations, set_target_sizes
        )
        end_time = time.perf_counter()

        if self.enable_storage_metrics:
            self.backup_pgs.append(len(set_keys))
            self.backup_bandwidth.append(
                len(set_keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(set_indices)):
            if put_result[i] == 0:
                exist_result[set_indices[i]] = 1

        success_count = 0
        for i in range(len(keys)):
            if exist_result[i] == 0:
                break
            success_count += 1
        # TODO: return the number of consecutive successful operations from the start.
        return success_count == len(keys)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        assert target_location is not None and target_sizes is not None
        get_result = self._get_batch_zero_copy_impl(
            [key], [target_location], [target_sizes]
        )
        return get_result[0] >= 0

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> int:
        assert len(keys) == len(target_locations) == len(target_sizes)
        if len(keys) == 0:
            return 0

        start_time = time.perf_counter()
        get_result = self._get_batch_zero_copy_impl(
            keys, target_locations, target_sizes
        )
        end_time = time.perf_counter()

        if self.is_mla_backend:
            key_multiplier = 1
        else:
            key_multiplier = 2

        if self.enable_storage_metrics:
            self.prefetch_pgs.append(len(keys))
            self.prefetch_bandwidth.append(
                len(keys) / (end_time - start_time) * self.gb_per_page
            )

        for i in range(len(keys)):
            if get_result[i] < 0:
                return i // key_multiplier
        return len(keys) // key_multiplier

    def exists(self, key) -> bool:
        exist_result = self._batch_exist([key])
        return exist_result[0] == 1

    def batch_exists(
        self, keys, extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        # Apply config prefix if available.
        keys = self._tag_keys(keys)

        if self.is_mla_backend:
            query_keys = [
                self._make_mla_physical_key(key, "k") for key in keys
            ]
            key_multiplier = 1
        else:
            query_keys = []
            if self.should_split_heads:
                for key in keys:
                    for suffix in self.mha_suffix:
                        query_keys.append(f"{key}_{suffix}_k")
                        query_keys.append(f"{key}_{suffix}_v")
                key_multiplier = 2 * self.split_factor
            else:
                for key in keys:
                    query_keys.append(f"{key}_{self.mha_suffix}_k")
                    query_keys.append(f"{key}_{self.mha_suffix}_v")
                key_multiplier = 2

        exist_result = self._batch_exist(query_keys)
        for i in range(len(query_keys)):
            if exist_result[i] != 1:
                return i // key_multiplier
        return len(query_keys) // key_multiplier

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> None:
        self.store.remove_all()

    def _put_batch_zero_copy_impl(
        self,
        key_strs: List[str],
        buffer_ptrs: List[Any],
        buffer_sizes: List[Any],
        group_ids: Optional[List[str]] = None,
    ) -> List[int]:
        config = None
        if self._can_use_group_semantics() and group_ids is not None:
            if len(group_ids) != len(key_strs):
                raise ValueError(
                    "Mooncake group_ids length must match key_strs length: "
                    f"{len(group_ids)} != {len(key_strs)}"
                )
            config = self._replicate_config_cls()
            config.group_ids = group_ids

        if self._needs_explicit_replicate_config():
            config = config or self._make_replicate_config()
            # Keep the explicit MEMORY count independent from DFS count. This
            # is the only place that changes the topology for DFS-only mode.
            config.replica_num = self._effective_replica_num

        # 需要写 DFS 副本时必须显式传 ReplicateConfig。
        #
        # 不传的话走的是 batch_put_from 的三参重载，C++ 侧用默认构造的
        # ReplicateConfig，其 dfs_replica_num 为 0（replica.h:107）——
        # 也就是说 **不做这一步，挂载的并行文件系统永远不会被写入**。
        #
        # 与上面的 group_ids 共用同一个 config 对象：两者都只是在配置上
        # 设不同字段，没有互斥关系。
        if self._use_dfs_replica:
            config = config or self._replicate_config_cls()
            config.dfs_replica_num = self.config.dfs_replica_num

        if self._uses_multi_buffer(buffer_ptrs):
            config = config or self._replicate_config_cls()
            return self.store.batch_put_from_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes, config
            )
        elif config is not None:
            return self.store.batch_put_from(
                key_strs, buffer_ptrs, buffer_sizes, config
            )
        else:
            return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    @staticmethod
    def _repair_object_size(value):
        if type(value) is int:
            return value if value > 0 else None
        if not isinstance(value, (list, tuple)) or not value:
            return None
        if any(type(size) is not int or size < 0 for size in value):
            return None
        total = sum(value)
        return total if total > 0 else None

    def _record_failed_page_reads(self, keys, results, sizes):
        if not getattr(self, "_dfs_only", False):
            return
        if (
            not isinstance(keys, (list, tuple))
            or not isinstance(results, (list, tuple))
            or not isinstance(sizes, (list, tuple))
            or not keys
            or len(keys) != len(results)
            or len(keys) != len(sizes)
            or any(type(key) is not str or not key for key in keys)
            or any(type(result) is not int for result in results)
        ):
            return
        totals = [self._repair_object_size(size) for size in sizes]
        if any(total is None for total in totals):
            return
        failures = [
            (key, total)
            for key, result, total in zip(keys, results, totals)
            if result == -801 or 0 < result < total
        ]
        if failures and self._dfs_repair_state.record(failures):
            logger.warning(
                "DFS repair tracking limit reached; metadata-only backup acknowledgements are disabled"
            )

    def get_repair_status(self):
        """Report aggregate state only; never expose keys or buffer addresses."""
        return self._dfs_repair_state.status()

    def _upsert_repair_batch(self, keys, pointers, sizes, group_ids):
        multi = self._uses_multi_buffer(pointers)
        method = getattr(
            self.store,
            "batch_upsert_from_multi_buffers" if multi else "batch_upsert_from",
            None,
        )
        if not callable(method):
            return [-1] * len(keys)
        config = self._make_replicate_config()
        # Repair never reintroduces a distributed MEMORY or NoF replica.
        config.replica_num = 0
        config.nof_replica_num = 0
        config.dfs_replica_num = 1
        if self._can_use_group_semantics() and group_ids is not None:
            config.group_ids = group_ids
        return method(keys, pointers, sizes, config)

    def _write_batch_with_repair(self, keys, pointers, sizes, group_ids=None, track_metrics=False):
        """Synchronous backup of current owned L2 data; no repair from read data."""
        failed = [-1] * len(keys)
        if (
            len(keys) != len(pointers)
            or len(keys) != len(sizes)
            or any(type(key) is not str or not key for key in keys)
            or (group_ids is not None and len(group_ids) != len(keys))
        ):
            return failed
        totals = [self._repair_object_size(size) for size in sizes]
        if any(total is None for total in totals):
            return failed
        try:
            exists = self._batch_exist(keys)
        except Exception:
            logger.warning("DFS backup existence query raised; backup remains pending")
            return failed
        if (
            not isinstance(exists, (list, tuple))
            or len(exists) != len(keys)
            or any(type(value) is not int for value in exists)
        ):
            return failed
        state = self._dfs_repair_state
        existing, ordinary, repairs, failure_epoch = state.classify(
            keys, totals, exists
        )
        ready_existing = []
        if existing:
            statuses = self._resolve_existing_readiness(
                [keys[index] for index in existing]
            )
            for index, status in zip(existing, statuses):
                if status == "ready":
                    ready_existing.append(index)
                elif status == "missing":
                    # The metadata disappeared while the duplicate gate was
                    # polling; allow a fresh PutStart to take ownership.
                    ordinary.append(index)
                else:
                    # PROCESSING, RPC failure, and timeout all fail closed.
                    pass
        result = failed[:]

        def invoke(indices, repair):
            if not indices:
                return []
            selected_keys = [keys[index] for index in indices]
            selected_ptrs = [pointers[index] for index in indices]
            selected_sizes = [sizes[index] for index in indices]
            selected_groups = self._filter_group_ids(group_ids, indices)
            started = time.perf_counter()
            try:
                method = self._upsert_repair_batch if repair else self._put_batch_zero_copy_impl
                values = method(selected_keys, selected_ptrs, selected_sizes, selected_groups)
            except Exception:
                logger.warning("DFS backup native call raised; backup remains pending")
                return [-1] * len(indices)
            finally:
                if track_metrics and self.enable_storage_metrics:
                    elapsed = time.perf_counter() - started
                    self.backup_pgs.append(len(indices))
                    self.backup_bandwidth.append(len(indices) / elapsed * self.gb_per_page)
            if (
                not isinstance(values, (list, tuple))
                or len(values) != len(indices)
                or any(type(value) is not int for value in values)
            ):
                return [-1] * len(indices)
            return list(values)

        try:
            ordinary_results = invoke(ordinary, False)
            for index, value in zip(ordinary, ordinary_results):
                # Overflow or a new failed read can arrive during native Put.
                # Its zero status may mean duplicate/no-op, so recheck state.
                if value == 0 and state.can_ack_existing(keys[index], failure_epoch):
                    result[index] = 0
            repair_indices = [index for index, _, _ in repairs]
            repair_results = invoke(repair_indices, True)
            # A malformed or partially failed repair vector clears no claims.
            repair_success = bool(repairs) and len(repair_results) == len(repairs) and all(
                type(value) is int and value == 0 for value in repair_results
            )
            for index, key, claim_id in repairs:
                if state.finish(key, claim_id, repair_success):
                    result[index] = 0
            for index in ready_existing:
                if state.can_ack_existing(keys[index], failure_epoch):
                    result[index] = 0
            return result
        finally:
            # Also release local claims if bookkeeping unexpectedly raises.
            # False never clears the failure marker or a newer generation.
            for _, key, claim_id in repairs:
                state.finish(key, claim_id, False)

    def _resolve_existing_readiness(self, keys):
        """Wait for existing DFS keys to become readable COMPLETE objects.

        ``batch_is_exist`` reports metadata presence, including PROCESSING
        objects. The master admin endpoint is preferred because its
        ``/batch_query_keys`` path reads COMPLETE metadata without incrementing
        the ordinary ``BatchGetReplicaList`` failure counters. The native query
        remains a compatibility fallback for older Mooncake builds. Missing
        keys are rechecked with ``batch_is_exist`` so a key revoked during the
        wait can be put again. Any unavailable probe fails closed.
        """
        if not keys:
            return []

        statuses = ["processing"] * len(keys)
        pending = list(range(len(keys)))
        deadline = time.monotonic() + self._DFS_DUPLICATE_READY_TIMEOUT_S
        backoffs = iter(self._DFS_DUPLICATE_READY_BACKOFF_S)
        delay = next(backoffs, self._DFS_DUPLICATE_READY_BACKOFF_S[-1])
        while pending:
            selected = [keys[index] for index in pending]
            replica_map = self._batch_readiness_probe(selected)
            try:
                exists = self._batch_exist(selected)
            except Exception:
                exists = [1] * len(selected)

            next_pending = []
            for local_index, index in enumerate(pending):
                key = selected[local_index]
                if replica_map is None:
                    # Mooncake's native ExistKey path already requires a
                    # readable COMPLETE replica. Use it when the selected
                    # build has neither the admin endpoint nor the legacy
                    # replica-description API.
                    if local_index < len(exists) and exists[local_index] == 1:
                        statuses[index] = "ready"
                    else:
                        statuses[index] = "missing"
                    continue
                if not isinstance(replica_map, dict):
                    replica_map = {}
                replicas = replica_map.get(key)
                if isinstance(replicas, (list, tuple)) and replicas:
                    statuses[index] = "ready"
                elif local_index < len(exists) and exists[local_index] == 0:
                    statuses[index] = "missing"
                else:
                    statuses[index] = "processing"
                    next_pending.append(index)
            pending = next_pending
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = next(
                backoffs,
                min(delay * 2, self._DFS_DUPLICATE_READY_BACKOFF_S[-1]),
            )

        if pending:
            logger.warning(
                "DFS duplicate readiness gate timed out; %d keys remain pending",
                len(pending),
            )
        return statuses

    def _batch_readiness_probe(self, keys):
        """Return keys with readable COMPLETE replicas without polluting counters."""
        admin_result = self._batch_admin_readiness_probe(keys)
        if admin_result is not None:
            return admin_result

        query = getattr(self.store, "batch_get_replica_desc", None)
        if not callable(query):
            return None
        logger.warning(
            "DFS readiness probe falling back to ordinary BatchGetReplicaList; "
            "PROCESSING probes may be included in master failure counters"
        )
        try:
            return query(keys)
        except Exception:
            return {}

    def _batch_admin_readiness_probe(self, keys):
        """Query the Mooncake admin endpoint when the selected build exposes it."""
        config = getattr(self, "config", None)
        master = getattr(config, "master_server_address", None)
        port = getattr(config, "master_metrics_port", None)
        if not master or not port or any("," in key for key in keys):
            return None

        # master_server_address is the existing ``host:rpc_port`` setting. The
        # admin HTTP server uses the already pinned master_metrics_port.
        host = str(master).split(":", 1)[0]
        url = f"http://{host}:{int(port)}/batch_query_keys"
        try:
            response = requests.get(
                url,
                params={"keys": ",".join(keys)},
                timeout=3,
            )
            if getattr(response, "status_code", None) != 200:
                return None
            payload = response.json()
        except Exception:
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None
        return {
            key: [True]
            for key in keys
            if isinstance(data.get(key), dict) and data[key].get("ok") is True
        }

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[Any], buffer_sizes: List[Any]
    ) -> List[int]:
        if self._uses_multi_buffer(buffer_ptrs):
            return self.store.batch_get_into_multi_buffers(
                key_strs, buffer_ptrs, buffer_sizes
            )
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def _batch_exist(self, key_strs: List[str]) -> List[int]:
        return self.store.batch_is_exist(key_strs)

    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics
