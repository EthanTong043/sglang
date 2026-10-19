"""Bounded, disabled-by-default tracing helper for prefetch diagnosis.

The generated overlay calls :func:`trace_event` at lifecycle boundaries. The
caller only enqueues a small JSON-safe record; a daemon writer performs file
I/O. Tracing never changes a timeout, queue decision, return value, exception,
or buffer lifetime. This module intentionally has no SGLang or Torch imports.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Any


_TRUE = {"1", "true", "yes", "on"}
_SAFE_FIELDS = {
    "operation_id", "request_id", "phase", "status", "reason", "policy",
    "pool", "batch_index", "batch_pages", "page_count", "queue_depth",
    "requested_tokens", "completed_tokens", "storage_hit_tokens",
    "host_tokens", "byte_count", "deadline_s", "elapsed_s",
    "can_terminate", "operation_terminated", "pool_transfers_done",
    "prefetch_done", "waiting_queue_len", "forward_batch_size",
    "draft_depth", "virtual_layer", "host_layer", "host_layer_count",
}
_MAX_STRING = 80
_MAX_EVENTS_DEFAULT = 4096
_QUEUE_CAPACITY_DEFAULT = 256


def _positive_int(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def _safe_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -(1 << 63) <= value < (1 << 63) else None
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return value[:_MAX_STRING]
    return None


class _PrefetchTrace:
    def __init__(self) -> None:
        self.max_events = _positive_int(
            "SGLANG_PREFETCH_TRACE_MAX_EVENTS", _MAX_EVENTS_DEFAULT, 100_000
        )
        self.queue_capacity = _positive_int(
            "SGLANG_PREFETCH_TRACE_QUEUE_CAPACITY", _QUEUE_CAPACITY_DEFAULT, 4096
        )
        self.path = Path(
            os.environ.get(
                "SGLANG_PREFETCH_TRACE_PATH",
                f"/tmp/sglang-prefetch-trace-{os.getpid()}.jsonl",
            )
        )
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=self.queue_capacity
        )
        self._secret = secrets.token_bytes(16)
        self._accepted = 0
        self._dropped = 0
        self._lock = threading.Lock()
        self._closed = False
        self._writer = threading.Thread(
            target=self._write_loop,
            name="sglang-prefetch-trace-writer",
            daemon=True,
        )
        self._writer.start()

    def _rid_hash(self, value: Any) -> str | None:
        if value is None:
            return None
        return hashlib.blake2b(
            str(value).encode("utf-8", "replace"), key=self._secret, digest_size=8
        ).hexdigest()

    def emit(self, event: str, **fields: Any) -> None:
        with self._lock:
            if self._closed or self._accepted >= self.max_events:
                self._dropped += 1
                return
            self._accepted += 1
        record: dict[str, Any] = {
            "schema": "sglang-prefetch-trace-v1",
            "event": str(event)[:_MAX_STRING],
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
        }
        request_id = fields.pop("request_id", None)
        if request_id is not None:
            record["rid_hash"] = self._rid_hash(request_id)
        if fields.get("operation_id") is not None:
            record["operation_id"] = _safe_value(fields.pop("operation_id"))
        for key, value in fields.items():
            if key not in _SAFE_FIELDS:
                continue
            clean = _safe_value(value)
            if clean is not None:
                record[key] = clean
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def _write_loop(self) -> None:
        stream = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.path.open("a", encoding="utf-8")
            while True:
                record = self._queue.get()
                if record is None:
                    break
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
        except Exception:
            # Diagnostics must never affect serving. The accepted/dropped counts
            # remain available to a local test through close_trace().
            pass
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def close(self) -> dict[str, int]:
        with self._lock:
            if self._closed:
                return {"accepted": self._accepted, "dropped": self._dropped}
            self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The writer will drain before receiving the sentinel. This path is
            # only used at shutdown and is intentionally bounded.
            deadline = time.monotonic() + 0.5
            while self._writer.is_alive() and time.monotonic() < deadline:
                try:
                    self._queue.put(None, timeout=0.01)
                    break
                except queue.Full:
                    continue
        self._writer.join(timeout=0.5)
        with self._lock:
            return {"accepted": self._accepted, "dropped": self._dropped}


_TRACE: _PrefetchTrace | None = None
_TRACE_LOCK = threading.Lock()


def _get_trace() -> _PrefetchTrace | None:
    global _TRACE
    if os.environ.get("SGLANG_PREFETCH_TRACE", "0").lower() not in _TRUE:
        return None
    if _TRACE is None:
        with _TRACE_LOCK:
            if _TRACE is None:
                _TRACE = _PrefetchTrace()
    return _TRACE


def trace_event(event: str, **fields: Any) -> None:
    """Enqueue one bounded safe event; all failures are swallowed."""
    try:
        trace = _get_trace()
        if trace is not None:
            trace.emit(event, **fields)
    except Exception:
        pass


def close_trace() -> dict[str, int]:
    trace = _TRACE
    return trace.close() if trace is not None else {"accepted": 0, "dropped": 0}


atexit.register(close_trace)
