"""CPU regressions for drop attribution when the UNIFIED tree evicts an
unbacked device leaf under a write-through policy.

Why this file exists: the original instrumentation (commit e859c4fce3) landed in
hiradix_cache.py::_evict_regular, which is dead code -- registry.py only ever
constructs UnifiedRadixCache. These tests cover the live path instead.

Both halves of the live path are executed from production source, AST-loaded
with no torch / GPU / sglang import:

  1. UnifiedTreeCore.evict_device_leaf  -- decides the `dropped_reason`
     (tree layer, owns hit_count, has NO metrics_collector)
  2. UnifiedRadixCache._evict_device_leaf -- turns that reason into a
     Prometheus increment (controller layer, owns metrics_collector)

A test that only modelled one half would pass with the other half missing, so
the reason label and the counter increment are asserted end-to-end.

Not covered: real transfers, host allocation, kv-event payloads, the eviction
priority policy, and the Rust TreeCore (its binding does not export
dropped_reason yet -- see adapter.py).
"""

from __future__ import annotations

import ast
import runpy
import unittest
from collections import defaultdict
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"

register_cpu_ci = runpy.run_path(str(REPO / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

WRITE_THROUGH_THRESHOLD = 2


def _extract_function(path: Path, class_name: str, func_name: str, scope: dict):
    """Exec a single production method body into `scope` as a bare function.

    Keeps the real statements -- including the attribution branch -- so that
    deleting or inverting them in production makes these tests fail.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == func_name:
                child = ast.FunctionDef(
                    name=child.name,
                    args=child.args,
                    body=child.body,
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                    type_params=[],
                )
                mod = ast.Module(
                    body=ast.parse("from __future__ import annotations\n").body
                    + [child],
                    type_ignores=[],
                )
                exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), scope)
                return scope[func_name]
    raise AssertionError(f"{class_name}.{func_name} not found in {path}")


class ComponentType(IntEnum):
    FULL = 0
    SWA = 1
    MAMBA = 2


class _EvictDeviceLeafResult:
    """Stand-in for the msgspec struct, with the production default."""

    def __init__(self):
        self.backup_kv = None
        self.dropped_reason = None  # production default: not a drop
        self.device_frees = defaultdict(list)
        self.host_frees = defaultdict(list)
        self.tracker = defaultdict(int)


class _MetricsCollector:
    def __init__(self):
        self.dropped = []

    def increment_dropped_tokens(self, num_tokens: int, reason: str, pool: str) -> None:
        self.dropped.append((num_tokens, reason, pool))

    def tokens_by_reason(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for num_tokens, reason, _pool in self.dropped:
            totals[reason] = totals.get(reason, 0) + num_tokens
        return totals


class _Node:
    def __init__(self, node_id, *, hit_count, backuped, num_tokens):
        self.id = node_id
        self.hit_count = hit_count
        self._backuped = backuped
        self.num_tokens = num_tokens
        self.evicted = False

    @property
    def backuped(self):
        return self._backuped


class _TreeCore:
    """Minimal host for the real evict_device_leaf body."""

    def __init__(self, node, *, is_write_back=False):
        self.node = node
        self.write_through_threshold = WRITE_THROUGH_THRESHOLD
        self.deleted = []
        self.demoted = []
        self._evict_device_leaf_impl = _extract_function(
            SRT / "mem_cache/unified_cache/unified_tree_core.py",
            "UnifiedTreeCore",
            "evict_device_leaf",
            {
                "EvictDeviceLeafResult": _EvictDeviceLeafResult,
                "__name__": __name__,
            },
        )

    def node_by_id(self, node_id):
        return self.node

    def _is_device_leaf(self, node):
        return True

    def _build_backup_kv_action(self, node, write_back=False):
        return SimpleNamespace(node_id=node.id)

    def _delete_unbacked_device_leaf(
        self, node, tracker, device_frees, host_frees
    ):
        # Real helper frees every component layer and cascades tombstone
        # ancestors; the token counts land in `tracker`, which is what the
        # controller reports. Model the FULL pool plus an ancestor-chain tail.
        self.deleted.append(node.id)
        tracker[ComponentType.FULL] += node.num_tokens
        node.evicted = True

    def _demote(self, node, tracker, device_frees, host_frees):
        self.demoted.append(node.id)
        tracker[ComponentType.FULL] += node.num_tokens

    def evict_device_leaf(self, node_id, is_write_back):
        return self._evict_device_leaf_impl(self, node_id, is_write_back)


class _Controller:
    """Minimal host for the real UnifiedRadixCache._evict_device_leaf body."""

    def __init__(self, tree_core, *, with_metrics=True, is_write_back=False):
        self.tree_core = tree_core
        self.is_write_back = is_write_back
        self.metrics_collector = _MetricsCollector() if with_metrics else None
        pool_label = {
            ComponentType.FULL: "kv",
            ComponentType.SWA: "swa",
            ComponentType.MAMBA: "mamba",
        }
        self._impl = _extract_function(
            SRT / "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "_evict_device_leaf",
            {
                "_COMPONENT_POOL_LABEL": pool_label,
                "__name__": __name__,
            },
        )

    def _free_values(self, device_frees, host_frees):
        device_frees.clear()
        host_frees.clear()

    def _accumulate_tracker(self, tracker, delta):
        for ct, n in delta.items():
            tracker[ct] += n

    def evict(self, node_id):
        tracker = defaultdict(int)
        backup_kv = self._impl(self, node_id, tracker)
        return backup_kv, tracker


class TestUnifiedEvictLeafDroppedTokens(unittest.TestCase):
    def _run(self, *, hit_count, backuped, num_tokens=4, **kwargs):
        node = _Node(7, hit_count=hit_count, backuped=backuped, num_tokens=num_tokens)
        core = _TreeCore(node, is_write_back=kwargs.get("is_write_back", False))
        ctrl = _Controller(core, **kwargs)
        backup_kv, tracker = ctrl.evict(7)
        return ctrl, core, backup_kv, tracker

    def test_below_threshold_leaf_is_attributed_to_starvation(self):
        """Evicted before earning its 2nd hit -> threshold starvation."""
        ctrl, core, backup_kv, tracker = self._run(hit_count=1, backuped=False)
        self.assertIsNone(backup_kv)
        self.assertEqual(core.deleted, [7])
        self.assertEqual(
            ctrl.metrics_collector.dropped, [(4, "below_write_threshold", "kv")]
        )
        self.assertEqual(tracker[ComponentType.FULL], 4)

    def test_threshold_reached_but_unbacked_is_attributed_to_backup_miss(self):
        ctrl, core, _bk, _t = self._run(
            hit_count=WRITE_THROUGH_THRESHOLD, backuped=False
        )
        self.assertEqual(ctrl.metrics_collector.dropped, [(4, "never_backuped", "kv")])

    def test_extra_hits_beyond_threshold_still_count_as_backup_miss(self):
        ctrl, _c, _bk, _t = self._run(
            hit_count=WRITE_THROUGH_THRESHOLD + 5, backuped=False
        )
        self.assertEqual(ctrl.metrics_collector.dropped, [(4, "never_backuped", "kv")])

    def test_backuped_leaf_is_demoted_and_never_counted_as_dropped(self):
        """Demotion is not loss: the host copy survives and is loadable back."""
        ctrl, core, _bk, _t = self._run(hit_count=1, backuped=True)
        self.assertEqual(core.demoted, [7])
        self.assertEqual(core.deleted, [])
        self.assertEqual(ctrl.metrics_collector.dropped, [])

    def test_write_back_defers_and_does_not_attribute_a_drop(self):
        """write_back returns a BackupKV; the drop decision happens later and
        is charged to host_pressure by _record_dropped_tokens, not here."""
        ctrl, core, backup_kv, _t = self._run(
            hit_count=1, backuped=False, is_write_back=True
        )
        self.assertIsNotNone(backup_kv)
        self.assertEqual(core.deleted, [])
        self.assertEqual(ctrl.metrics_collector.dropped, [])

    def test_reason_is_decided_before_the_node_leaves_the_tree(self):
        """hit_count must be read before the delete helper detaches the node."""
        node = _Node(7, hit_count=1, backuped=False, num_tokens=2)
        core = _TreeCore(node)
        observed = []
        original = core._delete_unbacked_device_leaf

        def spy(n, tracker, device_frees, host_frees):
            observed.append(n.hit_count)
            return original(n, tracker, device_frees, host_frees)

        core._delete_unbacked_device_leaf = spy
        ctrl = _Controller(core)
        ctrl.evict(7)
        self.assertEqual(observed, [1])
        self.assertEqual(
            ctrl.metrics_collector.dropped, [(2, "below_write_threshold", "kv")]
        )

    def test_multi_component_drop_is_split_per_pool(self):
        """A unified eviction can free several pools at once (and cascade the
        ancestor chain); every pool must be charged under the same reason."""
        node = _Node(7, hit_count=0, backuped=False, num_tokens=4)
        core = _TreeCore(node)
        original = core._delete_unbacked_device_leaf

        def multi(n, tracker, device_frees, host_frees):
            original(n, tracker, device_frees, host_frees)
            tracker[ComponentType.SWA] += 3  # aux pool freed on the same node
            tracker[ComponentType.FULL] += 6  # cascaded tombstone ancestors

        core._delete_unbacked_device_leaf = multi
        ctrl = _Controller(core)
        ctrl.evict(7)
        self.assertEqual(
            sorted(ctrl.metrics_collector.dropped),
            sorted([(10, "below_write_threshold", "kv"),
                    (3, "below_write_threshold", "swa")]),
        )

    def test_zero_token_step_is_not_reported(self):
        """A step that frees nothing must not emit a zero-delta sample."""
        node = _Node(7, hit_count=1, backuped=False, num_tokens=0)
        core = _TreeCore(node)
        ctrl = _Controller(core)
        ctrl.evict(7)
        self.assertEqual(ctrl.metrics_collector.dropped, [])

    def test_eviction_without_a_metrics_collector_still_drops(self):
        """Metrics are optional; the guard must not change eviction behavior."""
        ctrl, core, _bk, tracker = self._run(
            hit_count=1, backuped=False, with_metrics=False
        )
        self.assertIsNone(ctrl.metrics_collector)
        self.assertEqual(core.deleted, [7])
        self.assertEqual(tracker[ComponentType.FULL], 4)


if __name__ == "__main__":
    unittest.main()
