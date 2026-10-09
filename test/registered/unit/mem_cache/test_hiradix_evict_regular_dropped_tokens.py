"""CPU regressions for drop attribution when HiCache evicts a non-backuped leaf.

Execute the real HiRadixCache eviction ASTs from this checkout -- the
write-through eviction loop and both of its leaf exits -- against a tree model,
without importing SGLang, torch, or any GPU module. The device allocator is a
resource oracle (a slot cannot be freed twice), and the pool label comes from
the production PoolName enum rather than a literal, so a renamed label fails
here. These tests do not validate real transfers, host allocation, kv-event
payloads, or the eviction priority policy.

Covers the gap at _evict_regular: tokens destroyed because a node never
reached write_through_threshold hits (write_through_selective starvation) were
previously invisible, unlike the host-pressure drop in _drop_subtree_no_host.
"""

from __future__ import annotations

import ast
import heapq
import runpy
import time
import unittest
from collections import defaultdict
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SRT = REPO / "python/sglang/srt"

# CI registration itself is dependency-free. Importing sglang would load the
# optional runtime dependencies before these CPU-only tests could run.
register_cpu_ci = runpy.run_path(str(REPO / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

WRITE_THROUGH_THRESHOLD = 2


def _load_classes(path, wanted, scope, bases=None):
    """Exec the named production classes, keeping only the wanted methods."""
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ClassDef) or node.name not in wanted:
            continue
        members = wanted[node.name]
        if members is not None:
            node.body = [
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AnnAssign, ast.Assign))
                and getattr(child, "name", None) in members
            ]
            if {child.name for child in node.body} != set(members):
                raise AssertionError(f"Missing production methods in {path}")
        if bases is not None and node.name in bases:
            node.bases = [
                ast.Name(id=name, ctx=ast.Load()) for name in bases[node.name]
            ]
        node.decorator_list = []
        nodes.append(node)
    if {node.name for node in nodes} != set(wanted):
        raise AssertionError(f"Missing production classes in {path}")
    future = ast.parse("from __future__ import annotations\n").body
    code = ast.fix_missing_locations(ast.Module(body=future + nodes, type_ignores=[]))
    exec(compile(code, str(path), "exec"), scope)


def _actual_types():
    scope = {
        "__name__": __name__,
        "defaultdict": defaultdict,
        "Enum": Enum,
        "heapq": heapq,
        "time": time,
        "StorageMedium": SimpleNamespace(GPU="GPU", CPU="CPU"),
    }
    # The real pool label, so a rename of PoolName.KV breaks these tests.
    _load_classes(SRT / "mem_cache/hicache_storage.py", {"PoolName": None}, scope)
    # The real node: `evicted`/`backuped` drive which eviction exit a leaf takes.
    _load_classes(SRT / "mem_cache/radix_cache.py", {"TreeNode": None}, scope)
    _load_classes(
        SRT / "mem_cache/radix_cache.py",
        {"RadixCache": {"_delete_leaf", "_update_leaf_status"}},
        scope,
        {"RadixCache": []},
    )
    scope["BaseRadixCache"] = scope["RadixCache"]
    _load_classes(
        SRT / "mem_cache/hiradix_cache.py",
        {
            "HiRadixCache": {
                "_make_eviction_heap",
                "_promote_parent",
                "_evict_write_through",
                "_detach_backuped",
                "_evict_backuped",
                "_evict_regular",
                "_update_host_leaf_status",
            }
        },
        scope,
        {"HiRadixCache": ["BaseRadixCache"]},
    )
    return scope


class _Key:
    """Stand-in for RadixKey: only child-keying and length are exercised."""

    def __init__(self, token: str, length: int):
        self.token = token
        self.length = length

    def child_key(self, page_size: int = 1):
        return self.token

    def __len__(self):
        return self.length


class _Slots(list):
    """Device slots. Identity matters: the allocator rejects a double free."""


class _DeviceAllocator:
    """Resource oracle: a device slot cannot be freed twice or unowned."""

    def __init__(self):
        self.owned = set()
        self.freed = []
        self.next_slot = 0

    def alloc(self, count: int) -> _Slots:
        slots = _Slots(range(self.next_slot, self.next_slot + count))
        self.next_slot += count
        self.owned.update(slots)
        return slots

    def free(self, slots) -> None:
        for slot in slots:
            if slot not in self.owned:
                raise AssertionError("device slot freed twice or not owned")
            self.owned.remove(slot)
            self.freed.append(slot)


class _MetricsCollector:
    """Records increment_dropped_tokens / increment_backup_num_tokens calls."""

    def __init__(self):
        self.dropped = []
        self.backed_up = []

    def increment_dropped_tokens(self, num_tokens: int, reason: str, pool: str) -> None:
        self.dropped.append((num_tokens, reason, pool))

    def increment_backup_num_tokens(self, num_tokens: int, pool: str) -> None:
        self.backed_up.append((num_tokens, pool))

    def tokens_by_reason(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for num_tokens, reason, _pool in self.dropped:
            totals[reason] = totals.get(reason, 0) + num_tokens
        return totals


class _Fixture:
    """A flat tree of sibling device leaves under the root."""

    def __init__(self, types, *, with_metrics: bool = True):
        self.types = types
        self.allocator = _DeviceAllocator()
        self.metrics = _MetricsCollector() if with_metrics else None
        self.removed_events = []
        self.demoted = []

        tree = types["HiRadixCache"]()
        tree.page_size = 1
        tree.evictable_size_ = 0
        tree.evictable_leaves = set()
        tree.evictable_host_leaves = set()
        tree.write_through_threshold = WRITE_THROUGH_THRESHOLD
        tree.metrics_collector = self.metrics
        tree.kv_events = SimpleNamespace(record_remove=self._record_remove)
        tree.cache_controller = SimpleNamespace(
            write_policy="write_through_selective",
            mem_pool_device_allocator=self.allocator,
            evict_device=self.demoted.append,
        )
        # Evict in insertion order, independent of the real priority policy.
        self.order = {}
        tree.eviction_strategy = SimpleNamespace(
            get_priority=lambda node: self.order[node.id]
        )

        root = types["TreeNode"]()
        root.key = _Key("root", 0)
        root.value = _Slots()
        root.host_value = _Slots()
        root.parent = None
        root.children = {}
        tree.root_node = root
        self.tree = tree

    def _record_remove(self, node, medium=None):
        self.removed_events.append((node.id, medium))

    def leaf(self, token: str, *, num_tokens: int, hit_count: int, backuped: bool):
        node = self.types["TreeNode"]()
        node.key = _Key(token, num_tokens)
        node.value = self.allocator.alloc(num_tokens)
        node.host_value = _Slots(range(num_tokens)) if backuped else None
        node.hit_count = hit_count
        node.parent = self.tree.root_node
        node.children = {}
        self.tree.root_node.children[token] = node
        self.tree.evictable_size_ += num_tokens
        self.tree.evictable_leaves.add(node)
        self.order[node.id] = len(self.order)
        if backuped and self.metrics is not None:
            # A real backup is counted once, on its ack, by the backup series.
            self.metrics.increment_backup_num_tokens(
                num_tokens, self.types["PoolName"].KV.value
            )
        return node


class TestEvictRegularDroppedTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types = _actual_types()

    def fixture(self, **kwargs):
        return _Fixture(self.types, **kwargs)

    def test_below_threshold_leaf_is_attributed_to_starvation(self):
        """A node evicted before earning its 2nd hit is threshold starvation."""
        fx = self.fixture()
        node = fx.leaf("a", num_tokens=4, hit_count=1, backuped=False)
        self.assertLess(node.hit_count, fx.tree.write_through_threshold)

        self.assertEqual(fx.tree._evict_write_through(4), 4)

        self.assertEqual(fx.metrics.dropped, [(4, "below_write_threshold", "kv")])
        self.assertEqual(fx.metrics.backed_up, [])
        # The drop is real: device slots are gone and the node left the tree.
        self.assertEqual(sorted(fx.allocator.freed), [0, 1, 2, 3])
        self.assertEqual(fx.allocator.owned, set())
        self.assertEqual(fx.tree.root_node.children, {})
        self.assertEqual(fx.tree.evictable_size_, 0)
        self.assertEqual(fx.demoted, [])

    def test_threshold_reached_but_unbacked_leaf_is_attributed_to_backup_miss(self):
        """Enough hits to trigger a write, but no host copy landed."""
        fx = self.fixture()
        node = fx.leaf(
            "a", num_tokens=4, hit_count=WRITE_THROUGH_THRESHOLD, backuped=False
        )
        self.assertFalse(node.backuped)

        self.assertEqual(fx.tree._evict_write_through(4), 4)

        self.assertEqual(fx.metrics.dropped, [(4, "never_backuped", "kv")])
        self.assertEqual(fx.metrics.backed_up, [])
        self.assertEqual(fx.allocator.owned, set())
        self.assertEqual(fx.demoted, [])

    def test_extra_hits_beyond_threshold_still_count_as_backup_miss(self):
        fx = self.fixture()
        fx.leaf(
            "a", num_tokens=2, hit_count=WRITE_THROUGH_THRESHOLD + 5, backuped=False
        )
        self.assertEqual(fx.tree._evict_write_through(2), 2)
        self.assertEqual(fx.metrics.dropped, [(2, "never_backuped", "kv")])

    def test_backuped_leaf_is_demoted_and_never_counted_as_dropped(self):
        """_evict_backuped releases device slots but the host copy survives."""
        fx = self.fixture()
        node = fx.leaf("a", num_tokens=4, hit_count=1, backuped=True)
        device_slots = list(node.value)

        self.assertEqual(fx.tree._evict_write_through(4), 4)

        # Demotion, not loss: nothing is charged to the dropped counter.
        self.assertEqual(fx.metrics.dropped, [])
        self.assertEqual(fx.metrics.backed_up, [(4, "kv")])
        self.assertEqual([list(slots) for slots in fx.demoted], [device_slots])
        self.assertIsNone(node.value)
        self.assertIsNotNone(node.host_value)
        # The device slots went to evict_device, not through the drop free().
        self.assertEqual(fx.allocator.freed, [])
        self.assertEqual(fx.allocator.owned, set(device_slots))
        self.assertEqual(fx.tree.evictable_size_, 0)

    def test_hit_count_is_read_before_the_node_leaves_the_tree(self):
        """Attribution must not depend on post-detach node state."""
        fx = self.fixture()
        node = fx.leaf("a", num_tokens=2, hit_count=1, backuped=False)
        observed = []
        original = fx.tree._delete_leaf

        def spy(leaf):
            observed.append((leaf.hit_count, list(fx.metrics.dropped)))
            return original(leaf)

        fx.tree._delete_leaf = spy
        fx.tree._evict_write_through(2)

        self.assertEqual(len(observed), 1)
        hit_count_at_delete, dropped_at_delete = observed[0]
        self.assertEqual(hit_count_at_delete, 1)
        # Already attributed by the time the node is unlinked.
        self.assertEqual(dropped_at_delete, [(2, "below_write_threshold", "kv")])
        self.assertIs(node.parent, fx.tree.root_node)

    def test_mixed_sweep_partitions_evicted_tokens_across_the_three_exits(self):
        """dropped(below) + dropped(never) + backup ~= evicted, exactly here.

        The identity that validates instrumentation coverage: every device token this loop
        releases leaves through exactly one accounted exit. A deficit would
        mean tokens vanish through an un-instrumented path.
        """
        fx = self.fixture()
        fx.leaf("a", num_tokens=4, hit_count=1, backuped=False)
        fx.leaf("b", num_tokens=2, hit_count=WRITE_THROUGH_THRESHOLD, backuped=False)
        demoted_leaf = fx.leaf("c", num_tokens=8, hit_count=3, backuped=True)
        demoted_slots = list(demoted_leaf.value)
        fx.leaf("d", num_tokens=6, hit_count=0, backuped=False)

        evicted = fx.tree._evict_write_through(20)

        self.assertEqual(evicted, 20)
        self.assertEqual(
            fx.metrics.tokens_by_reason(),
            {"below_write_threshold": 10, "never_backuped": 2},
        )
        backed_up = sum(num for num, _pool in fx.metrics.backed_up)
        self.assertEqual(backed_up, 8)
        dropped = sum(fx.metrics.tokens_by_reason().values())
        self.assertEqual(dropped + backed_up, evicted)
        # Only the dropped tokens were destroyed; the backed-up one was demoted
        # to evict_device and its device slots are recycled, not leaked.
        self.assertEqual(len(fx.allocator.freed), dropped)
        self.assertEqual(fx.allocator.owned, set(demoted_slots))
        self.assertEqual([list(slots) for slots in fx.demoted], [demoted_slots])
        self.assertEqual(fx.tree.evictable_size_, 0)
        # Dropped leaves left the tree; the demoted one stays, host-resident
        # and loadable back -- that asymmetry is exactly why it is not a drop.
        self.assertEqual(list(fx.tree.root_node.children), ["c"])
        self.assertTrue(fx.tree.root_node.children["c"].evicted)
        self.assertTrue(fx.tree.root_node.children["c"].backuped)
        # No device-resident leaf survives the sweep. The root itself may be
        # re-added by _update_leaf_status(parent); production skips it via its
        # own root_node guards, and this instrumentation does not change that.
        self.assertEqual(fx.tree.evictable_leaves - {fx.tree.root_node}, set())

    def test_every_dropped_reason_carries_a_known_pool_label(self):
        fx = self.fixture()
        fx.leaf("a", num_tokens=2, hit_count=1, backuped=False)
        fx.leaf("b", num_tokens=2, hit_count=WRITE_THROUGH_THRESHOLD, backuped=False)
        fx.tree._evict_write_through(4)

        pool_labels = {pool for _n, _r, pool in fx.metrics.dropped}
        self.assertEqual(pool_labels, {self.types["PoolName"].KV.value})
        self.assertEqual(
            {reason for _n, reason, _p in fx.metrics.dropped},
            {"below_write_threshold", "never_backuped"},
        )
        self.assertTrue(all(num > 0 for num, _r, _p in fx.metrics.dropped))

    def test_eviction_without_a_metrics_collector_still_drops(self):
        """Metrics are optional; the guard must not change eviction behavior."""
        fx = self.fixture(with_metrics=False)
        self.assertIsNone(fx.tree.metrics_collector)
        fx.leaf("a", num_tokens=4, hit_count=1, backuped=False)
        fx.leaf("b", num_tokens=4, hit_count=WRITE_THROUGH_THRESHOLD, backuped=False)

        self.assertEqual(fx.tree._evict_write_through(8), 8)
        self.assertEqual(fx.allocator.owned, set())
        self.assertEqual(fx.tree.root_node.children, {})


if __name__ == "__main__":
    unittest.main()
