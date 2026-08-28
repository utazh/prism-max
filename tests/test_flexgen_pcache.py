import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from contiguous_fuxian.flexgen_pcache import (
    _ExistingMemmapSink,
    _reuse_existing_memmaps,
    FlexGenPcacheConfig,
    FlexGenLayerLoader,
    FlexGenPcacheStore,
    expected_chunk_count,
    gather_prefetched_tokens,
    physical_token_ids_for_positions,
    prefetch_source_tensor_tokens,
    retained_token_ids,
    selected_tokens_for_plan,
)
from contiguous_fuxian.sparse_qwen_reprefill import PrefixStoreInfo


class FlexGenPcacheTest(unittest.TestCase):
    def test_retained_token_ids_expands_selected_chunks_and_clamps_tail(self):
        self.assertEqual(
            retained_token_ids(
                ["drop", "int8", "drop", "int4"],
                chunk_size=4,
                prefix_tokens=14,
            ),
            [4, 5, 6, 7, 12, 13],
        )

    def test_expected_chunk_count_rounds_up(self):
        self.assertEqual(expected_chunk_count(prefix_tokens=0, chunk_size=16), 0)
        self.assertEqual(expected_chunk_count(prefix_tokens=16, chunk_size=16), 1)
        self.assertEqual(expected_chunk_count(prefix_tokens=17, chunk_size=16), 2)

    def test_gather_prefetched_tokens_exact_match_is_zero_copy(self):
        import torch

        physical_key = torch.arange(12).reshape(3, 4)
        physical_value = physical_key + 100
        physical_token_ids = torch.tensor([0, 2, 3])

        gathered_key, gathered_value = gather_prefetched_tokens(
            physical_key,
            physical_value,
            physical_token_ids,
            [0, 2, 3],
        )
        self.assertIs(gathered_key, physical_key)
        self.assertIs(gathered_value, physical_value)

        sparse_key, sparse_value = gather_prefetched_tokens(
            physical_key,
            physical_value,
            physical_token_ids,
            [0, 3],
        )
        self.assertTrue(torch.equal(sparse_key, physical_key[[0, 2]]))
        self.assertTrue(torch.equal(sparse_value, physical_value[[0, 2]]))

        with self.assertRaisesRegex(RuntimeError, "omitted selected token"):
            gather_prefetched_tokens(
                physical_key, physical_value, physical_token_ids, [0, 4]
            )

    def test_physical_token_ids_expand_sparse_positions_to_complete_chunks(self):
        self.assertEqual(
            physical_token_ids_for_positions(
                [1, 5, 6, 13],
                chunk_size=4,
                prefix_tokens=14,
            ),
            [0, 1, 2, 3, 4, 5, 6, 7, 12, 13],
        )

    def test_invalid_chunk_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            retained_token_ids(["int8"], chunk_size=0, prefix_tokens=1)
        with self.assertRaises(ValueError):
            expected_chunk_count(prefix_tokens=1, chunk_size=0)

    def test_existing_memmap_sink_validates_size_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npy"
            original = bytes(range(16))
            path.write_bytes(original)
            sink = _ExistingMemmapSink(path, dtype=np.float16, shape=(8,))
            sink[:] = np.zeros((8,), dtype=np.float16)
            sink.flush()
            self.assertEqual(path.read_bytes(), original)

    def test_existing_memmap_sink_rejects_incomplete_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npy"
            path.write_bytes(b"\x00" * 14)
            with self.assertRaises(RuntimeError):
                _ExistingMemmapSink(path, dtype=np.float16, shape=(8,))

    def test_resume_memmap_context_preserves_existing_and_writes_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.npy"
            missing = root / "missing.npy"
            original = np.arange(4, dtype=np.float16)
            original.tofile(existing)
            module = type("Module", (), {"np": np})()

            with _reuse_existing_memmaps(module, allow_missing=True):
                preserved = module.np.memmap(existing, dtype=np.float16, mode="w+", shape=(4,))
                preserved[:] = np.zeros(4, dtype=np.float16)
                created = module.np.memmap(missing, dtype=np.float16, mode="w+", shape=(4,))
                created[:] = np.ones(4, dtype=np.float16)
                created.flush()
                del created

            self.assertEqual(np.fromfile(existing, dtype=np.float16).tolist(), original.tolist())
            self.assertEqual(np.fromfile(missing, dtype=np.float16).tolist(), [1.0] * 4)

    def test_reuse_and_resume_modes_are_mutually_exclusive(self):
        config = FlexGenPcacheConfig(
            flexgen_root=Path("unused"),
            kv_dir=Path("unused"),
            chunk_size=16,
            reuse_existing=True,
            resume_existing=True,
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.validate()

    def test_selector_head_ids_must_be_nonempty_and_nonnegative(self):
        for head_ids in ((), (0, -1)):
            config = FlexGenPcacheConfig(
                flexgen_root=Path("unused"),
                kv_dir=Path("unused"),
                chunk_size=16,
                selector_kv_head_ids=head_ids,
            )
            with self.assertRaisesRegex(ValueError, "selector_kv_head_ids"):
                config.validate()

    def test_online_contiguous_period_reuses_first_layer_selection(self):
        info = PrefixStoreInfo("rte", 10, 4, 128, 4, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["int4"] * 5 for _ in range(4)],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 2),
            method="contigkv",
            online_selection=True,
            keep_ratio=0.4,
        )

        loader.configure_contiguous_period(
            period_start=0,
            selected_chunks=[1, 4],
            chunk_scores=[0.1, 0.8, 0.2, 0.3, 0.7],
            period_size=4,
        )
        plans, scores = loader.online_cache_plan_and_scores()

        self.assertEqual(loader._selected_by_layer, [[2, 3, 8, 9]] * 4)
        self.assertEqual(plans[0], ["drop", "int4", "drop", "drop", "int4"])
        self.assertEqual(plans, [plans[0]] * 4)
        self.assertEqual(scores, [scores[0]] * 4)
        self.assertEqual(loader.selected_tokens_by_layer(), [[2, 3, 8, 9]] * 4)

    def test_online_contiguous_accepts_matched_selector_index(self):
        selector_task = type("SelectorTask", (), {"group_size": 32})()
        selector_index = type(
            "SelectorIndex",
            (),
            {
                "selector_kv_head_ids": (0, 1, 2, 3),
                "tasks": {"rte": selector_task},
            },
        )()
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["online", "online"]],
            config=FlexGenPcacheConfig(
                Path("unused"),
                Path("unused"),
                4,
                selector_kv_head_ids=(0, 1, 2, 3),
            ),
            method="contigkv",
            online_selection=True,
            keep_ratio=0.5,
            selector_index=selector_index,
            selector_index_task="rte",
        )
        try:
            self.assertIs(loader._selector_index, selector_index)
            self.assertEqual(loader._selector_index_group_size, 32)
        finally:
            loader.close()

    def test_task_registration_can_keep_selector_cold_until_activated(self):
        class FakeSelectorIndex:
            preloaded_compressed_bytes = 123

            def __init__(self):
                self.validated = []
                self.preloaded = []

            def validate_task(self, task, **metadata):
                self.validated.append((task, metadata))

            def preload_task(self, task):
                self.preloaded.append(task)

        class FakePcache:
            def __init__(self):
                self.inserted = []

            def insert(self, **payload):
                self.inserted.append(payload)

        selector_index = FakeSelectorIndex()
        pcache = FakePcache()
        store = FlexGenPcacheStore.__new__(FlexGenPcacheStore)
        store.config = FlexGenPcacheConfig(Path("unused"), Path("unused"), 4)
        store._selector_index = selector_index
        store._pcache = pcache
        store._pcache_module = object()
        store._task_prefix_ids = {}
        store._task_infos = {}
        store._physical_to_logical = {}
        store._logical_to_physical = {}
        store.selector_index_preload_ms = 0.0
        store.selector_index_preloaded_tasks = []
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")

        with patch(
            "contiguous_fuxian.flexgen_pcache.read_store_info",
            return_value=info,
        ), patch(
            "contiguous_fuxian.flexgen_pcache._load_task_tensor",
            return_value=(object(), object()),
        ):
            store.add_task(
                store_root="unused",
                task="rte",
                preload_selector_index=False,
            )
            self.assertEqual(
                [task for task, _ in selector_index.validated],
                ["rte"],
            )
            self.assertEqual(selector_index.preloaded, [])
            self.assertEqual(store.selector_index_preloaded_tasks, [])

            store.add_task(
                store_root="unused",
                task="rte",
                preload_selector_index=True,
            )
            store.add_task(
                store_root="unused",
                task="rte",
                preload_selector_index=True,
            )

        self.assertEqual(selector_index.preloaded, ["rte"])
        self.assertEqual(store.selector_index_preloaded_tasks, ["rte"])
        self.assertEqual(store.selector_index_preloaded_bytes, 123)
        self.assertEqual(len(pcache.inserted), 1)

    def test_online_impress_tracks_useful_tokens_separately_from_chunks(self):
        info = PrefixStoreInfo("rte", 10, 4, 128, 1, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["int4"] * 3],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
            method="impress",
            online_selection=True,
            keep_ratio=0.2,
        )

        loader.configure_impress_layer(layer=0, selected_tokens=[1, 6])

        self.assertEqual(loader._selected_by_layer[0], [1, 6])
        self.assertEqual(loader._online_layer_plan[0], ["int4", "int4", "drop"])

    def test_online_impress_uses_validated_per_layer_keep_ratios(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 3, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["online", "online"] for _ in range(3)],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
            method="impress",
            online_selection=True,
            keep_ratio=0.5,
            layer_keep_ratios=(0.25, 0.5, 0.75),
        )
        try:
            self.assertEqual(
                [loader.keep_ratio_for_layer(layer) for layer in range(3)],
                [0.25, 0.5, 0.75],
            )
        finally:
            loader.close()

        with self.assertRaisesRegex(ValueError, "cover 2 layers"):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["online", "online"] for _ in range(3)],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                layer_keep_ratios=(0.25, 0.75),
            )

    def test_exact_block_budget_repays_dense_fallback_from_later_layers(self):
        info = PrefixStoreInfo("rte", 16, 4, 128, 3, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["online"] * 4 for _ in range(3)],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
            method="impress",
            online_selection=True,
            keep_ratio=0.5,
            layer_keep_ratios=(0.5, 0.5, 0.5),
            layer_keep_blocks=(2, 2, 2),
        )
        try:
            self.assertEqual(loader.keep_blocks_for_layer(0), 2)
            self.assertEqual(loader.max_blocks_for_layer(0), 4)
            loader.configure_impress_layer(layer=0, selected_tokens=range(16))

            self.assertEqual(loader.keep_blocks_for_layer(1), 1)
            self.assertEqual(loader.max_blocks_for_layer(1), 1)
            loader.configure_impress_layer(layer=1, selected_tokens=range(4))
            self.assertEqual(loader.keep_blocks_for_layer(2), 1)
            loader.configure_impress_layer(layer=2, selected_tokens=range(4))

            self.assertEqual(loader._exact_block_budget_target, 6)
            self.assertEqual(loader._exact_block_budget_consumed, 6)
        finally:
            loader.close()

    def test_as_lru_resolves_budget_independent_full_kv(self):
        import torch

        class FakeLayer:
            chunk_num = 2
            device_map = ["disk", "disk", "disk", "disk"]

        class FakePrefix:
            layers = [FakeLayer()]

        class FakePcache:
            cache = [FakePrefix()]

            def __init__(self):
                self.calls = []

            def get(self, *, prefix_id, pos_id, layer):
                self.calls.append((prefix_id, pos_id.tolist(), layer))
                key = torch.arange(16, dtype=torch.float16).reshape(4, 2, 2)
                return key, key + 100

        pcache = FakePcache()
        info = PrefixStoreInfo("rte", 4, 2, 2, 1, "hash")
        loader = FlexGenLayerLoader(
            pcache=pcache,
            prefix_id=0,
            info=info,
            layer_plan=[["online", "online"]],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 2),
            method="as_lru",
            online_selection=True,
            keep_ratio=0.05,
        )
        try:
            loader.configure_as_full_retention()
            key, value = loader.resolve(0)
            self.assertEqual(pcache.calls, [(0, [0, 1, 2, 3], 0)])
            self.assertEqual(tuple(key.shape), (4, 2, 2))
            self.assertTrue(torch.equal(value, key + 100))
            metrics = loader.metrics()
            self.assertEqual(metrics["effective_mean_keep_ratio"], 1.0)
            self.assertEqual(metrics["selected_kv_bytes"], 64)
        finally:
            loader.close()

    def test_as_h2o_loads_full_keys_and_only_selected_values(self):
        import torch

        class FakeLayer:
            chunk_num = 2
            device_map = ["disk", "disk", "disk", "disk"]

        class FakePrefix:
            layers = [FakeLayer()]

        class FakePcache:
            cache = [FakePrefix()]

            def __init__(self):
                self.key_calls = []
                self.value_calls = []
                self.full_keys = torch.arange(
                    16, dtype=torch.float16
                ).reshape(4, 2, 2)

            def get_key(self, *, prefix_id, pos_id, layer):
                self.key_calls.append((prefix_id, pos_id, layer))
                return self.full_keys

            def get_value(self, *, prefix_id, pos_id, layer):
                self.value_calls.append((prefix_id, pos_id.clone(), layer))
                result = torch.empty((pos_id.shape[1], 2, 2), dtype=torch.float16)
                for head in range(2):
                    for row, token in enumerate(pos_id[head].tolist()):
                        result[row, head] = torch.tensor(
                            [token * 10 + head, token * 10 + head + 0.5],
                            dtype=torch.float16,
                        )
                return result

        pcache = FakePcache()
        info = PrefixStoreInfo("rte", 4, 2, 2, 1, "hash")
        loader = FlexGenLayerLoader(
            pcache=pcache,
            prefix_id=0,
            info=info,
            layer_plan=[["online", "online"]],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 2),
            method="as_h2o_lru",
            online_selection=True,
            keep_ratio=0.5,
        )
        positions = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        try:
            loaded_keys = loader.load_selector_keys(0)
            self.assertIs(loaded_keys, pcache.full_keys)
            loader.configure_as_h2o_layer(layer=0, positions=positions)
            key, value = loader.resolve(0)
            self.assertIs(key, pcache.full_keys)
            self.assertEqual(pcache.key_calls, [(0, None, 0)])
            self.assertTrue(torch.equal(pcache.value_calls[0][1], positions))
            self.assertTrue(torch.equal(value[1, 0], torch.zeros(2)))
            self.assertTrue(torch.equal(value[0, 1], torch.zeros(2)))
            self.assertTrue(torch.equal(value[0, 0], torch.tensor([0.0, 0.5])))
            self.assertTrue(torch.equal(value[3, 1], torch.tensor([31.0, 31.5])))
            metrics = loader.metrics()
            self.assertEqual(metrics["as_h2o_full_key_ratio"], 1.0)
            self.assertEqual(metrics["as_h2o_value_keep_ratio"], 0.5)
            self.assertEqual(metrics["as_h2o_total_logical_payload_ratio"], 0.75)
            self.assertEqual(metrics["selected_kv_bytes"], 48)
        finally:
            loader.close()

    def test_paper_impress_online_mode_uses_synchronous_get(self):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.method = "impress"
        loader.online_selection = True
        loader.impress_async_prefetch = False
        loader._loaded = {}
        calls = []
        loader._resolve_impress = lambda layer: calls.append(layer) or ("key", "value")

        self.assertEqual(loader.resolve(3), ("key", "value"))
        self.assertEqual(calls, [3])

    def test_impress_logical_positions_map_to_ranked_physical_storage(self):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._logical_to_physical = [[2, 0, 3, 1]]

        self.assertEqual(loader._storage_positions(0, [0, 1, 3]), [2, 0, 1])

    def test_reordered_physical_segment_restores_logical_tensor_order(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local unit-test runtime")
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._physical_to_logical = [[2, 0, 3, 1]]
        key = torch.tensor([[10], [11], [12], [13]])
        value = torch.tensor([[20], [21], [22], [23]])
        physical_ids = torch.tensor([0, 1, 2, 3])

        logical_key, logical_value, logical_ids = loader._logical_segment(
            0, key, value, physical_ids
        )

        self.assertEqual(logical_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(logical_key[:, 0].tolist(), [11, 13, 10, 12])
        self.assertEqual(logical_value[:, 0].tolist(), [21, 23, 20, 22])

    def test_hyperinfer_prefetch_requires_online_impress(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        with self.assertRaisesRegex(ValueError, "only valid for online IMPRESS"):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="contigkv",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=True,
            )

    def test_priority_prefetch_requires_priority_enabled_pcache(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        with self.assertRaisesRegex(ValueError, "priority-enabled Pcache"):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=True,
                impress_priority_prefetch=True,
            )

    def test_deferred_compute_timing_requires_async_online_impress(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        with self.assertRaisesRegex(
            ValueError,
            "deferred compute timing requires asynchronous online IMPRESS",
        ):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=False,
                impress_deferred_compute_timing=True,
            )

    def test_rolling_period_prefetch_requires_predictive_async_mode(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        with self.assertRaisesRegex(
            ValueError,
            "rolling Period prefetch requires predictive",
        ):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=True,
                impress_period_prefetch_size=1,
                impress_rolling_period_prefetch=True,
            )

    def test_value_ordered_prefetch_requires_async_online_impress(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        with self.assertRaisesRegex(
            ValueError,
            "value-ordered prefetch requires asynchronous online IMPRESS",
        ):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=False,
                impress_value_ordered_prefetch=True,
            )

    def test_value_prefetch_budget_scale_requires_value_ordering(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")
        for scale in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError,
                r"budget scale must be in \(0, 1\]",
            ):
                FlexGenLayerLoader(
                    pcache=object(),
                    prefix_id=0,
                    info=info,
                    layer_plan=[["int4", "int4"]],
                    config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                    method="impress",
                    online_selection=True,
                    keep_ratio=0.5,
                    impress_async_prefetch=True,
                    impress_value_ordered_prefetch=True,
                    impress_value_prefetch_budget_scale=scale,
                )
        with self.assertRaisesRegex(
            ValueError,
            "budget scaling requires value-ordered prefetch",
        ):
            FlexGenLayerLoader(
                pcache=object(),
                prefix_id=0,
                info=info,
                layer_plan=[["int4", "int4"]],
                config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
                method="impress",
                online_selection=True,
                keep_ratio=0.5,
                impress_async_prefetch=True,
                impress_value_prefetch_budget_scale=0.75,
            )

    def test_value_ordered_prefetch_changes_io_order_not_selected_set(self):
        info = PrefixStoreInfo("rte", 8, 4, 128, 2, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["int4", "int4"] for _ in range(2)],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
            method="impress",
            online_selection=True,
            keep_ratio=0.5,
            impress_async_prefetch=True,
            impress_value_ordered_prefetch=True,
            impress_value_prefetch_budget_scale=0.5,
        )
        submissions = []

        def submit(layer, positions, pending, **kwargs):
            submissions.append((layer, list(positions), kwargs))
            pending[layer] = (object(), 0.0)

        loader._submit_positions = submit
        loader._impress_prefetch_time_budget = 0.02
        try:
            loader.configure_impress_layer(
                layer=0,
                selected_tokens=[0, 1, 4, 5],
                prefetch_priority_tokens=[4, 5, 0, 1],
            )
            loader.schedule_impress_next(0)

            self.assertEqual(loader._selected_by_layer[0], [0, 1, 4, 5])
            self.assertEqual(loader._prefetch_priority_by_layer[0], [4, 5, 0, 1])
            self.assertEqual(submissions[0][1], [4, 5, 0, 1])
            self.assertAlmostEqual(submissions[0][2]["time_budget"], 0.01)
            self.assertEqual(loader._impress_value_ordered_prefetch_jobs, 1)
        finally:
            loader.close()

    def test_unreordered_physical_segment_is_sorted_for_sparse_gather(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local unit-test runtime")
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._physical_to_logical = None
        key = torch.tensor([[20], [21], [10], [11]])
        value = torch.tensor([[40], [41], [30], [31]])
        physical_ids = torch.tensor([4, 5, 0, 1])

        logical_key, logical_value, logical_ids = loader._logical_segment(
            0, key, value, physical_ids
        )

        self.assertEqual(logical_ids.tolist(), [0, 1, 4, 5])
        self.assertEqual(logical_key[:, 0].tolist(), [10, 11, 20, 21])
        self.assertEqual(logical_value[:, 0].tolist(), [30, 31, 40, 41])

    def test_deferred_compute_timing_harvests_without_forced_wait(self):
        class StartedEvent:
            def elapsed_time(self, finished):
                return finished.elapsed_ms

        class FinishedEvent:
            def __init__(self, elapsed_ms):
                self.elapsed_ms = elapsed_ms
                self.ready = False
                self.synchronize_calls = 0

            def query(self):
                return self.ready

            def synchronize(self):
                self.synchronize_calls += 1
                self.ready = True

        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.impress_deferred_compute_timing = True
        loader.online_selection = True
        loader.method = "impress"
        loader.impress_async_prefetch = True
        loader._last_selector_compute_ms = 5.0
        loader._impress_prefetch_time_budget = 0.0
        loader._impress_compute_event_pairs = []
        loader._impress_deferred_compute_samples = 0
        loader._impress_deferred_compute_pending_max = 0
        started = StartedEvent()
        finished = FinishedEvent(15.0)

        loader.defer_impress_layer_compute(started, finished)
        self.assertEqual(loader.resolve_deferred_impress_compute(), 0)
        self.assertEqual(finished.synchronize_calls, 0)

        finished.ready = True
        self.assertEqual(loader.resolve_deferred_impress_compute(), 1)
        self.assertEqual(finished.synchronize_calls, 0)
        self.assertEqual(loader._impress_prefetch_time_budget, 0.02)
        self.assertEqual(loader._impress_deferred_compute_samples, 1)
        self.assertEqual(loader._impress_deferred_compute_pending_max, 1)
        self.assertEqual(loader._impress_compute_event_pairs, [])

    def test_deferred_compute_timing_can_drain_on_close_path(self):
        class StartedEvent:
            def elapsed_time(self, finished):
                return 10.0

        class FinishedEvent:
            def __init__(self):
                self.synchronize_calls = 0

            def synchronize(self):
                self.synchronize_calls += 1

        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.impress_deferred_compute_timing = True
        loader.online_selection = True
        loader.method = "impress"
        loader.impress_async_prefetch = True
        loader._last_selector_compute_ms = 2.0
        loader._impress_prefetch_time_budget = 0.0
        loader._impress_compute_event_pairs = []
        loader._impress_deferred_compute_samples = 0
        loader._impress_deferred_compute_pending_max = 0
        started = StartedEvent()
        finished = FinishedEvent()

        loader.defer_impress_layer_compute(started, finished)
        self.assertEqual(loader.resolve_deferred_impress_compute(wait=True), 1)
        self.assertEqual(finished.synchronize_calls, 1)
        self.assertEqual(loader._impress_prefetch_time_budget, 0.012)

    def test_priority_prefetch_labels_and_orders_scheduler_work(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local unit-test runtime")

        class FakePcache:
            def __init__(self):
                self.calls = []

            def prefetch_async(self, **kwargs):
                self.calls.append(kwargs)
                return object()

        pcache = FakePcache()
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._pcache = pcache
        loader._prefix_id = 3
        loader._info = type("Info", (), {"prefix_tokens": 8})()
        loader._config = FlexGenPcacheConfig(Path("unused"), Path("unused"), 4)
        loader._storage_positions = lambda layer, positions: list(positions)
        loader.impress_priority_prefetch = True
        pending = {}

        for kind in ("current", "next", "period"):
            loader._submit_positions(
                0,
                [1, 2],
                pending,
                prefetch_kind=kind,
            )

        self.assertEqual(
            [(call["prefetch_kind"], call["priority"]) for call in pcache.calls],
            [("current", 0), ("next", 1), ("period", 2)],
        )
        self.assertTrue(
            all(torch.equal(call["pos_id"], torch.tensor([1, 2])) for call in pcache.calls)
        )

    def test_fifo_prefetch_keeps_kind_labels_without_priority_override(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local unit-test runtime")

        class FakePcache:
            def __init__(self):
                self.call = None

            def prefetch_async(self, **kwargs):
                self.call = kwargs
                return object()

        pcache = FakePcache()
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._pcache = pcache
        loader._prefix_id = 0
        loader._info = type("Info", (), {"prefix_tokens": 8})()
        loader._config = FlexGenPcacheConfig(Path("unused"), Path("unused"), 4)
        loader._storage_positions = lambda layer, positions: list(positions)
        loader.impress_priority_prefetch = False

        loader._submit_positions(
            0,
            [1],
            {},
            prefetch_kind="period",
        )

        self.assertEqual(pcache.call["prefetch_kind"], "period")
        self.assertNotIn("priority", pcache.call)

    def test_scheduler_metrics_are_scoped_to_one_request(self):
        class FakePcache:
            def prefetch_scheduler_metrics(self):
                return {
                    "current": {
                        "submitted": 5,
                        "started": 5,
                        "completed": 5,
                        "failed": 0,
                        "cancelled": 0,
                        "queue_wait_ms": 12.5,
                        "execution_ms": 20.0,
                    },
                    "period": {
                        "submitted": 3,
                        "started": 3,
                        "completed": 3,
                        "failed": 0,
                        "cancelled": 0,
                        "queue_wait_ms": 9.0,
                        "execution_ms": 30.0,
                    },
                }

        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._pcache = FakePcache()
        loader._prefetch_scheduler_start = {
            "current": {
                "submitted": 2,
                "started": 2,
                "completed": 2,
                "failed": 0,
                "cancelled": 0,
                "queue_wait_ms": 5.0,
                "execution_ms": 8.0,
            }
        }

        metrics = loader._prefetch_scheduler_metrics_delta()

        self.assertEqual(metrics["prefetch_scheduler_current_submitted"], 3)
        self.assertEqual(metrics["prefetch_scheduler_period_submitted"], 3)
        self.assertEqual(metrics["prefetch_scheduler_total_submitted"], 6)
        self.assertEqual(metrics["prefetch_scheduler_current_queue_wait_ms"], 7.5)
        self.assertEqual(metrics["prefetch_scheduler_total_execution_ms"], 42.0)

    def test_online_impress_prefetches_next_layer_and_only_loads_uncovered_chunks(self):
        info = PrefixStoreInfo("rte", 12, 4, 128, 2, "hash")
        loader = FlexGenLayerLoader(
            pcache=object(),
            prefix_id=0,
            info=info,
            layer_plan=[["int4"] * 3 for _ in range(2)],
            config=FlexGenPcacheConfig(Path("unused"), Path("unused"), 4),
            method="impress",
            online_selection=True,
            keep_ratio=0.2,
            impress_async_prefetch=True,
        )
        submissions = []

        def submit(layer, positions, pending, **kwargs):
            submissions.append((layer, list(positions), pending, kwargs))
            pending[layer] = (object(), 0.0)

        loader._submit_positions = submit
        loader.configure_impress_layer(layer=0, selected_tokens=[1, 6])
        loader.schedule_impress_next(0)
        loader._resolved_speculative[1] = (object(), object(), object())
        loader._speculative_positions[1] = [0, 1, 2, 3]
        loader.configure_impress_layer(layer=1, selected_tokens=[2, 6, 9])
        loader.schedule_impress_missing(1)

        self.assertEqual(submissions[0][:2], (1, [1, 6]))
        self.assertEqual(
            submissions[0][3],
            {"time_budget": 0.0, "prefetch_kind": "next"},
        )
        self.assertEqual(submissions[1][:2], (1, [6, 9]))
        self.assertEqual(submissions[1][3], {"prefetch_kind": "current"})

    def test_online_impress_resolves_actual_prefetched_physical_tokens(self):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.online_selection = True
        loader.method = "impress"
        loader.impress_async_prefetch = True
        loader._speculative_pending = {1: ("handle", 1.0)}
        loader._resolved_speculative = {}
        loader._speculative_positions = {1: [8, 9]}
        segment = ("key", "value", np.array([0, 1, 2, 3]))
        loader._resolve_entry = lambda layer, entry: segment

        loader.resolve_impress_speculation(1)

        self.assertEqual(loader._resolved_speculative[1], segment)
        self.assertEqual(loader._speculative_positions[1], [0, 1, 2, 3])

    def test_online_impress_budget_uses_selector_plus_layer_compute_time(self):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.online_selection = True
        loader.method = "impress"
        loader.impress_async_prefetch = True
        loader._last_selector_compute_ms = 12.5
        loader._impress_prefetch_time_budget = 0.0

        loader.record_impress_layer_compute(37.5)

        self.assertEqual(loader._impress_prefetch_time_budget, 0.05)

    def test_prime_period_queues_every_layer_before_resolving_subperiod(self):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._info = type("Info", (), {"layers": 8})()
        calls = []
        loader.schedule_range = lambda start, end: calls.append(("schedule", start, end))
        loader.resolve = lambda layer: calls.append(("resolve", layer))

        loader.prime_period(0, subperiod_size=4, period_size=8)

        self.assertEqual(calls[0], ("schedule", 0, 8))
        self.assertEqual(calls[1:], [("resolve", layer) for layer in range(4)])

    def test_online_contiguous_cache_score_updates_once_per_loaded_layer(self):
        calls = []
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.method = "contigkv"
        loader._config = FlexGenPcacheConfig(
            Path("unused"), Path("unused"), chunk_size=4, cache_type="CKLFU"
        )
        loader._selected_by_layer = [[0, 1, 2, 3, 8, 9]]
        loader._online_chunk_scores = [[0.25, 0.5, 0.75]]
        loader._cache_score_updater = lambda layer, selected, scores: (
            calls.append((layer, list(selected), list(scores))) or len(selected)
        )
        loader._cache_score_updated_layers = set()
        loader._cache_score_updates = 0
        loader._cache_update_ms = 0.0

        loader._update_loaded_layer_score(0)
        loader._update_loaded_layer_score(0)

        self.assertEqual(calls, [(0, [0, 2], [0.25, 0.5, 0.75])])
        self.assertEqual(loader._cache_score_updated_layers, {0})
        self.assertEqual(loader._cache_score_updates, 2)
        self.assertGreaterEqual(loader._cache_update_ms, 0.0)

    def test_online_impress_cache_score_uses_selected_logical_tokens(self):
        calls = []
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader.method = "impress"
        loader._selected_by_layer = [[1, 5, 9]]
        loader._cache_score_updater = lambda layer, selected, scores: (
            calls.append((layer, list(selected), scores)) or 3
        )
        loader._cache_score_updated_layers = set()
        loader._cache_score_updates = 0
        loader._cache_update_ms = 0.0

        loader._update_loaded_layer_score(0)

        self.assertEqual(calls, [(0, [1, 5, 9], None)])
        self.assertEqual(loader._cache_score_updates, 3)

    def test_explicit_useful_tokens_are_separate_from_physical_chunks(self):
        selected = selected_tokens_for_plan(
            [["int8", "int8"], ["int8", "drop"]],
            chunk_size=4,
            prefix_tokens=8,
            layer_token_selections=[[1, 5], [0]],
        )

        self.assertEqual(selected, [[1, 5], [0]])

    def test_explicit_useful_token_cannot_belong_to_dropped_chunk(self):
        with self.assertRaises(ValueError):
            selected_tokens_for_plan(
                [["int8", "drop"]],
                chunk_size=4,
                prefix_tokens=8,
                layer_token_selections=[[5]],
            )

    def test_cklfu_updates_key_and_value_with_cumulative_attention_times_frequency(self):
        class Control:
            def __init__(self):
                self.calls = []

            def update_score(self, *, item, new_score):
                self.calls.append((item, list(new_score)))

        class Layer:
            key_tokens = ["k0", "k1"]
            value_tokens = ["v0", "v1"]

        class Prefix:
            layers = [Layer()]

        control = Control()
        store = FlexGenPcacheStore.__new__(FlexGenPcacheStore)
        store.config = FlexGenPcacheConfig(
            flexgen_root=Path("unused"),
            kv_dir=Path("unused"),
            chunk_size=16,
            cache_type="CKLFU",
        )
        store._pcache = type("Pcache", (), {"cache": [Prefix()], "control": control})()
        store._task_prefix_ids = {"rte": 0}
        store._attention_state = {}
        store.cache_score_updates = 0

        first = store.update_attention_scores(
            task="rte",
            layer_plan=[["drop", "int8"]],
            layer_chunk_scores=[[0.1, 0.25]],
        )
        second = store.update_attention_scores(
            task="rte",
            layer_plan=[["drop", "int8"]],
            layer_chunk_scores=[[0.1, 0.25]],
        )

        self.assertEqual((first, second), (1, 1))
        self.assertEqual(control.calls[:2], [("k1", [0.25, 0.25]), ("v1", [0.25, 0.25])])
        self.assertEqual(control.calls[-2:], [("k1", [1.0, 1.0]), ("v1", [1.0, 1.0])])
        self.assertEqual(store.cache_score_updates, 2)

    def test_cklfu_rejects_non_finite_attention_scores(self):
        class Control:
            def update_score(self, *, item, new_score):
                raise AssertionError("invalid scores must be rejected before updating the heap")

        class Layer:
            key_tokens = ["k0"]
            value_tokens = ["v0"]

        class Prefix:
            layers = [Layer()]

        store = FlexGenPcacheStore.__new__(FlexGenPcacheStore)
        store.config = FlexGenPcacheConfig(
            flexgen_root=Path("unused"),
            kv_dir=Path("unused"),
            chunk_size=16,
            cache_type="CKLFU",
        )
        store._pcache = type("Pcache", (), {"cache": [Prefix()], "control": Control()})()
        store._task_prefix_ids = {"rte": 0}
        store._attention_state = {}
        store.cache_score_updates = 0

        with self.assertRaisesRegex(ValueError, "invalid attention score"):
            store.update_attention_scores(
                task="rte",
                layer_plan=[["int4"]],
                layer_chunk_scores=[[float("nan")]],
            )

    def test_impress_cklfu_counts_accesses_and_selected_tokens_per_chunk(self):
        class Control:
            def __init__(self):
                self.calls = []

            def update_score(self, *, item, new_score):
                self.calls.append((item, list(new_score)))

        class Layer:
            def __init__(self, layer):
                self.key_tokens = [f"k{layer}-{chunk}" for chunk in range(3)]
                self.value_tokens = [f"v{layer}-{chunk}" for chunk in range(3)]

        class Prefix:
            layers = [Layer(0), Layer(1)]

        control = Control()
        store = FlexGenPcacheStore.__new__(FlexGenPcacheStore)
        store.config = FlexGenPcacheConfig(
            flexgen_root=Path("unused"),
            kv_dir=Path("unused"),
            chunk_size=4,
            cache_type="CKLFU",
        )
        store._pcache = type("Pcache", (), {"cache": [Prefix()], "control": control})()
        store._task_prefix_ids = {"rte": 0}
        store._task_infos = {"rte": PrefixStoreInfo("rte", 10, 4, 128, 2, "hash")}
        store._impress_score_state = {}
        store.cache_score_updates = 0

        first = store.update_impress_scores(
            task="rte",
            layer_token_selections=[[0, 1, 5], [9]],
        )
        second = store.update_impress_scores(
            task="rte",
            layer_token_selections=[[0, 1, 5], [9]],
        )

        self.assertEqual((first, second), (3, 3))
        self.assertEqual(
            control.calls[:6],
            [
                ("k0-0", [1, 2]),
                ("v0-0", [1, 2]),
                ("k0-1", [1, 1]),
                ("v0-1", [1, 1]),
                ("k1-2", [1, 1]),
                ("v1-2", [1, 1]),
            ],
        )
        self.assertEqual(control.calls[-2:], [("k1-2", [2, 2]), ("v1-2", [2, 2])])
        self.assertEqual(store.cache_score_updates, 6)

    def test_impress_cklfu_scores_reordered_physical_chunks(self):
        class Control:
            def __init__(self):
                self.calls = []

            def update_score(self, *, item, new_score):
                self.calls.append((item, list(new_score)))

        class Layer:
            key_tokens = ["k0", "k1"]
            value_tokens = ["v0", "v1"]

        control = Control()
        store = FlexGenPcacheStore.__new__(FlexGenPcacheStore)
        store.config = FlexGenPcacheConfig(
            Path("unused"), Path("unused"), 4, cache_type="CKLFU"
        )
        store._pcache = type(
            "Pcache",
            (),
            {"cache": [type("Prefix", (), {"layers": [Layer()]})()], "control": control},
        )()
        store._task_prefix_ids = {"rte": 0}
        store._task_infos = {"rte": PrefixStoreInfo("rte", 8, 4, 128, 1, "hash")}
        # Logical tokens 0 and 4 are adjacent after importance reordering.
        store._logical_to_physical = {"rte": [[0, 2, 3, 4, 1, 5, 6, 7]]}
        store._impress_score_state = {}
        store.cache_score_updates = 0

        updates = store.update_impress_scores(
            task="rte", layer_token_selections=[[0, 4]]
        )

        self.assertEqual(updates, 1)
        self.assertEqual(control.calls, [("k0", [1, 2]), ("v0", [1, 2])])

    def test_prefetch_source_counts_key_and_value_chunk_tiers(self):
        layer = type(
            "Layer",
            (),
            {
                "chunk_num": 2,
                "device_map": np.array(["cuda:0", "cpu", "disk", "cpu"]),
            },
        )()
        prefix = type("Prefix", (), {"layers": [layer]})()
        pcache = type("Pcache", (), {"cache": [prefix]})()

        counts = prefetch_source_tensor_tokens(
            pcache,
            prefix_id=0,
            layer=0,
            token_ids=[0, 1, 2, 3, 4, 5, 6],
            chunk_size=4,
            prefix_tokens=7,
        )

        self.assertEqual(counts, {"gpu": 4, "cpu": 6, "disk": 4})


if __name__ == "__main__":
    unittest.main()
