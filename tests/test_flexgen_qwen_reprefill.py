import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contiguous_fuxian.flexgen_qwen_reprefill import (
    _load_layer_chunk_scores,
    _load_optional_layer_plan,
    _load_plan_metadata,
    allocate_exact_layer_blocks,
    build_online_layer_plan,
    configure_online_layer_selection,
    impress_contiguous_block_selection,
    impress_contiguous_block_selection_with_ranking,
    layer_token_selection_sha256,
    layer_selection_agreement,
    parse_head_ids,
    prepare_impress_contiguous_block_scores,
    prepare_impress_block_scores,
    resolve_store_tasks,
    runtime_variant,
    select_prepared_impress_blocks,
    selector_key_slots,
    validate_impress_block_mode,
    validate_online_selector_mapping,
    validate_plan_keep_ratio,
)


class FlexGenQwenReprefillTest(unittest.TestCase):
    def test_store_task_order_preserves_numeric_prefix_ids(self):
        self.assertEqual(
            resolve_store_tasks(
                ("rte",),
                ("sst2", "subj", "trec", "rte"),
            ),
            ("sst2", "subj", "trec", "rte"),
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            resolve_store_tasks(("rte",), ("sst2", "subj"))
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_store_tasks(("sst2",), ("sst2", "sst2"))

    def test_layer_token_selection_hash_is_canonical_and_order_sensitive(self):
        self.assertEqual(
            layer_token_selection_sha256([[0, 2], [1]]),
            "bb8bfd012cead4feda132268c59b45440da98e238ce0cb1e182e708cdaa52495",
        )
        self.assertNotEqual(
            layer_token_selection_sha256([[0, 2], [1]]),
            layer_token_selection_sha256([[2, 0], [1]]),
        )

    def test_runtime_variant_labels_paper_and_extension_modes(self):
        cases = [
            ("contigkv", True, False, False, "contiguouskv-online-period-prefetch"),
            ("impress", True, False, True, "paper-impress-sync-reorder"),
            ("impress", True, False, False, "impress-sync-no-reorder-ablation"),
            ("impress", True, True, True, "hyperinfer-async-reorder"),
            ("impress", True, True, False, "hyperinfer-async-no-reorder"),
            ("as_lru", True, False, False, "attentionstore-full-kv-lru-c64"),
            (
                "as_h2o_lru",
                True,
                False,
                False,
                "attentionstore-h2o-full-k-selector-compact-kv-lru-c64",
            ),
            ("contigkv", False, False, False, "offline-plan"),
            ("impress", False, True, True, "offline-plan"),
        ]
        for method, online, asynchronous, reordered, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    runtime_variant(
                        method=method,
                        online_selection=online,
                        impress_async_prefetch=asynchronous,
                        impress_reorder_enabled=reordered,
                    ),
                    expected,
                )

        self.assertEqual(
            runtime_variant(
                method="impress",
                online_selection=True,
                impress_async_prefetch=False,
                impress_reorder_enabled=False,
                impress_selection_block_size=16,
            ),
            "impress-sync-contiguous-blocks-c16",
        )
        self.assertEqual(
            runtime_variant(
                method="impress",
                online_selection=True,
                impress_async_prefetch=True,
                impress_reorder_enabled=False,
                impress_selection_block_size=16,
            ),
            "hyperinfer-async-contiguous-blocks-c16",
        )
        self.assertEqual(
            runtime_variant(
                method="impress",
                online_selection=True,
                impress_async_prefetch=True,
                impress_reorder_enabled=False,
                impress_selection_block_size=16,
                layer_budget_profile_enabled=True,
                impress_period_prefetch_size=8,
                impress_period_prefetch_budget_scale=0.25,
                impress_priority_prefetch=True,
                impress_deferred_compute_timing=True,
                impress_rolling_period_prefetch=True,
                impress_value_ordered_prefetch=True,
                impress_value_prefetch_budget_scale=0.75,
            ),
            (
                "hyperinfer-async-contiguous-blocks-c16"
                "+layer-budget+predictive-period-p8-budget-s0.25+priority-prefetch"
                "+deferred-compute-timing+rolling-period-source"
                "+value-ordered-prefetch+value-budget-s0.75"
            ),
        )
        self.assertEqual(
            runtime_variant(
                method="impress",
                online_selection=True,
                impress_async_prefetch=True,
                impress_reorder_enabled=False,
                impress_selection_block_size=16,
                layer_budget_profile_enabled=True,
                exact_layer_block_budget=True,
                impress_selection_period_size=8,
                impress_known_period_prefetch=True,
            ),
            (
                "hyperinfer-async-contiguous-blocks-c16"
                "+layer-budget+exact-total-block-budget"
                "+periodic-selection-p8+known-period-prefetch"
            ),
        )

    def test_runtime_variant_rejects_unknown_online_method(self):
        with self.assertRaisesRegex(ValueError, "unsupported online runtime method"):
            runtime_variant(
                method="unknown",
                online_selection=True,
                impress_async_prefetch=False,
                impress_reorder_enabled=False,
            )

    def test_online_plan_is_shape_only_and_does_not_require_request_uid(self):
        plan = build_online_layer_plan(layers=2, prefix_tokens=65, chunk_size=64)

        self.assertEqual(plan, [["online", "online"], ["online", "online"]])

    def test_online_plan_rejects_empty_model_or_prefix(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_online_layer_plan(layers=0, prefix_tokens=65, chunk_size=64)
        with self.assertRaisesRegex(ValueError, "non-empty prefix"):
            build_online_layer_plan(layers=2, prefix_tokens=0, chunk_size=64)

    def test_online_metadata_load_does_not_require_a_request_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"metadata": {"method": "contigkv"}}), encoding="utf-8")

            self.assertEqual(_load_plan_metadata(path)["method"], "contigkv")
            self.assertIsNone(_load_optional_layer_plan(path, "new-request"))

    def test_layer_chunk_scores_reject_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps(
                    {"layer_chunk_attention_scores": {"cmpl-x-score": [[0.5, float("nan")]]}}
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "layer 0 chunk 1"):
                _load_layer_chunk_scores(path, "x")

    def test_plan_keep_ratio_matches_explicit_expectation(self):
        self.assertEqual(validate_plan_keep_ratio({"keep_ratio": 0.05}, 0.05), 0.05)

    def test_plan_keep_ratio_rejects_filename_scale_mistake(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_plan_keep_ratio({"keep_ratio": 0.005}, 0.05)

    def test_plan_keep_ratio_is_required_and_fractional(self):
        with self.assertRaisesRegex(ValueError, "must declare"):
            validate_plan_keep_ratio({})
        with self.assertRaisesRegex(ValueError, "must be in"):
            validate_plan_keep_ratio({"keep_ratio": 5.0})

    def test_parse_head_ids_rejects_empty_or_negative_values(self):
        self.assertEqual(parse_head_ids("0, 1,2"), (0, 1, 2))
        with self.assertRaisesRegex(ValueError, "one or more"):
            parse_head_ids("")
        with self.assertRaisesRegex(ValueError, "one or more"):
            parse_head_ids("0,-1")

    def test_online_selector_mapping_handles_qwen_gqa(self):
        config = type(
            "Config",
            (),
            {"num_attention_heads": 28, "num_key_value_heads": 4},
        )()

        validate_online_selector_mapping(
            config,
            method="contigkv",
            selector_kv_head_ids=(0, 1, 2, 3),
            probe_query_heads=(0, 1, 2),
        )
        validate_online_selector_mapping(
            config,
            method="as_lru",
            selector_kv_head_ids=(0,),
            probe_query_heads=(0,),
        )
        validate_online_selector_mapping(
            config,
            method="as_h2o_lru",
            selector_kv_head_ids=(0, 1, 2, 3),
            probe_query_heads=(0,),
        )
        validate_online_selector_mapping(
            config,
            method="impress",
            selector_kv_head_ids=(0, 0, 1),
            probe_query_heads=(0, 6, 7),
        )
        validate_online_selector_mapping(
            config,
            method="impress",
            selector_kv_head_ids=(0,),
            probe_query_heads=(0, 1, 2),
        )
        validate_online_selector_mapping(
            config,
            method="impress",
            selector_kv_head_ids=(0, 1),
            probe_query_heads=(0, 6, 7),
        )
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            validate_online_selector_mapping(
                config,
                method="impress",
                selector_kv_head_ids=(0, 1, 2),
                probe_query_heads=(0, 1, 2),
            )

    def test_selector_key_slots_deduplicate_shared_gqa_keys(self):
        self.assertEqual(
            selector_key_slots(
                (0, 6, 7),
                num_query_heads=28,
                num_kv_heads=4,
                selector_kv_head_ids=(0, 1),
            ),
            (0, 0, 1),
        )
        self.assertEqual(
            selector_key_slots(
                (0, 6, 7),
                num_query_heads=28,
                num_kv_heads=4,
                selector_kv_head_ids=(0, 0, 1),
            ),
            (0, 0, 2),
        )

    def test_contiguous_selector_timing_uses_events_without_stream_sync(self):
        import torch

        class FakeEvent:
            instances = []

            def __init__(self, *, enable_timing):
                self.enable_timing = enable_timing
                self.recorded = False
                self.synchronized = False
                self.instances.append(self)

            def record(self):
                self.recorded = True

            def synchronize(self):
                self.synchronized = True

            def elapsed_time(self, finished):
                self.assert_finished(finished)
                return 7.5

            def assert_finished(self, finished):
                if not finished.recorded or not finished.synchronized:
                    raise AssertionError("finished event was not resolved")

        class FakeLoader:
            method = "contigkv"
            chunk_size = 2
            prefix_tokens = 4
            keep_ratio = 0.5

            def __init__(self):
                self.compute_ms = None
                self.selection = None

            def load_selector_keys(self, layer):
                self.loaded_layer = layer
                return object()

            def configure_contiguous_period(self, **selection):
                self.selection = selection

            def record_selector_compute(self, elapsed_ms):
                self.compute_ms = elapsed_ms

        loader = FakeLoader()
        scores = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
        with patch.object(torch.cuda, "Event", FakeEvent), patch.object(
            torch.cuda,
            "current_stream",
            side_effect=AssertionError("current stream was synchronized"),
        ), patch(
            "contiguous_fuxian.flexgen_qwen_reprefill."
            "qwen_online_prefix_head_scores",
            return_value=scores,
        ), patch(
            "contiguous_fuxian.flexgen_qwen_reprefill.time.perf_counter",
            side_effect=(10.0, 10.002),
        ):
            configured = configure_online_layer_selection(
                decoder_layer=object(),
                hidden_states=object(),
                position_embeddings=(object(), object()),
                layer_index=3,
                loader=loader,
                period_size=8,
            )

        self.assertEqual(configured, 8)
        self.assertAlmostEqual(loader.compute_ms, 9.5)
        self.assertEqual(len(FakeEvent.instances), 2)
        self.assertTrue(all(event.recorded for event in FakeEvent.instances))

    def test_h2o_selector_configures_per_kv_head_value_positions(self):
        import torch

        class FakeEvent:
            def __init__(self, *, enable_timing):
                self.enable_timing = enable_timing

            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, finished):
                return 5.0

        class FakeLoader:
            method = "as_h2o_lru"

            def __init__(self):
                self.selection = None
                self.compute_ms = None

            def load_selector_keys(self, layer):
                self.loaded_layer = layer
                return object()

            def keep_ratio_for_layer(self, layer):
                return 0.25

            def configure_as_h2o_layer(self, **selection):
                self.selection = selection

            def record_selector_compute(self, elapsed_ms):
                self.compute_ms = elapsed_ms

        config = type(
            "Config",
            (),
            {"num_attention_heads": 4, "num_key_value_heads": 2},
        )()
        decoder_layer = type(
            "Layer",
            (),
            {"self_attn": type("Attention", (), {"config": config})()},
        )()
        head_scores = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
            ]
        )
        loader = FakeLoader()
        with patch.object(torch.cuda, "Event", FakeEvent), patch(
            "contiguous_fuxian.flexgen_qwen_reprefill."
            "qwen_online_prefix_head_scores",
            return_value=head_scores,
        ) as score_mock, patch(
            "contiguous_fuxian.flexgen_qwen_reprefill.time.perf_counter",
            side_effect=(10.0, 10.001),
        ):
            configured = configure_online_layer_selection(
                decoder_layer=decoder_layer,
                hidden_states=object(),
                position_embeddings=(object(), object()),
                layer_index=3,
                loader=loader,
                period_size=8,
            )

        self.assertEqual(configured, 1)
        self.assertEqual(loader.loaded_layer, 3)
        self.assertEqual(loader.selection["layer"], 3)
        self.assertTrue(
            torch.equal(
                loader.selection["positions"],
                torch.tensor([[1], [2]], dtype=torch.long),
            )
        )
        self.assertAlmostEqual(loader.compute_ms, 6.0)
        self.assertIsNone(score_mock.call_args.kwargs["query_heads"])

    def test_impress_contiguous_block_selection_votes_over_aligned_blocks(self):
        selected, used_probe, similarity = impress_contiguous_block_selection(
            [
                [1.0, 1.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
                [0.9, 0.9, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.95, 0.95, 0.0, 0.0],
            ],
            keep_ratio=0.5,
            block_size=2,
            similarity_alpha=0.6,
        )

        self.assertEqual(selected, [0, 1, 4, 5])
        self.assertTrue(used_probe)
        self.assertEqual(similarity, 1.0)

    def test_impress_block_prefetch_ranking_preserves_selected_tokens(self):
        selected, priority, used_probe, similarity = (
            impress_contiguous_block_selection_with_ranking(
                [
                    [0.4, 0.4, 0.0, 0.0, 0.45, 0.45],
                    [0.35, 0.35, 0.0, 0.0, 0.5, 0.5],
                ],
                keep_ratio=2 / 3,
                block_size=2,
                similarity_alpha=0.6,
            )
        )

        self.assertTrue(used_probe)
        self.assertEqual(similarity, 1.0)
        self.assertEqual(selected, [0, 1, 4, 5])
        self.assertEqual(priority, [4, 5, 0, 1])

    def test_exact_layer_block_allocation_preserves_global_budget(self):
        counts = allocate_exact_layer_blocks(
            (0.25, 0.5, 0.75, 0.5),
            blocks_per_layer=10,
            target_ratio=0.5,
        )

        self.assertEqual(sum(counts), 20)
        self.assertEqual(counts, (2, 5, 8, 5))

    def test_exact_uniform_block_rounding_is_depth_balanced(self):
        counts = allocate_exact_layer_blocks(
            (0.25,) * 28,
            blocks_per_layer=239,
            target_ratio=0.25,
        )

        self.assertEqual(sum(counts), 1673)
        self.assertEqual(set(counts), {59, 60})
        self.assertGreater(min(index for index, count in enumerate(counts) if count == 60), -1)
        self.assertTrue(any(count == 60 for count in counts[14:]))

    def test_impress_block_selection_accepts_exact_block_count(self):
        selected, _, used_probe, _ = impress_contiguous_block_selection_with_ranking(
            [[1.0, 1.0, 0.5, 0.5, 0.0, 0.0]] * 3,
            keep_ratio=0.75,
            block_size=2,
            similarity_alpha=0.6,
            keep_blocks=1,
        )

        self.assertTrue(used_probe)
        self.assertEqual(selected, [0, 1])

    def test_prepared_block_scores_preserve_selection_for_each_layer_budget(self):
        scores = [
            [1.0, 0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1],
            [0.9, 1.0, 0.7, 0.8, 0.3, 0.4, 0.1, 0.2],
            [0.8, 0.7, 1.0, 0.9, 0.2, 0.1, 0.4, 0.3],
        ]
        prepared = prepare_impress_contiguous_block_scores(
            scores,
            block_size=2,
        )

        gpu_reduced_equivalent = prepare_impress_block_scores(
            [
                [sum(row[start : start + 2]) for start in range(0, 8, 2)]
                for row in scores
            ],
            block_size=2,
            prefix_tokens=8,
        )
        self.assertEqual(gpu_reduced_equivalent, prepared)

        for keep_blocks in (1, 2, 3, 4):
            with self.subTest(keep_blocks=keep_blocks):
                expected = impress_contiguous_block_selection_with_ranking(
                    scores,
                    keep_ratio=0.5,
                    block_size=2,
                    similarity_alpha=0.6,
                    keep_blocks=keep_blocks,
                    fallback_keep_blocks_limit=keep_blocks,
                )
                actual = select_prepared_impress_blocks(
                    prepared,
                    keep_ratio=0.5,
                    similarity_alpha=0.6,
                    keep_blocks=keep_blocks,
                    fallback_keep_blocks_limit=keep_blocks,
                )
                self.assertEqual(actual, expected)

    def test_impress_contiguous_block_selection_preserves_fallback(self):
        selected, used_probe, similarity = impress_contiguous_block_selection(
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            ],
            keep_ratio=0.25,
            block_size=2,
            similarity_alpha=0.6,
        )

        self.assertEqual(selected, list(range(8)))
        self.assertFalse(used_probe)
        self.assertEqual(similarity, 0.0)

    def test_exact_budget_caps_only_infeasible_dense_fallback(self):
        selected, priority, used_probe, similarity = (
            impress_contiguous_block_selection_with_ranking(
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                ],
                keep_ratio=0.25,
                block_size=2,
                similarity_alpha=0.6,
                keep_blocks=1,
                fallback_keep_blocks_limit=2,
            )
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected), set(priority))
        self.assertFalse(used_probe)
        self.assertEqual(similarity, 0.0)

    def test_impress_block_mode_requires_matching_unreordered_chunks(self):
        validate_impress_block_mode(
            method="impress",
            online_selection=True,
            block_size=16,
            physical_chunk_size=16,
            reorder_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "match"):
            validate_impress_block_mode(
                method="impress",
                online_selection=True,
                block_size=16,
                physical_chunk_size=64,
                reorder_enabled=False,
            )
        with self.assertRaisesRegex(ValueError, "reordering"):
            validate_impress_block_mode(
                method="impress",
                online_selection=True,
                block_size=16,
                physical_chunk_size=16,
                reorder_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "online IMPRESS"):
            validate_impress_block_mode(
                method="contigkv",
                online_selection=True,
                block_size=16,
                physical_chunk_size=16,
                reorder_enabled=False,
            )

    def test_layer_selection_agreement_reports_jaccard_and_exact_fraction(self):
        jaccard, exact = layer_selection_agreement(
            [[1, 2], [4]],
            [[2, 3], [4]],
        )

        self.assertAlmostEqual(jaccard, 2 / 3)
        self.assertEqual(exact, 0.5)


if __name__ == "__main__":
    unittest.main()
