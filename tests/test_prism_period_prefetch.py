import unittest

from contiguous_fuxian.flexgen_pcache import FlexGenLayerLoader


class PrismPeriodPrefetchTest(unittest.TestCase):
    @staticmethod
    def _loader(
        *,
        layers=8,
        selections=None,
        online_selection=True,
        method="impress",
        async_prefetch=True,
        time_budget=0.125,
        period_budget_scale=1.0,
        rolling_period_source=False,
        value_ordered=False,
        value_budget_scale=1.0,
        priority_selections=None,
    ):
        loader = FlexGenLayerLoader.__new__(FlexGenLayerLoader)
        loader._info = type("Info", (), {"layers": layers})()
        loader.online_selection = online_selection
        loader.method = method
        loader.impress_async_prefetch = async_prefetch
        loader.impress_rolling_period_prefetch = rolling_period_source
        loader.impress_value_ordered_prefetch = value_ordered
        loader.impress_value_prefetch_budget_scale = value_budget_scale
        loader._selected_by_layer = [
            list(tokens)
            for tokens in (
                selections
                if selections is not None
                else ([layer] for layer in range(layers))
            )
        ]
        loader._prefetch_priority_by_layer = [
            list(tokens)
            for tokens in (
                priority_selections
                if priority_selections is not None
                else loader._selected_by_layer
            )
        ]
        loader._pending = {}
        loader._speculative_pending = {}
        loader._resolved_speculative = {}
        loader._missing_pending = {}
        loader._loaded = {}
        loader._speculative_positions = {}
        loader._speculative_kind = {}
        loader._speculative_requested_positions = {}
        loader._speculative_source_layer = {}
        loader._impress_prefetch_time_budget = time_budget
        loader.impress_period_prefetch_budget_scale = period_budget_scale
        loader._impress_prefetch_budgets = []
        loader._impress_next_prefetch_jobs = 0
        loader._impress_period_prefetch_jobs = 0
        loader._impress_period_prefetch_tokens = 0
        loader._impress_value_ordered_prefetch_jobs = 0
        submissions = []

        def submit(layer, positions, pending, **kwargs):
            submissions.append(
                (
                    layer,
                    list(positions),
                    pending is loader._speculative_pending,
                    kwargs,
                )
            )
            pending[layer] = (object(), 0.0)

        loader._submit_positions = submit
        return loader, submissions

    def test_period_prefetch_is_inactive_outside_async_online_impress(self):
        inactive_modes = (
            (False, "impress", True),
            (True, "contigkv", True),
            (True, "impress", False),
        )

        for online_selection, method, async_prefetch in inactive_modes:
            with self.subTest(
                online_selection=online_selection,
                method=method,
                async_prefetch=async_prefetch,
            ):
                loader, submissions = self._loader(
                    online_selection=online_selection,
                    method=method,
                    async_prefetch=async_prefetch,
                )

                loader.schedule_impress_period(0, period_size=4)

                self.assertEqual(submissions, [])
                self.assertEqual(loader._speculative_positions, {})
                self.assertEqual(loader._impress_prefetch_budgets, [])
                self.assertEqual(loader._impress_period_prefetch_jobs, 0)
                self.assertEqual(loader._impress_period_prefetch_tokens, 0)

    def test_period_prefetch_requires_a_target_inside_a_multi_layer_period(self):
        inactive_calls = (
            (0, 0),
            (0, 1),
            (2, 4),
            (3, 4),
            (6, 4),
        )

        for layer, period_size in inactive_calls:
            with self.subTest(layer=layer, period_size=period_size):
                loader, submissions = self._loader()

                loader.schedule_impress_period(layer, period_size)

                self.assertEqual(submissions, [])
                self.assertEqual(loader._speculative_positions, {})
                self.assertEqual(loader._impress_period_prefetch_jobs, 0)
                self.assertEqual(loader._impress_period_prefetch_tokens, 0)

    def test_period_prefetch_submits_exact_targets_positions_and_budgets(self):
        selections = [[0], [1], [2], [3], [2, 9], [5], [6], [7]]
        loader, submissions = self._loader(
            layers=8,
            selections=selections,
            time_budget=0.125,
        )

        loader.schedule_impress_next(4)
        loader.schedule_impress_period(4, period_size=4)

        self.assertEqual(
            submissions,
            [
                (
                    5,
                    [2, 9],
                    True,
                    {"time_budget": 0.125, "prefetch_kind": "next"},
                ),
                (
                    6,
                    [2, 9],
                    True,
                    {"time_budget": 0.125, "prefetch_kind": "period"},
                ),
            ],
        )
        self.assertEqual(
            loader._speculative_positions,
            {5: [2, 9], 6: [2, 9]},
        )
        self.assertEqual(loader._impress_prefetch_budgets, [0.125, 0.125])
        self.assertEqual(loader._impress_next_prefetch_jobs, 1)
        self.assertEqual(loader._impress_period_prefetch_jobs, 1)
        self.assertEqual(loader._impress_period_prefetch_tokens, 2)

    def test_period_prefetch_skips_loaded_pending_and_resolved_targets(self):
        loader, submissions = self._loader(
            layers=8,
            selections=[[2, 5, 9]] + [[layer] for layer in range(1, 8)],
            time_budget=0.25,
        )
        target_maps = (
            "_pending",
            "_speculative_pending",
            "_resolved_speculative",
            "_missing_pending",
            "_loaded",
        )
        for target_map in target_maps:
            with self.subTest(target_map=target_map):
                loader, submissions = self._loader(
                    layers=8,
                    selections=[[2, 5, 9]] + [[layer] for layer in range(1, 8)],
                    time_budget=0.25,
                )
                getattr(loader, target_map)[2] = ("occupied", 0.0)
                loader._speculative_positions[2] = [102]

                loader.schedule_impress_period(0, period_size=7)

                self.assertEqual(submissions, [])
                self.assertEqual(loader._speculative_positions, {2: [102]})
                self.assertEqual(loader._impress_prefetch_budgets, [])
                self.assertEqual(loader._impress_period_prefetch_jobs, 0)
                self.assertEqual(loader._impress_period_prefetch_tokens, 0)

    def test_period_predictions_preserve_final_target_selections(self):
        selections = [[1, 7], [2], [3, 8], [4, 5, 9]]
        loader, _ = self._loader(layers=4, selections=selections)
        original_rows = list(loader._selected_by_layer)
        original_values = [list(tokens) for tokens in loader._selected_by_layer]

        loader.schedule_impress_period(0, period_size=4)

        self.assertEqual(loader._selected_by_layer, original_values)
        for original, current in zip(original_rows, loader._selected_by_layer):
            self.assertIs(current, original)
        self.assertEqual(
            loader._speculative_positions,
            {2: [1, 7]},
        )

    def test_next_prefetch_still_covers_the_next_period_leader(self):
        selections = [[0, 4], [10], [20, 21], [30], [40], [50]]
        loader, submissions = self._loader(
            layers=6,
            selections=selections,
            time_budget=0.5,
        )

        loader.schedule_impress_next(0)
        loader.schedule_impress_period(0, period_size=3)
        loader.schedule_impress_next(2)

        self.assertEqual(
            submissions,
            [
                (
                    1,
                    [0, 4],
                    True,
                    {"time_budget": 0.5, "prefetch_kind": "next"},
                ),
                (
                    2,
                    [0, 4],
                    True,
                    {"time_budget": 0.5, "prefetch_kind": "period"},
                ),
                (
                    3,
                    [20, 21],
                    True,
                    {"time_budget": 0.5, "prefetch_kind": "next"},
                ),
            ],
        )
        self.assertEqual(loader._speculative_positions[3], [20, 21])
        self.assertEqual(loader._selected_by_layer[3], [30])
        self.assertEqual(loader._impress_next_prefetch_jobs, 2)
        self.assertEqual(loader._impress_period_prefetch_jobs, 1)
        self.assertEqual(loader._impress_period_prefetch_tokens, 2)
        self.assertEqual(loader._impress_prefetch_budgets, [0.5, 0.5, 0.5])

    def test_period_predictions_are_issued_with_bounded_lookahead(self):
        loader, submissions = self._loader(
            layers=5,
            selections=[[7, 9], [1], [2], [3], [4]],
            time_budget=0.2,
        )

        loader.schedule_impress_next(0)
        loader.schedule_impress_period(0, period_size=4)
        loader.schedule_impress_period(1, period_size=4)
        loader.schedule_impress_period(2, period_size=4)

        self.assertEqual(
            submissions,
            [
                (
                    1,
                    [7, 9],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "next"},
                ),
                (
                    2,
                    [7, 9],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "period"},
                ),
                (
                    3,
                    [7, 9],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "period"},
                ),
            ],
        )
        self.assertEqual(loader._impress_period_prefetch_jobs, 2)

    def test_rolling_period_predictions_use_the_latest_available_layer(self):
        loader, submissions = self._loader(
            layers=4,
            selections=[[0], [10], [20], [30]],
            time_budget=0.2,
            rolling_period_source=True,
        )

        loader.schedule_impress_period(0, period_size=4)
        loader.schedule_impress_period(1, period_size=4)

        self.assertEqual(
            submissions,
            [
                (
                    2,
                    [0],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "period"},
                ),
                (
                    3,
                    [10],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "period"},
                ),
            ],
        )
        self.assertEqual(loader._speculative_source_layer, {2: 0, 3: 1})

    def test_period_prediction_scales_only_its_own_time_budget(self):
        loader, submissions = self._loader(
            layers=4,
            selections=[[7, 9], [1], [2], [3]],
            time_budget=0.2,
            period_budget_scale=0.25,
        )

        loader.schedule_impress_next(0)
        loader.schedule_impress_period(0, period_size=4)

        self.assertEqual(
            submissions,
            [
                (
                    1,
                    [7, 9],
                    True,
                    {"time_budget": 0.2, "prefetch_kind": "next"},
                ),
                (
                    2,
                    [7, 9],
                    True,
                    {"time_budget": 0.05, "prefetch_kind": "period"},
                ),
            ],
        )

    def test_value_budget_scale_applies_to_next_and_period_prefetch(self):
        loader, submissions = self._loader(
            layers=4,
            selections=[[7, 9], [1], [2], [3]],
            priority_selections=[[9, 7], [1], [2], [3]],
            time_budget=0.2,
            period_budget_scale=0.25,
            value_ordered=True,
            value_budget_scale=0.5,
        )

        loader.schedule_impress_next(0)
        loader.schedule_impress_period(0, period_size=4)

        self.assertEqual(
            submissions,
            [
                (
                    1,
                    [9, 7],
                    True,
                    {"time_budget": 0.1, "prefetch_kind": "next"},
                ),
                (
                    2,
                    [9, 7],
                    True,
                    {"time_budget": 0.025, "prefetch_kind": "period"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
