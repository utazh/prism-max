import importlib.util
import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_prism_max_grid.py"
SPEC = importlib.util.spec_from_file_location("analyze_prism_max_grid", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


def percentile95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def make_row(uid, *, correct, logits, critical, selector, keep=0.1, selected=4 * 1024 * 1024):
    first_token = logits + 0.1
    latency = logits + 0.2
    response = logits + 0.3
    evaluation = logits + 10.0
    return {
        "uid": uid,
        "task": uid.split("-", 1)[0],
        "correct": correct,
        "ttft_ms": logits,
        "logits_ready_ms": logits,
        "first_token_ready_ms": first_token,
        "latency_ms": latency,
        "response_ready_ms": response,
        "evaluation_ready_ms": evaluation,
        "accuracy_scores_ready_ms": evaluation,
        "prefetch_wait_ms": logits * 0.2,
        "critical_ssd_read_bytes": critical,
        "selector_disk_source_bytes": selector,
        "total_ssd_read_bytes": critical + selector,
        "effective_mean_keep_ratio": keep,
        "selected_kv_bytes": selected,
    }


def runtime_for(method, budget):
    runtime = {
        "backend": "fp16",
        "accuracy_scoring": "label_continuation_loglikelihood",
        "generation_max_tokens": 1,
        "online_selection": True,
        "cache_type": "CKLFU",
        "gpu_cache_mb": 55.0,
        "cpu_cache_mb": 131.0,
        "model_compute_dtype": "bfloat16",
        "pcache_storage_dtype": "float16",
        "defer_cache_score_updates": False,
        "warmup_passes": 1,
        "warmup_samples_per_task": 1,
        "keep_ratio": int(budget) / 100.0,
        "period_size": 64,
        "subperiod_size": 16,
        "prefetch_time_budget": 0.0,
        "cache_update_in_ttft": True,
        "response_ready_metric_valid_for_first_token": True,
        "response_ready_excludes_accuracy_scoring": True,
        "evaluation_ready_includes_accuracy_scoring": True,
        "selector_index_dir": None,
        "selector_index_bits": None,
        "selector_index_group_size": None,
        "selector_index_manifest_sha256": None,
        "selector_index_preloaded_bytes": 0,
        "similarity_alpha": 1.0,
    }
    runtime.update(grid.METHOD_CONTRACTS[method])
    return runtime


def write_run(path, *, task, budget, method, rows, runtime_overrides=None):
    path.mkdir(parents=True)
    logits = [row["logits_ready_ms"] for row in rows]
    aggregate = {
        "samples": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_logits_ready_ms": sum(logits) / len(logits),
        "p95_logits_ready_ms": percentile95(logits),
        "mean_ttft_ms": sum(logits) / len(logits),
        "p95_ttft_ms": percentile95(logits),
    }
    runtime = runtime_for(method, budget)
    runtime.update(runtime_overrides or {})
    summary = {
        "measurement": "test measurement contract",
        "model_path": "/model",
        "runtime": runtime,
        "tasks": {task: aggregate},
        "overall": aggregate,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "scored_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class PrismMaxGridAnalysisTest(unittest.TestCase):
    def test_impress_reorder_sha256_is_canonical_literal(self):
        canonical = (
            "36c5e1ec62187f8916e04e7758e79c9a28cfe1c75cb8999739e6be15287542cf"
        )
        actual = grid.IMPRESS_REORDER_SHA256
        self.assertEqual(actual, canonical)
        self.assertEqual(len(actual), 64)
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", actual))

    def test_fp16_preloaded_bytes_accepts_none_or_numeric_zero_only(self):
        def loaded(preloaded_bytes):
            spec = grid.RunSpec(
                "trec",
                "010",
                "impress",
                "r0",
                Path("/unused/impress/r0"),
            )
            return grid.LoadedRun(
                spec=spec,
                summary={
                    "runtime": {
                        "selector_index_dir": None,
                        "selector_index_bits": None,
                        "selector_index_group_size": None,
                        "selector_index_manifest_sha256": None,
                        "selector_index_preloaded_bytes": preloaded_bytes,
                    }
                },
                records={"trec-1": {"correct": True}},
                raw_count=1,
                excluded_uids=[],
            )

        for accepted in (None, 0, 0.0):
            with self.subTest(accepted=accepted):
                grid.validate_impress_cells_are_fp16([loaded(accepted)])

        for rejected in (False, 1, "0"):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(
                    ValueError,
                    "must be None or numeric zero",
                ):
                    grid.validate_impress_cells_are_fp16([loaded(rejected)])

    def test_bundle_preexclusions_are_reported_as_input_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata_path = Path(temp) / "metadata.json"
            excluded = {task: [f"{task}-0"] for task in grid.TASKS}
            metadata_path.write_text(
                json.dumps(
                    {
                        "tasks": {
                            task: {"strict_excluded_uids": values}
                            for task, values in excluded.items()
                        },
                        "strict_eval_filter": {
                            "schema_version": 1,
                            "exclusions_manifest": "/bundle/strict.json",
                            "exclusions_manifest_sha256": "a" * 64,
                            "excluded_uids_by_task": excluded,
                            "source_records_by_task": {},
                            "evaluation_records_by_task": {},
                        },
                    }
                ),
                encoding="utf-8",
            )

            provenance = grid.load_bundle_preexclusions(metadata_path)

        self.assertTrue(provenance["preapplied"])
        self.assertEqual(provenance["excluded_uids_by_task"], excluded)
        self.assertEqual(provenance["manifest_path"], "/bundle/strict.json")
        self.assertEqual(len(provenance["metadata_sha256"]), 64)
        self.assertEqual(
            provenance["application_stage"],
            "strict bundle construction before benchmark runs",
        )

    def test_repeat_average_exclusions_metrics_and_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = []
            correct_by_method = {
                "contigkv": [False, True, False],
                "impress": [False, False, False],
                "promixed": [False, True, True],
            }
            latency_by_method = {"contigkv": 100.0, "impress": 90.0, "promixed": 80.0}
            traffic_by_method = {
                "contigkv": (3 * 1024 * 1024, 0),
                "impress": (2 * 1024 * 1024, 1024 * 1024),
                "promixed": (2 * 1024 * 1024, 512 * 1024),
            }
            for method in grid.METHODS:
                for repeat_index, repeat in enumerate(("r0", "r1")):
                    rows = [
                        make_row(
                            f"trec-{uid_index}",
                            correct=correct_by_method[method][uid_index],
                            logits=latency_by_method[method] + uid_index + 20.0 * repeat_index,
                            critical=traffic_by_method[method][0],
                            selector=traffic_by_method[method][1],
                            keep=0.095,
                        )
                        for uid_index in range(3)
                    ]
                    path = root / method / repeat
                    write_run(path, task="trec", budget="010", method=method, rows=rows)
                    specs.append(grid.RunSpec("trec", "010", method, repeat, path))

            exclusion_path = root / "exclusions.json"
            exclusion_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "purpose": "exclude colocated warmup UID",
                        "exclude_uids_by_task": {"trec": ["trec-0"]},
                    }
                ),
                encoding="utf-8",
            )
            exclusions, exclusion_info = grid.load_exclusions(exclusion_path)
            result = grid.analyze(
                specs,
                exclusions,
                source={"kind": "test"},
                exclusion_info=exclusion_info,
                bootstrap_samples=100,
                seed=7,
            )

            self.assertEqual(
                result["protocol_by_task"]["trec"]["selector_index_preloaded_bytes"],
                0,
            )
            promixed = result["tasks"]["trec"]["budgets"]["010"]["methods"]["promixed"]
            self.assertEqual(promixed["uids"], 2)
            self.assertEqual(promixed["repeats"], 2)
            self.assertAlmostEqual(promixed["accuracy"], 1.0)
            self.assertAlmostEqual(promixed["logits_ready_mean_ms"], 91.5)
            self.assertAlmostEqual(promixed["response_ready_mean_ms"], 91.8)
            self.assertAlmostEqual(promixed["ssd_mib_per_request"]["critical"], 2.0)
            self.assertAlmostEqual(promixed["ssd_mib_per_request"]["selector"], 0.5)
            self.assertAlmostEqual(promixed["ssd_mib_per_request"]["total"], 2.5)
            self.assertAlmostEqual(promixed["prefetch_stall_ratio"], 0.2)
            self.assertAlmostEqual(promixed["fairness_context"]["actual_keep_ratio"], 0.095)

            paired = result["tasks"]["trec"]["budgets"]["010"]["paired"]["promixed_vs_contigkv"]
            self.assertTrue(paired["available"])
            self.assertEqual(paired["uids"], 2)
            self.assertAlmostEqual(paired["accuracy_delta_pp"], 50.0)
            self.assertAlmostEqual(paired["response_ready_delta_ms"], -20.0)
            self.assertAlmostEqual(paired["logits_ready_delta_ms"], -20.0)
            self.assertEqual(len(paired["accuracy_delta_pp_paired_bootstrap_95ci"]), 2)

            exclusions_report = result["exclusions"]
            self.assertEqual(exclusions_report["observed_by_task"]["trec"], ["trec-0"])
            self.assertEqual(exclusions_report["observed_count_by_task"]["trec"], 1)
            run_report = exclusions_report["per_run"]["trec/k010/promixed/r0"]
            self.assertEqual(run_report["raw_records"], 3)
            self.assertEqual(run_report["excluded_records"], 1)
            self.assertEqual(run_report["analysis_records"], 2)

            markdown = grid.render_markdown(result)
            self.assertIn("Tasks are never pooled", markdown)
            self.assertIn("SSD total (critical + selector)", markdown)
            self.assertIn("Primary latency: response-ready", markdown)
            self.assertIn("Response-ready mean / P95", markdown)
            self.assertIn("Logits-ready phase mean / P95", markdown)
            self.assertIn("Analysis-time exclusion manifest", markdown)
            self.assertNotIn("Overall", markdown)

    def test_strict_timing_and_ssd_alias_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case in ("timing", "ssd"):
                with self.subTest(case=case):
                    path = root / case
                    row = make_row(
                        "trec-0",
                        correct=True,
                        logits=100.0,
                        critical=1024,
                        selector=512,
                    )
                    if case == "timing":
                        row["ttft_ms"] = 99.0
                    else:
                        row["total_ssd_read_bytes"] += 1
                    write_run(path, task="trec", budget="010", method="promixed", rows=[row])
                    spec = grid.RunSpec("trec", "010", "promixed", "r0", path)
                    with self.assertRaisesRegex(ValueError, "ttft_ms == logits_ready_ms|total SSD bytes"):
                        grid.load_run(spec, set())

    def test_wrong_impress_reorder_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "impress"
            row = make_row(
                "trec-0",
                correct=True,
                logits=100.0,
                critical=1024,
                selector=512,
            )
            write_run(
                path,
                task="trec",
                budget="010",
                method="impress",
                rows=[row],
                runtime_overrides={"impress_reorder_enabled": False},
            )
            spec = grid.RunSpec("trec", "010", "impress", "r0", path)
            with self.assertRaisesRegex(ValueError, "impress_reorder_enabled"):
                grid.load_run(spec, set())

    def test_method_repeat_label_sets_must_match(self):
        def loaded(method, repeat):
            spec = grid.RunSpec(
                "trec",
                "010",
                method,
                repeat,
                Path("/unused") / method / repeat,
            )
            return grid.LoadedRun(
                spec=spec,
                summary={"runtime": runtime_for(method, "010")},
                records={"trec-1": {"correct": True}},
                raw_count=1,
                excluded_uids=[],
            )

        runs = [
            loaded("contigkv", "r0"),
            loaded("contigkv", "r1"),
            loaded("impress", "r0"),
            loaded("promixed", "r0"),
            loaded("promixed", "r1"),
        ]
        with self.assertRaisesRegex(ValueError, "repeat label sets differ"):
            grid.group_runs(runs)

    def test_stall_ratio_averages_run_ratios_not_ratio_of_means(self):
        def loaded(repeat, logits, wait):
            spec = grid.RunSpec(
                "trec",
                "010",
                "promixed",
                repeat,
                Path("/unused") / repeat,
            )
            row = {
                "correct": True,
                "logits_ready_ms": logits,
                "response_ready_ms": logits + 0.3,
                "prefetch_wait_ms": wait,
                "critical_ssd_read_bytes": 0,
                "selector_disk_source_bytes": 0,
                "total_ssd_read_bytes": 0,
                "effective_mean_keep_ratio": 0.1,
                "selected_kv_bytes": 1024,
            }
            return grid.LoadedRun(
                spec=spec,
                summary={"runtime": runtime_for("promixed", "010")},
                records={"trec-1": row},
                raw_count=1,
                excluded_uids=[],
            )

        aggregate = grid.aggregate_uids(
            [loaded("r0", 1.0, 1.0), loaded("r1", 9.0, 0.0)]
        )
        result = aggregate["trec-1"]
        self.assertAlmostEqual(result["prefetch_stall_ratio"], 0.5)
        ratio_of_means = (
            float(result["prefetch_wait_ms"])
            / float(result["logits_ready_ms"])
        )
        self.assertAlmostEqual(ratio_of_means, 0.1)
        self.assertNotAlmostEqual(result["prefetch_stall_ratio"], ratio_of_means)

    def test_mismatched_method_uids_are_not_intersection_pooled(self):
        def aggregated(uid):
            return {
                uid: {
                    "correct": True,
                    "logits_ready_ms": 1.0,
                    "response_ready_ms": 1.3,
                    "total_ssd_read_bytes": 0.0,
                    "prefetch_stall_ratio": 0.0,
                }
            }

        paired = grid.paired_comparison(
            aggregated("trec-1"),
            aggregated("trec-2"),
            baseline_method="contigkv",
            bootstrap_samples=10,
            seed=1,
        )
        self.assertFalse(paired["available"])
        self.assertIn("no intersection pooling", paired["reason"])
        self.assertEqual(paired["candidate_only"], ["trec-1"])
        self.assertEqual(paired["baseline_only"], ["trec-2"])

    def test_cross_budget_pareto(self):
        points = [
            {"method": "promixed", "budget": "005", "accuracy": 0.7, "response_ready_mean_ms": 80.0},
            {"method": "promixed", "budget": "010", "accuracy": 0.8, "response_ready_mean_ms": 90.0},
            {"method": "promixed", "budget": "025", "accuracy": 0.75, "response_ready_mean_ms": 100.0},
            {"method": "promixed", "budget": "050", "accuracy": 0.9, "response_ready_mean_ms": 120.0},
        ]
        marked = {point["budget"]: point["pareto_optimal"] for point in grid.mark_pareto(points)}
        self.assertEqual(marked, {"005": True, "010": True, "025": False, "050": True})

    def test_manifest_paths_and_duplicate_identity_are_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = {
                "task": "trec",
                "budget": "k010",
                "method": "promixed",
                "repeat": "r0",
                "path": "runs/p0",
            }
            manifest = root / "schedule.json"
            manifest.write_text(
                json.dumps({"schema_version": 1, "purpose": "test", "runs": [item]}),
                encoding="utf-8",
            )
            specs, source = grid.load_specs(manifest=manifest, explicit_runs=None)
            self.assertEqual(specs[0].path, (root / "runs" / "p0").resolve())
            self.assertEqual(source["kind"], "manifest")

            manifest.write_text(
                json.dumps({"schema_version": 1, "runs": [item, item]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                grid.load_specs(manifest=manifest, explicit_runs=None)


if __name__ == "__main__":
    unittest.main()
