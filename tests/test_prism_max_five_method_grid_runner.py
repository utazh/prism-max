import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_prism_max_five_method_grid.sh"


class PrismMaxFiveMethodGridRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = LAUNCHER.read_bytes()
        cls.script = cls.payload.decode("utf-8")

    def test_shell_file_is_lf_only(self):
        self.assertNotIn(b"\r", self.payload)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_launcher_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exact_80_execution_protocol(self):
        for fragment in (
            "TASKS=(sst2 subj trec rte)",
            "BUDGETS=(005 010 025 050)",
            'if len(runs) != 80:',
            '[[ "$execution_count" -eq 80 ]]',
            "executions=80",
            "expanded_cells=80",
            "80 actual executions",
            "independently timed with effective full K/V",
        ):
            self.assertIn(fragment, self.script)
        self.assertNotIn("expected 68 runs", self.script)
        self.assertNotIn("80 projected cells", self.script)

    def test_all_five_methods_rotate_within_each_budget_cell(self):
        for order in (
            '"impress contigkv promixed as_lru as_h2o_lru"',
            '"contigkv promixed as_lru as_h2o_lru impress"',
            '"promixed as_lru as_h2o_lru impress contigkv"',
            '"as_lru as_h2o_lru impress contigkv promixed"',
            '"as_h2o_lru impress contigkv promixed as_lru"',
        ):
            self.assertIn(order, self.script)
        self.assertNotIn("AS_INSERT_OFFSETS", self.script)

    def test_as_lru_gets_four_independent_paths_but_full_runtime_semantics(self):
        for fragment in (
            'name = f"k{budget}_{method}_{selector}_nodefer_warm1_response_r1"',
            '[[ "$method" != "as_lru" ]] || runner_budget="full"',
            'BUDGET_TAG="$runner_budget"',
            '"keep_ratio": 1.0',
            '"actual_key_keep_ratio": 1.0',
            '"actual_value_keep_ratio": 1.0',
            '"actual_total_logical_kv_ratio": 1.0',
            '"budget_semantics": "full-kv-budget-independent"',
        ):
            self.assertIn(fragment, self.script)

    def test_as_h2o_uses_compact_attention_schema_with_full_key_selector(self):
        for fragment in (
            '"actual_key_keep_ratio": ratio',
            '"actual_value_keep_ratio": ratio',
            '"actual_total_logical_kv_ratio": ratio',
            '"logical_attention_keep_ratio": ratio',
            '"selector_full_key_load_ratio": 1.0',
            '"minimum_transfer_ratio": (1.0 + ratio) / 2.0',
            '"h2o-logical-attention-retention-with-full-key-selector-transfer"',
            '"as_h2o_selector_full_key_ratio"',
            '"as_h2o_logical_attention_keep_ratio"',
            '"as_h2o_selected_value_transfer_ratio"',
            '"as_h2o_minimum_transfer_ratio"',
        ):
            self.assertIn(fragment, self.script)
        for obsolete in (
            '"as_h2o_full_key_ratio"',
            '"as_h2o_value_keep_ratio"',
            '"as_h2o_total_logical_payload_ratio"',
        ):
            self.assertNotIn(obsolete, self.script)

    def test_method_specific_backends_and_clean_measurement_boundary(self):
        for fragment in (
            'selector = "k4" if method == "promixed" else "fp16"',
            '[[ "$method" != "promixed" ]] || selector="k4"',
            "WARMUP_PASSES=1",
            "WARMUP_SAMPLES_PER_TASK=32",
            "DEFER_CACHE_SCORE_UPDATES=false",
            "SAMPLES_PER_TASK=1000000",
            "primary_latency=response_ready_ms",
            "accuracy_scoring=label_continuation_loglikelihood",
            '"response_ready_metric_valid_for_first_token": True',
            '"response_ready_excludes_accuracy_scoring": True',
        ):
            self.assertIn(fragment, self.script)
        self.assertNotIn("WARMUP_PASSES=0", self.script)

    def test_legacy_68_manifest_is_safely_migrated(self):
        for fragment in (
            "LEGACY_68_PROTOCOL=false",
            "len(runs) != 68",
            'row.get("budget") != "full"',
            'manifest_path.name + ".legacy68"',
            'execution_path.name + ".legacy68"',
            '"summary.json").is_file()',
            '"scored_records.jsonl"',
            'method == "as_lru" and budget == "005"',
            "68-to-80 protocol source upgrade",
            "archived 68-run fingerprint",
            "protocol80_migration_pending",
            "a missing imported legacy output will not be recreated outside RUN_ROOT",
        ):
            self.assertIn(fragment, self.script)

    def test_strict_bundle_plain_as_and_runtime_validation_are_frozen(self):
        for fragment in (
            "bf70ce022f771b59bcc664d321b0787a5c8e1d50f761e6833d95c1c9160abdb5",
            'if f"{task}-0" in set(uids):',
            '"physical_layout": "plain-logical-token-order"',
            '"impress_reorder_sha256": None',
            '"selector_kv_head_ids": [0, 1, 2, 3]',
            '"warmup_requests": 32',
            '"accuracy_scoring": "label_continuation_loglikelihood"',
            '"cache_type": "LRU"',
            '"cache_type": "CKLFU"',
            '"selector_index_bits") != 4',
            'runtime.get("selector_index_preloaded_tasks") != [task]',
            'runtime.get("impress_reorder_enabled") is not True',
        ):
            self.assertIn(fragment, self.script)

    def test_resume_safety_lock_and_source_fingerprint(self):
        for fragment in (
            "flock -n 6",
            "assert_source_frozen",
            'sha256sum --status --check "$FINGERPRINT"',
            "skip validated completed",
            "will not be overwritten",
            "archive_stale_attempts",
            "source commit changed during the grid",
            "resource guard/lock busy",
        ):
            self.assertIn(fragment, self.script)

    def test_new_analyzer_is_used_without_touching_legacy_launcher(self):
        self.assertIn(
            'ANALYZER="${ANALYZER:-$ROOT/scripts/analyze_five_method_grid.py}"',
            self.script,
        )
        self.assertIn('--manifest "$SCHEDULE_MANIFEST"', self.script)
        self.assertIn('--output "$analysis_candidate"', self.script)
        self.assertNotIn("analyze_prism_max_grid.py", self.script)
        self.assertNotIn("run_prism_max_grid_once.sh", self.script)


if __name__ == "__main__":
    unittest.main()
