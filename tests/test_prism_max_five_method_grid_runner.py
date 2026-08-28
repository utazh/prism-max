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

    def test_exact_68_execution_80_cell_protocol(self):
        for fragment in (
            "TASKS=(sst2 subj trec rte)",
            "BUDGETS=(005 010 025 050)",
            'if len(runs) != 68:',
            '[[ "$execution_count" -eq 68 ]]',
            "executions=68",
            "expanded_cells=80",
            '"budget": "full"',
            '"method": "as_lru"',
        ):
            self.assertIn(fragment, self.script)

    def test_four_varying_methods_rotate_and_as_positions_differ(self):
        for order in (
            '"impress contigkv promixed as_h2o_lru"',
            '"contigkv promixed as_h2o_lru impress"',
            '"promixed as_h2o_lru impress contigkv"',
            '"as_h2o_lru impress contigkv promixed"',
        ):
            self.assertIn(order, self.script)
        self.assertIn("AS_INSERT_OFFSETS=(0 5 10 16)", self.script)
        self.assertIn("as_insert_offsets = (0, 5, 10, 16)", self.script)

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

    def test_strict_bundle_plain_as_and_runtime_validation_are_frozen(self):
        for fragment in (
            "bf70ce022f771b59bcc664d321b0787a5c8e1d50f761e6833d95c1c91160abdb5",
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
