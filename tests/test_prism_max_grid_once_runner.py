import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_prism_max_grid_once.sh"


class PrismMaxGridOnceRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = RUNNER.read_text(encoding="utf-8")

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_launcher_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_grid_is_fixed_to_the_requested_48_executions(self):
        self.assertIn("TASKS=(sst2 subj trec rte)", self.script)
        self.assertIn("BUDGETS=(005 010 025 050)", self.script)
        self.assertIn('"contigkv impress promixed"', self.script)
        self.assertIn('"impress promixed contigkv"', self.script)
        self.assertIn('"promixed contigkv impress"', self.script)
        self.assertIn('if len(runs) != 48:', self.script)
        self.assertIn('"repeat": "r1"', self.script)
        self.assertIn(
            'local run_name="k${budget}_${method}_fp16_nodefer_r1"',
            self.script,
        )

    def test_full_strict_fp16_nodefer_no_warmup_contract_is_explicit(self):
        expected_fragments = (
            "paper_task_bundles_full_eval_strict",
            "SELECTOR_BACKEND=fp16",
            "SAMPLES_PER_TASK=1000000",
            "WARMUP_PASSES=0",
            "WARMUP_SAMPLES_PER_TASK=32",
            "DEFER_CACHE_SCORE_UPDATES=false",
            "PROMIXED_ADAPTIVE_COVERAGE=false",
            '"warmup_passes": 0',
            '"warmup_requests": 0',
            '"defer_cache_score_updates": False',
            '"online_selection": True',
            '"cache_type": "CKLFU"',
            '"registered_store_tasks": ["sst2", "subj", "trec", "rte"]',
            '"selector_index_dir": None',
            '"selector_index_preloaded_bytes"',
            '"adaptive_coverage"',
        )
        for fragment in expected_fragments:
            self.assertIn(fragment, self.script)
        self.assertNotIn("--exclude-uids", self.script)

    def test_frozen_counts_and_calibration_exclusion_are_validated(self):
        for task, count in (
            ("sst2", 867),
            ("subj", 998),
            ("trec", 495),
            ("rte", 272),
        ):
            self.assertRegex(self.script, rf"\[{re.escape(task)}\]={count}\b")
        self.assertIn('if f"{task}-0" in set(uids):', self.script)
        self.assertIn("strict_eval_filter", self.script)
        self.assertIn("bf70ce022f771b59bcc664d321b0787", self.script)
        self.assertIn("output_task_jsonl_sha256", self.script)
        self.assertIn("UID set differs from strict", self.script)
        self.assertIn('"$BUNDLE_DIR/$task.jsonl"', self.script)

    def test_resume_does_not_overwrite_outputs_and_handles_only_safe_codes(self):
        self.assertIn('if [[ -e "$output" ]]; then', self.script)
        self.assertIn("skip validated completed", self.script)
        self.assertIn("will not be overwritten", self.script)
        self.assertRegex(self.script, r"\n\s+3\)\n")
        self.assertRegex(self.script, r"\n\s+4\)\n")
        self.assertIn("accepted validated final output after runner status", self.script)
        self.assertIn("archive_stale_attempts", self.script)
        self.assertIn('mv -- "${artifacts[@]}" "$archive/"', self.script)
        self.assertIn("runner failed for", self.script)

    def test_global_lock_commit_and_input_fingerprints_are_enforced(self):
        self.assertIn("prism_max_grid_fp16_nodefer_strict_r1.lock", self.script)
        self.assertIn("flock -n 6", self.script)
        self.assertIn("SOURCE_COMMIT_FILE", self.script)
        self.assertIn("assert_tracked_source_clean", self.script)
        self.assertIn("ls-files --error-unmatch", self.script)
        self.assertIn("assert_source_frozen", self.script)
        self.assertIn("source_and_input_sha256.txt", self.script)
        self.assertIn('sha256sum --status --check "$FINGERPRINT"', self.script)

    def test_final_analyzer_uses_frozen_manifest_without_overwrite(self):
        self.assertIn('SCHEDULE_MANIFEST="$RUN_ROOT/schedule_manifest.json"', self.script)
        self.assertIn('--manifest "$SCHEDULE_MANIFEST"', self.script)
        self.assertIn('--bootstrap-samples "$FINAL_BOOTSTRAP_SAMPLES"', self.script)
        self.assertIn(
            'ANALYSIS_STEM="$RUN_ROOT/prism_max_grid_fp16_nodefer_strict_r1"',
            self.script,
        )
        self.assertIn("existing analysis differs", self.script)
        self.assertIn('analysis_candidate_dir="$RUN_ROOT/.analysis_candidate.$$"', self.script)
        self.assertIn('analysis_candidate="$analysis_candidate_dir/prism_max_grid_', self.script)
        self.assertIn('local validation_stem="$validation_dir/check"', self.script)
        self.assertIn('"$RUN_ROOT/grid.done"', self.script)


if __name__ == "__main__":
    unittest.main()
