import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_prism_max_cell.sh"


class PrismMaxRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = RUNNER.read_text(encoding="utf-8")
        cls.impress_case = cls.script.split("\n  impress)\n", 1)[1].split(
            "\n  promixed)\n", 1
        )[0]
        cls.as_lru_case = cls.script.split("\n  as_lru)\n", 1)[1].split(
            "\n  as_h2o_lru)\n", 1
        )[0]
        cls.as_h2o_lru_case = cls.script.split("\n  as_h2o_lru)\n", 1)[1].split(
            "\n  *)\n", 1
        )[0]

    @staticmethod
    def run_runner(**environment_updates):
        environment = os.environ.copy()
        environment.update(
            {
                "METHOD": "contigkv",
                "SELECTOR_BACKEND": "fp16",
                "BUDGET_TAG": "010",
                "TASK": "trec",
                "BUNDLE_DIR": str(ROOT / "data" / "paper_task_bundles_full_eval"),
                "STORE_ROOT": "/home/panzihang/contiguous_fuxian_ssd/qwen25_7b_paper_seed42",
                "KV_DIR": "/home/panzihang/contiguous_fuxian_ssd/paper4_online_contig_c16_v14",
                "DEFER_CACHE_SCORE_UPDATES": "false",
                "PROMIXED_ADAPTIVE_COVERAGE": "false",
                **environment_updates,
            }
        )
        return subprocess.run(
            [str(RUNNER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_runner_has_valid_bash_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_impress_case_uses_canonical_sync_reorder_contract(self):
        self.assertIn(
            "paper4_online_impress_c64_gqa_unique_reordered_disjoint_v33",
            self.script,
        )
        self.assertIn(
            "qwen25_7b_paper4_disjoint_history32_35_v2.json",
            self.script,
        )
        expected = (
            'PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_impress.json"',
            'KV_DIR="$IMPRESS_KV_DIR"',
            "--probe-query-heads 0,1,2",
            "--selector-kv-head-ids 0",
            "--similarity-alpha 0.6",
            '--impress-reorder-manifest "$selected_reorder_manifest"',
            "--no-impress-async-prefetch",
        )
        for fragment in expected:
            self.assertIn(fragment, self.impress_case)
        self.assertIn(
            '[[ "$SELECTOR_BACKEND" != "fp16" ]]',
            self.impress_case,
        )
        self.assertNotIn("--selector-index-dir", self.impress_case)

    def test_protocol_fingerprints_plan_kv_store_and_reorder(self):
        for field in (
            "plan_sha256",
            "bundle_dir",
            "bundle_metadata",
            "bundle_metadata_sha256",
            "store_root",
            "store_task_metadata",
            "store_task_metadata_sha256",
            "kv_complete_sha256",
            "impress_reorder_manifest",
            "impress_reorder_sha256",
            "kv_complete_impress_reorder_sha256",
        ):
            self.assertIn(field, self.script)

    def test_as_lru_case_is_original_full_kv_lru_contract(self):
        expected = (
            'PLAN="$ROOT/configs/qwen25_k005_impress.json"',
            'KV_DIR="$AS_KV_DIR"',
            'CACHE_TYPE="LRU"',
            'AS_BASELINE_MODE="as_lru"',
            'BUDGET_SEMANTICS="full-kv-budget-independent"',
            'AS_FULL_KEY_RATIO="1.0"',
            'AS_VALUE_KEEP_RATIO="1.0"',
            'AS_LOGICAL_TOTAL_KV_RATIO="1.0"',
            '--as-baseline-mode "$AS_BASELINE_MODE"',
            '--probe-query-heads 0,7,14,21',
            '--selector-kv-head-ids 0,1,2,3',
        )
        for fragment in expected:
            self.assertIn(fragment, self.as_lru_case)
        for forbidden in (
            "--selector-index-dir",
            "--impress-async-prefetch",
            "--impress-reorder-manifest",
            "--promixed-",
            "CKLFU",
        ):
            self.assertNotIn(forbidden, self.as_lru_case)

    def test_as_h2o_lru_case_is_full_key_selected_value_lru_contract(self):
        expected = (
            'PLAN="$ROOT/configs/qwen25_k${BUDGET_TAG}_impress.json"',
            'KV_DIR="$AS_KV_DIR"',
            'CACHE_TYPE="LRU"',
            'AS_BASELINE_MODE="as_h2o_lru"',
            'BUDGET_SEMANTICS="full-keys-selected-values"',
            'AS_FULL_KEY_RATIO="1.0"',
            'AS_VALUE_KEEP_RATIO="$KEEP_RATIO"',
            '005) AS_LOGICAL_TOTAL_KV_RATIO="0.525"',
            '010) AS_LOGICAL_TOTAL_KV_RATIO="0.55"',
            '025) AS_LOGICAL_TOTAL_KV_RATIO="0.625"',
            '050) AS_LOGICAL_TOTAL_KV_RATIO="0.75"',
            '--as-baseline-mode "$AS_BASELINE_MODE"',
            '--probe-query-heads 0,7,14,21',
            '--selector-kv-head-ids 0,1,2,3',
        )
        for fragment in expected:
            self.assertIn(fragment, self.as_h2o_lru_case)
        for forbidden in (
            "--selector-index-dir",
            "--impress-async-prefetch",
            "--impress-reorder-manifest",
            "--promixed-",
            "CKLFU",
        ):
            self.assertNotIn(forbidden, self.as_h2o_lru_case)

    def test_as_completion_marker_contract_is_strict_and_plain_chunk64(self):
        expected = (
            '"schema_version": 3',
            '"method": "attentionstore_as_baselines"',
            '"chunk_size": 64',
            '"selector_kv_head_ids": [0, 1, 2, 3]',
            '"online_selection": True',
            '"physical_layout": "plain-logical-token-order"',
            '"impress_reorder_sha256": None',
            '"registered_store_tasks": ["sst2", "subj", "trec", "rte"]',
        )
        for fragment in expected:
            self.assertIn(fragment, self.script)
        self.assertIn('--cache-type "$CACHE_TYPE"', self.script)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_as_lru_rejects_k4_before_gpu_or_storage_work(self):
        with tempfile.TemporaryDirectory() as directory:
            selector_index = Path(directory)
            (selector_index / "manifest.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            completed = self.run_runner(
                METHOD="as_lru",
                BUDGET_TAG="full",
                SELECTOR_BACKEND="k4",
                SELECTOR_INDEX=str(selector_index),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("AS+LRU requires SELECTOR_BACKEND=fp16", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_as_h2o_rejects_full_budget_label(self):
        completed = self.run_runner(
            METHOD="as_h2o_lru",
            BUDGET_TAG="full",
            SELECTOR_BACKEND="fp16",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("BUDGET_TAG=full is only valid", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_as_lru_accepts_full_label_before_bundle_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self.run_runner(
                METHOD="as_lru",
                BUDGET_TAG="full",
                SELECTOR_BACKEND="fp16",
                BUNDLE_DIR=directory,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Bundle metadata is missing", completed.stderr)
        self.assertNotIn("Unsupported budget tag", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_missing_bundle_metadata_is_rejected_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self.run_runner(BUNDLE_DIR=directory)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Bundle metadata is missing", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_missing_store_task_metadata_is_rejected_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self.run_runner(STORE_ROOT=directory)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Store task metadata is missing", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_impress_rejects_reorder_sha_mismatch_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kv_dir = root / "kv"
            kv_dir.mkdir()
            (kv_dir / ".contiguous_fuxian_complete").write_text(
                '{"impress_reorder_sha256": "' + "0" * 64 + '"}\n',
                encoding="utf-8",
            )
            reorder = root / "reorder.json"
            reorder.write_text("{}\n", encoding="utf-8")
            completed = self.run_runner(
                METHOD="impress",
                SELECTOR_BACKEND="fp16",
                IMPRESS_KV_DIR=str(kv_dir),
                IMPRESS_REORDER=str(reorder),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("IMPRESS reorder SHA mismatch", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_impress_accepts_matching_reorder_sha_before_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kv_dir = root / "kv"
            kv_dir.mkdir()
            reorder = root / "reorder.json"
            reorder.write_text("{}\n", encoding="utf-8")
            reorder_sha256 = hashlib.sha256(reorder.read_bytes()).hexdigest()
            (kv_dir / ".contiguous_fuxian_complete").write_text(
                '{"impress_reorder_sha256": "' + reorder_sha256 + '"}\n',
                encoding="utf-8",
            )
            run_root = root / "results"
            run_name = "k010_impress_fp16_nodefer"
            (run_root / "trec" / run_name).mkdir(parents=True)
            completed = self.run_runner(
                METHOD="impress",
                SELECTOR_BACKEND="fp16",
                IMPRESS_KV_DIR=str(kv_dir),
                IMPRESS_REORDER=str(reorder),
                RUN_ROOT=str(run_root),
                RUN_NAME=run_name,
            )
        self.assertEqual(completed.returncode, 4, completed.stderr)
        self.assertIn("Output already exists", completed.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_impress_rejects_k4_before_gpu_or_storage_work(self):
        with tempfile.TemporaryDirectory() as directory:
            selector_index = Path(directory)
            (selector_index / "manifest.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            completed = self.run_runner(
                METHOD="impress",
                SELECTOR_BACKEND="k4",
                SELECTOR_INDEX=str(selector_index),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires SELECTOR_BACKEND=fp16", completed.stderr)
