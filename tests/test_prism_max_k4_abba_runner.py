import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_prism_max_k4_abba_diagnostics.sh"


class PrismMaxK4AbbaRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = RUNNER.read_bytes()
        cls.script = cls.payload.decode("utf-8")

    def test_shell_file_is_lf_only(self):
        self.assertNotIn(b"\r", self.payload)

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

    def test_exact_four_cell_abba_schedule(self):
        self.assertIn(
            'CELLS=("subj 025" "subj 050" "trec 010" "trec 050")',
            self.script,
        )
        self.assertIn(
            'ABBA=("contigkv r1" "promixed r1" "promixed r2" "contigkv r2")',
            self.script,
        )
        self.assertIn('if len(runs) != 16:', self.script)

    def test_matched_clean_protocol_is_explicit(self):
        for fragment in (
            "SELECTOR_BACKEND=k4",
            "DEFER_CACHE_SCORE_UPDATES=false",
            "WARMUP_PASSES=1",
            "WARMUP_SAMPLES_PER_TASK=32",
            "PROMIXED_ADAPTIVE_COVERAGE=false",
            "GPU_CACHE_MB=55 CPU_CACHE_MB=131",
            "--bundle-metadata",
            "--backend k4 --score-mode nodefer",
            "primary_latency=response_ready_ms",
        ):
            self.assertIn(fragment, self.script)


if __name__ == "__main__":
    unittest.main()
