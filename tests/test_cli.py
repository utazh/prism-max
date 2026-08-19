import json
import tempfile
import unittest
from pathlib import Path

from contiguous_fuxian.run_reproduction import main


class ContiguousKVCLITest(unittest.TestCase):
    def test_synthetic_cli_writes_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            code = main([
                "synthetic",
                "--prefix-tokens",
                "1024",
                "--num-layers",
                "8",
                "--keep-ratio",
                "0.25",
                "--output",
                str(output),
            ])

            self.assertEqual(code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("speedup_vs_impress", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
