import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "plot_datasetwise_matrix.py"
SPEC = importlib.util.spec_from_file_location("plot_datasetwise_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DatasetwisePlotTest(unittest.TestCase):
    def test_loader_rejects_pooled_result_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"overall": {}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pooled"):
                MODULE.load_results(path)

    def test_required_metric_uses_task_budget_and_method(self):
        metric = {"accuracy": 0.9}
        results = {"sst2": {"005_ours": metric}}

        self.assertIs(MODULE.require_metric(results, "sst2", 5, "ours"), metric)
        with self.assertRaisesRegex(ValueError, "sst2/010_ours"):
            MODULE.require_metric(results, "sst2", 10, "ours")


if __name__ == "__main__":
    unittest.main()
