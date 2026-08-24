import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_strict_eval_bundle import build_strict_eval_bundle


class StrictEvalBundleTest(unittest.TestCase):
    def _write_source(self, root: Path, rows_by_task):
        source = root / "source"
        source.mkdir()
        task_metadata = {}
        counts = {}
        raw_by_task = {}
        for task, rows in rows_by_task.items():
            raw_rows = [
                json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
                + "\n"
                for row in rows
            ]
            raw = "".join(raw_rows).encode("utf-8")
            (source / f"{task}.jsonl").write_bytes(raw)
            raw_by_task[task] = raw_rows
            counts[task] = len(rows)
            task_metadata[task] = {
                "records": len(rows),
                "evaluation_requests": len(rows),
                "prefix_text_sha256": "preserved-source-field",
                "prefix_fewshot_examples": 3,
            }
        metadata = {
            "schema_version": 3,
            "evaluation_mode": "each dataset is an independent workload",
            "pooled_headline_metrics_allowed": False,
            "evaluation_requests_by_task": counts,
            "tasks": task_metadata,
        }
        metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
        (source / "metadata.json").write_bytes(metadata_bytes)
        return source, metadata_bytes, raw_by_task

    def _write_exclusions(self, root: Path, excluded_by_task):
        path = root / "exclusions.json"
        payload = {
            "schema_version": 1,
            "exclude_uids_by_task": excluded_by_task,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_filters_rows_byte_for_byte_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata_bytes, raw = self._write_source(
                root,
                {
                    "sst2": [
                        {"uid": "sst2-0", "prefix_text": "共享前缀", "value": 0},
                        {"uid": "sst2-1", "prefix_text": "共享前缀", "value": 1},
                        {"uid": "sst2-2", "prefix_text": "共享前缀", "value": 2},
                    ],
                    "trec": [
                        {"uid": "trec-0", "prefix_text": "trec prefix", "value": 0},
                        {"uid": "trec-1", "prefix_text": "trec prefix", "value": 1},
                    ],
                },
            )
            exclusions = self._write_exclusions(
                root,
                {"sst2": ["sst2-0"], "trec": ["trec-0"]},
            )
            output = root / "strict"

            result = build_strict_eval_bundle(
                source_bundle=source,
                output_bundle=output,
                exclusions_manifest=exclusions,
            )

            self.assertEqual(
                (output / "sst2.jsonl").read_text(encoding="utf-8"),
                raw["sst2"][1] + raw["sst2"][2],
            )
            self.assertEqual(
                (output / "trec.jsonl").read_text(encoding="utf-8"),
                raw["trec"][1],
            )
            self.assertEqual(
                result["evaluation_requests_by_task"],
                {"sst2": 2, "trec": 1},
            )
            self.assertEqual(result["tasks"]["sst2"]["records"], 2)
            self.assertEqual(result["tasks"]["sst2"]["evaluation_requests"], 2)
            self.assertEqual(result["tasks"]["sst2"]["strict_source_records"], 3)
            self.assertEqual(
                result["tasks"]["sst2"]["strict_excluded_uids"],
                ["sst2-0"],
            )
            provenance = result["strict_eval_filter"]
            self.assertEqual(
                provenance["excluded_uids_by_task"],
                {"sst2": ["sst2-0"], "trec": ["trec-0"]},
            )
            self.assertEqual(
                provenance["source_metadata_sha256"],
                hashlib.sha256(metadata_bytes).hexdigest(),
            )
            self.assertEqual(
                provenance["output_task_jsonl_sha256"]["trec"],
                hashlib.sha256(raw["trec"][1].encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                result["tasks"]["sst2"]["prefix_text_sha256"],
                "preserved-source-field",
            )

    def test_rejects_a_configured_uid_missing_from_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._write_source(
                root,
                {"trec": [{"uid": "trec-1", "prefix_text": "prefix"}]},
            )
            exclusions = self._write_exclusions(root, {"trec": ["trec-0"]})
            output = root / "strict"

            with self.assertRaisesRegex(ValueError, "missing configured exclusion"):
                build_strict_eval_bundle(
                    source_bundle=source,
                    output_bundle=output,
                    exclusions_manifest=exclusions,
                )
            self.assertFalse(output.exists())

    def test_rejects_duplicate_source_uids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._write_source(
                root,
                {
                    "trec": [
                        {"uid": "trec-0", "prefix_text": "prefix"},
                        {"uid": "trec-0", "prefix_text": "prefix"},
                        {"uid": "trec-1", "prefix_text": "prefix"},
                    ]
                },
            )
            exclusions = self._write_exclusions(root, {"trec": ["trec-0"]})

            with self.assertRaisesRegex(ValueError, "duplicate UID"):
                build_strict_eval_bundle(
                    source_bundle=source,
                    output_bundle=root / "strict",
                    exclusions_manifest=exclusions,
                )

    def test_rejects_duplicate_uids_in_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._write_source(
                root,
                {
                    "trec": [
                        {"uid": "trec-0", "prefix_text": "prefix"},
                        {"uid": "trec-1", "prefix_text": "prefix"},
                    ]
                },
            )
            exclusions = self._write_exclusions(
                root,
                {"trec": ["trec-0", "trec-0"]},
            )

            with self.assertRaisesRegex(ValueError, "duplicate UIDs"):
                build_strict_eval_bundle(
                    source_bundle=source,
                    output_bundle=root / "strict",
                    exclusions_manifest=exclusions,
                )

    def test_rejects_metadata_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self._write_source(
                root,
                {
                    "trec": [
                        {"uid": "trec-0", "prefix_text": "prefix"},
                        {"uid": "trec-1", "prefix_text": "prefix"},
                    ]
                },
            )
            metadata_path = source / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["tasks"]["trec"]["records"] = 3
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            exclusions = self._write_exclusions(root, {"trec": ["trec-0"]})

            with self.assertRaisesRegex(ValueError, "metadata counts"):
                build_strict_eval_bundle(
                    source_bundle=source,
                    output_bundle=root / "strict",
                    exclusions_manifest=exclusions,
                )


if __name__ == "__main__":
    unittest.main()
