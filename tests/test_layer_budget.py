import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from contiguous_fuxian.layer_budget import (
    LayerBudgetProfile,
    build_ranked_three_level_profile,
    load_layer_budget_profile,
    profile_to_json_dict,
)


class LayerBudgetLoadTest(unittest.TestCase):
    def _payload(self):
        return {
            "schema_version": 1,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "target_mean_ratio": 0.5,
            "layer_ratios": [0.4, 0.5, 0.6],
            "calibration": {"dataset": "phase-a", "samples": 32},
        }

    def _load_payload(self, payload, expected_layers=3, expected_model=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_layer_budget_profile(
                path,
                expected_layers=expected_layers,
                expected_model=expected_model,
            )

    def test_valid_load_records_exact_file_hash_and_metadata(self):
        payload = self._payload()
        raw = json.dumps(payload, indent=2).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_bytes(raw)
            profile = load_layer_budget_profile(
                path,
                expected_layers=3,
                expected_model=payload["model"],
            )

            self.assertEqual(profile.source_path, str(path))
            self.assertEqual(profile.source_sha256, hashlib.sha256(raw).hexdigest())

        self.assertEqual(profile.layer_ratios, (0.4, 0.5, 0.6))
        self.assertEqual(profile.calibration, payload["calibration"])
        with self.assertRaises(FrozenInstanceError):
            profile.model = "changed"

    def test_rejects_malformed_ratios_and_wrong_mean(self):
        malformed = (
            [0.0, 0.5, 1.0],
            [0.4, 0.5, 1.1],
            [0.4, float("nan"), 0.6],
            [0.4, "0.5", 0.6],
            [0.4, 0.4, 0.4],
        )
        for ratios in malformed:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                payload = self._payload()
                payload["layer_ratios"] = ratios
                self._load_payload(payload)

    def test_rejects_layer_count_and_model_mismatches(self):
        with self.assertRaisesRegex(ValueError, "expected 4"):
            self._load_payload(self._payload(), expected_layers=4)
        with self.assertRaisesRegex(ValueError, "expected_model"):
            self._load_payload(self._payload(), expected_model="other-model")

    def test_rejects_missing_extra_and_malformed_schema_fields(self):
        missing = self._payload()
        del missing["calibration"]
        extra = {**self._payload(), "notes": "unexpected"}
        wrong_version = {**self._payload(), "schema_version": 2}
        bad_calibration = {**self._payload(), "calibration": []}

        for payload in (missing, extra, wrong_version, bad_calibration):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self._load_payload(payload)


class LayerBudgetBuildTest(unittest.TestCase):
    def test_ranking_and_ties_are_deterministic(self):
        tied = build_ranked_three_level_profile(
            [1.0, 1.0, 1.0, 1.0, 1.0],
            0.6,
            0.1,
            "qwen",
            {"method": "tie-test"},
        )
        self.assertEqual(tied.layer_ratios, (0.7, 0.6, 0.6, 0.6, 0.5))

        ranked = build_ranked_three_level_profile(
            [0.1, 0.8, 0.4, 0.9, 0.2],
            0.6,
            0.1,
            "qwen",
            {},
        )
        self.assertEqual(ranked.layer_ratios, (0.5, 0.6, 0.6, 0.7, 0.6))

    def test_odd_layer_count_preserves_the_exact_target_mean(self):
        profile = build_ranked_three_level_profile(
            [0.7, 0.1, 0.9, 0.2, 0.8, 0.3, 0.6],
            0.53,
            0.17,
            "qwen",
            {},
        )

        actual_mean = math.fsum(profile.layer_ratios) / len(profile.layer_ratios)
        self.assertAlmostEqual(actual_mean, profile.target_mean_ratio, places=15)
        self.assertEqual(profile.layer_ratios.count(0.53), 5)
        self.assertTrue(all(0.0 < ratio <= 1.0 for ratio in profile.layer_ratios))

    def test_rejects_invalid_builder_parameters_and_bounds(self):
        valid = ([0.1, 0.2, 0.3], 0.5, 0.1, "qwen", {})
        invalid = (
            ([], *valid[1:]),
            ([0.1, float("inf")], *valid[1:]),
            (valid[0], 0.0, valid[2], valid[3], valid[4]),
            (valid[0], valid[1], 0.0, valid[3], valid[4]),
            (valid[0], 0.95, 0.1, valid[3], valid[4]),
            (valid[0], 0.05, 0.05, valid[3], valid[4]),
            (valid[0], valid[1], valid[2], "", valid[4]),
            (valid[0], valid[1], valid[2], valid[3], []),
        )

        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_ranked_three_level_profile(*arguments)

        for fraction in (0.0, -0.1, 0.51, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_ranked_three_level_profile(*valid, extreme_fraction=fraction)

    def test_even_layer_count_keeps_a_real_middle_group(self):
        profile = build_ranked_three_level_profile(
            list(range(28)),
            0.5,
            0.25,
            "qwen",
            {},
        )

        self.assertEqual(profile.layer_ratios.count(0.75), 7)
        self.assertEqual(profile.layer_ratios.count(0.5), 14)
        self.assertEqual(profile.layer_ratios.count(0.25), 7)

    def test_calibration_metadata_is_deeply_immutable(self):
        profile = build_ranked_three_level_profile(
            [0.1, 0.2, 0.3],
            0.5,
            0.1,
            "qwen",
            {"nested": {"values": [1, 2]}},
        )

        with self.assertRaises(TypeError):
            profile.calibration["new"] = True
        with self.assertRaises(TypeError):
            profile.calibration["nested"]["new"] = True

    def test_profile_json_round_trip_omits_source_fields(self):
        calibration = {"dataset": "phase-a", "seeds": [1, 2]}
        profile = build_ranked_three_level_profile(
            [0.4, 0.9, 0.1, 0.7, 0.3],
            0.5,
            0.2,
            "Qwen/Qwen2.5-7B-Instruct",
            calibration,
        )
        payload = profile_to_json_dict(profile)

        self.assertNotIn("source_path", payload)
        self.assertNotIn("source_sha256", payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round-trip.json"
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            loaded = load_layer_budget_profile(
                path,
                expected_layers=5,
                expected_model=profile.model,
            )

        self.assertEqual(profile_to_json_dict(loaded), payload)
        self.assertIsInstance(loaded, LayerBudgetProfile)


if __name__ == "__main__":
    unittest.main()
