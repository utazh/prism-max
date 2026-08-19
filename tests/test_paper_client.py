import unittest

from contiguous_fuxian.paper_client import (
    continuation_token_ids,
    label_token_ids,
    normalize_label,
    percentile95,
    predict_from_label_logits,
    predict_from_label_token_logprobs,
    prediction_is_correct,
)
from contiguous_fuxian.paper_compare import compare_summaries


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    ids = {
        "positive": [2],
        "negative": [5],
        "entailment": [7, 8],
        "not_entailment": [11, 12, 13],
        " duplicate_a": [17],
        " duplicate_b": [17, 18],
    }

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return _Tokenized(self.ids[text])


class _Score:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class PaperClientTest(unittest.TestCase):
    def test_label_normalization_and_prefix_match_handle_short_generations(self):
        self.assertEqual(normalize_label("Type: LOC!"), "type loc")
        self.assertTrue(prediction_is_correct("positive\nExplanation", "positive"))
        self.assertTrue(prediction_is_correct("Type: HUM", "HUM"))
        self.assertTrue(prediction_is_correct("Subjectivity: subjective", "subjective"))
        self.assertFalse(prediction_is_correct("negative", "positive"))

    def test_p95_uses_nearest_rank(self):
        self.assertEqual(percentile95([1, 2, 3, 4]), 4.0)

    def test_label_logit_scoring_uses_only_legal_first_tokens(self):
        logits = [_Score(0.0) for _ in range(20)]
        logits[2] = _Score(1.5)
        logits[5] = _Score(0.7)

        prediction, scores, token_ids = predict_from_label_logits(
            logits,
            _Tokenizer(),
            ("positive", "negative"),
        )

        self.assertEqual(prediction, "positive")
        self.assertEqual(scores, {"positive": 1.5, "negative": 0.7})
        self.assertEqual(token_ids, {"positive": (2,), "negative": (5,)})

    def test_label_logit_scoring_accepts_multitoken_labels_with_unique_first_token(self):
        self.assertEqual(
            label_token_ids(
                _Tokenizer(),
                ("entailment", "not_entailment"),
            ),
            {
                "entailment": (7, 8),
                "not_entailment": (11, 12, 13),
            },
        )

    def test_label_logit_scoring_rejects_first_token_collisions(self):
        with self.assertRaisesRegex(ValueError, "distinct first tokens"):
            label_token_ids(
                _Tokenizer(),
                ("duplicate_a", "duplicate_b"),
                continuation_prefix=" ",
            )

    def test_continuation_tokenization_allows_shared_first_tokens(self):
        self.assertEqual(
            continuation_token_ids(
                _Tokenizer(),
                ("duplicate_a", "duplicate_b"),
                continuation_prefix=" ",
            ),
            {"duplicate_a": (17,), "duplicate_b": (17, 18)},
        )

    def test_continuation_scoring_uses_mean_token_logprob(self):
        prediction, scores = predict_from_label_token_logprobs(
            {
                "short": (-0.4,),
                "long": (-0.1, -0.3, -0.2),
            }
        )

        self.assertEqual(prediction, "long")
        self.assertAlmostEqual(scores["short"], -0.4)
        self.assertAlmostEqual(scores["long"], -0.2)

    def test_continuation_scoring_rejects_empty_token_scores(self):
        with self.assertRaisesRegex(ValueError, "no token log-probabilities"):
            predict_from_label_token_logprobs({"empty": ()})

    def test_comparison_is_only_contiguouskv_against_impress(self):
        contig = {"tasks": {"rte": {"accuracy": 1.0, "mean_ttft_ms": 20.0, "p95_ttft_ms": 25.0}}, "overall": {"accuracy": 1.0, "mean_ttft_ms": 20.0, "p95_ttft_ms": 25.0}}
        impress = {"tasks": {"rte": {"accuracy": 0.5, "mean_ttft_ms": 80.0, "p95_ttft_ms": 90.0}}, "overall": {"accuracy": 0.5, "mean_ttft_ms": 80.0, "p95_ttft_ms": 90.0}}

        report = compare_summaries(contig, impress)

        self.assertEqual(report["comparison"], "ContiguousKV versus IMPRESS")
        self.assertEqual(report["overall"]["ttft_speedup_vs_impress"], 4.0)
        self.assertEqual(set(report), {"comparison", "tasks", "overall"})

    def test_comparison_preserves_physical_io_metrics_when_available(self):
        contig = {
            "tasks": {
                "rte": {
                    "samples": 2,
                    "accuracy": 1.0,
                    "mean_ttft_ms": 20.0,
                    "p95_ttft_ms": 25.0,
                    "mean_physical_prefetch_kv_bytes": 100.0,
                    "mean_read_amplification": 1.2,
                    "mean_ssd_prefetch_kv_bytes": 25.0,
                    "mean_total_ssd_read_bytes": 50.0,
                }
            },
            "overall": {"accuracy": 1.0, "mean_ttft_ms": 20.0, "p95_ttft_ms": 25.0},
        }
        impress = {
            "tasks": {
                "rte": {
                    "samples": 2,
                    "accuracy": 1.0,
                    "mean_ttft_ms": 80.0,
                    "p95_ttft_ms": 90.0,
                    "mean_physical_prefetch_kv_bytes": 500.0,
                    "mean_read_amplification": 6.0,
                    "mean_ssd_prefetch_kv_bytes": 400.0,
                    "mean_total_ssd_read_bytes": 500.0,
                }
            },
            "overall": {"accuracy": 1.0, "mean_ttft_ms": 80.0, "p95_ttft_ms": 90.0},
        }

        report = compare_summaries(contig, impress)

        self.assertEqual(report["overall"]["physical_read_reduction_vs_impress"], 5.0)
        self.assertEqual(report["overall"]["ssd_read_reduction_vs_impress"], 16.0)
        self.assertEqual(report["overall"]["total_ssd_read_reduction_vs_impress"], 10.0)
        self.assertEqual(report["tasks"]["rte"]["contiguous_mean_read_amplification"], 1.2)


if __name__ == "__main__":
    unittest.main()
