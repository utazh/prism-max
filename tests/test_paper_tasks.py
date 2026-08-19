import unittest

from contiguous_fuxian.paper_tasks import (
    PAPER_TASKS,
    build_task_records,
    format_example,
    length_matched_stratified_sample,
    stratified_sample,
)


class PaperTasksTest(unittest.TestCase):
    def test_stratified_sample_cycles_deterministically_through_labels(self):
        rows = [
            {"text": "a", "label": "negative"},
            {"text": "b", "label": "negative"},
            {"text": "c", "label": "positive"},
            {"text": "d", "label": "positive"},
        ]

        sample = stratified_sample(rows, count=4, seed=42)

        self.assertEqual([row["label"] for row in sample], ["negative", "positive", "negative", "positive"])

    def test_build_task_records_uses_paper_fewshot_count_and_shared_prefix(self):
        train = [
            *({"text": f"negative {index}", "label": "negative"} for index in range(60)),
            *({"text": f"positive {index}", "label": "positive"} for index in range(60)),
        ]
        evaluation = [
            {"text": "test positive", "label": "positive"},
            {"text": "test negative", "label": "negative"},
        ]

        prefix, records = build_task_records("sst2", train, evaluation, eval_samples=2)

        self.assertEqual(prefix.count("Sentiment:"), PAPER_TASKS["sst2"].fewshot_examples)
        self.assertTrue(prefix.startswith("<|im_start|>user\n"))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["prefix_text"] == prefix for record in records))
        self.assertTrue(all(record["query_text"].endswith("<|im_start|>assistant\n") for record in records))

    def test_rte_template_contains_premise_hypothesis_and_answer_slot(self):
        example = format_example(
            "rte",
            {"premise": "P", "hypothesis": "H", "label": "entailment"},
            include_label=False,
        )

        self.assertEqual(example, "Premise: P\nHypothesis: H\nRelation:\n\n")

    def test_length_matched_sample_preserves_label_balance(self):
        rows = [
            *({"text": "a" * index, "label": "negative"} for index in range(1, 8)),
            *({"text": "b" * index, "label": "positive"} for index in range(1, 8)),
        ]

        sample = length_matched_stratified_sample(
            "sst2", rows, count=6, target_prefix_tokens=100, token_count=len, seed=42
        )

        self.assertEqual(len(sample), 6)
        self.assertEqual(sum(row["label"] == "negative" for row in sample), 3)
        self.assertEqual(sum(row["label"] == "positive" for row in sample), 3)


if __name__ == "__main__":
    unittest.main()
