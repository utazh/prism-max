import unittest

from contiguous_fuxian.qwen_attention_probe import (
    build_parser,
    split_token_ids_for_prefill_query,
    validate_probe_args,
)


class QwenAttentionProbeSafetyTest(unittest.TestCase):
    def test_default_device_is_cpu_to_avoid_shared_gpu_interference(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.allow_gpu)
        self.assertEqual(args.max_prompt_tokens, 2048)
        self.assertEqual(args.attention_mode, "prefill_query")
        self.assertEqual(args.query_tail_tokens, 1)
        self.assertEqual(args.attn_implementation, "sdpa")
        self.assertEqual(args.query_attn_implementation, "eager")
        self.assertEqual(args.chunk_load_ms, 0.08)
        self.assertEqual(args.compute_ms, 1.0)

    def test_cuda_requires_explicit_allow_gpu_flag(self):
        args = build_parser().parse_args(["--device", "cuda"])

        with self.assertRaises(ValueError):
            validate_probe_args(args)

    def test_cuda_allowed_when_flag_is_explicit(self):
        args = build_parser().parse_args(["--device", "cuda", "--allow-gpu"])

        validate_probe_args(args)

    def test_split_token_ids_uses_tail_as_query_and_rest_as_prefix(self):
        prefix, query = split_token_ids_for_prefill_query([10, 11, 12, 13], query_tail_tokens=1)

        self.assertEqual(prefix, [10, 11, 12])
        self.assertEqual(query, [13])

    def test_split_token_ids_keeps_at_least_one_prefix_token(self):
        prefix, query = split_token_ids_for_prefill_query([10, 11], query_tail_tokens=4)

        self.assertEqual(prefix, [10])
        self.assertEqual(query, [11])


if __name__ == "__main__":
    unittest.main()
