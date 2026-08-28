from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_as_pcache_store.sh"


def test_builder_is_plain_chunk64_attentionstore_contract():
    text = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        'qwen25_k005_impress.json',
        '--as-baseline-mode as_lru',
        '--cache-type LRU',
        '--gpu-cache-mb 0',
        '--cpu-cache-mb 0',
        '--selector-kv-head-ids 0,1,2,3',
        '--store-tasks sst2,subj,trec,rte',
        '"schema_version": 3',
        '"method": "attentionstore_as_baselines"',
        '"chunk_size": 64',
        '"physical_layout": "plain-logical-token-order"',
        '"impress_reorder_sha256": None',
    ):
        assert fragment in text
    for forbidden in (
        '--selector-index-dir',
        '--impress-reorder-manifest',
        '--promixed-gqa-selection',
        '--cache-type CKLFU',
    ):
        assert forbidden not in text


def test_builder_only_marks_completion_after_successful_run():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.index('[[ -s "$attempt/summary.json" ]]') < text.index(
        '"build_semantics": "fresh Qwen FP16 K/V payload; no IMPRESS physical reorder"'
    )
    assert 'os.replace(temporary, path)' in text
