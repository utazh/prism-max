import json
from pathlib import Path

import numpy as np
import pytest

from contiguous_fuxian.quantized_key_index import (
    INDEX_FORMAT,
    QuantizedKeyIndex,
    dequantize_int4_numpy,
    pack_signed_int4,
    quantize_symmetric_int4,
    unpack_signed_int4,
)


def test_signed_int4_pack_round_trip():
    values = np.array([[[-7, -1, 0, 1, 6, 7]]], dtype=np.int8)
    codes = pack_signed_int4(values)
    assert codes.shape == (1, 1, 3)
    assert np.array_equal(unpack_signed_int4(codes, width=6), values)


def test_symmetric_int4_quantization_preserves_zero_and_shape():
    values = np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, -0.5, 0.25, -1.0]],
            [[0.1, 0.2, 0.3, 0.4], [-2.0, 1.0, -1.0, 2.0]],
        ],
        dtype=np.float16,
    )
    codes, scales = quantize_symmetric_int4(values)
    restored = dequantize_int4_numpy(codes, scales, head_dim=4)
    assert codes.shape == (2, 2, 2)
    assert scales.shape == (2, 2, 1)
    assert np.array_equal(restored[0, 0], np.zeros(4, dtype=np.float32))
    assert np.max(np.abs(restored - values.astype(np.float32))) <= 2.0 / 7.0


def test_quantized_key_index_validates_and_loads_cpu(tmp_path: Path):
    pytest.importorskip("torch")
    task_dir = tmp_path / "trec"
    task_dir.mkdir()
    values = np.array(
        [
            [[-1.0, -0.5, 0.5, 1.0], [0.25, -0.25, 0.5, -0.5]],
            [[2.0, 1.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float16,
    )
    codes, scales = quantize_symmetric_int4(values, group_size=2)
    codes.tofile(task_dir / "layer_00.codes.u8")
    scales.tofile(task_dir / "layer_00.scales.f16")
    manifest = {
        "format": INDEX_FORMAT,
        "bits": 4,
        "group_size": 2,
        "selector_kv_head_ids": [0, 1],
        "tasks": {
            "trec": {
                "directory": "trec",
                "prefix_tokens": 2,
                "layers": 1,
                "kv_heads": 2,
                "head_dim": 4,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    index = QuantizedKeyIndex(tmp_path)
    index.validate_task(
        "trec", prefix_tokens=2, layers=1, kv_heads=2, head_dim=4
    )
    loaded, compressed_bytes, source = index.load_layer(
        "trec", 0, device="cpu"
    )
    expected = dequantize_int4_numpy(codes, scales, head_dim=4, group_size=2)
    assert np.allclose(loaded.numpy(), expected, atol=1e-3)
    assert compressed_bytes == codes.nbytes + scales.nbytes
    assert source == "disk"

    assert index.preload_task("trec") == codes.nbytes + scales.nbytes
    assert index.preload_task("trec") == 0
    assert index.task_is_preloaded("trec")
    cached, cached_bytes, cached_source = index.load_layer(
        "trec", 0, device="cpu"
    )
    assert np.allclose(cached.numpy(), expected, atol=1e-3)
    assert cached_bytes == compressed_bytes
    assert cached_source == "cpu"



def test_quantized_key_index_rejects_directory_escape(tmp_path: Path):
    manifest = {
        "format": INDEX_FORMAT,
        "bits": 4,
        "selector_kv_head_ids": [0],
        "tasks": {
            "trec": {
                "directory": "../outside",
                "prefix_tokens": 1,
                "layers": 1,
                "kv_heads": 1,
                "head_dim": 4,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        QuantizedKeyIndex(tmp_path)
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("directory traversal should be rejected")
