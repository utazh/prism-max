import json
from pathlib import Path

import numpy as np

from prism_gao.mixed_precision_reader import MixedPrecisionPayloadReader
from prism_gao.precision_run_coalescer import DROP, FP16, INT8


def _write_payload(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "prism-ultra-payload-v1",
                "tasks": {
                    "toy": {
                        "prefix_tokens": 5,
                        "layers": 1,
                        "kv_heads": 1,
                        "head_dim": 4,
                        "group_size": 2,
                        "block_size": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    task = root / "toy"
    task.mkdir()
    for tensor, base in (("key", 0), ("value", 100)):
        fp16 = (np.arange(20, dtype=np.float16) + base).reshape(5, 1, 4)
        codes = (np.arange(20, dtype=np.int8) + base % 7).reshape(5, 1, 4)
        scales = np.ones((5, 1, 2), dtype=np.float16)
        stem = task / f"layer_00_{tensor}"
        fp16.tofile(f"{stem}.f16")
        codes.tofile(f"{stem}.int8.codes.i8")
        scales.tofile(f"{stem}.int8.scales.f16")


def test_host_read_preserves_original_selected_block_order(tmp_path):
    _write_payload(tmp_path)
    reader = MixedPrecisionPayloadReader(tmp_path)
    # Physical blocks: 0=INT8, 1=drop, 2=FP16 (partial final block).
    payload = reader.read_kv_host(
        task="toy", layer=0, tiers=(INT8, DROP, FP16)
    )
    assert payload.selected_blocks == (0, 2)
    assert payload.tier_codes == (1, 0)
    assert payload.source_slots == (0, 0)
    assert payload.key.int8.shape == (1, 8)
    assert payload.key.fp16.shape == (1, 8)
    assert payload.key.pread_calls == 3  # INT8 codes/scales + FP16
    assert payload.value.pread_calls == 3
