"""Run the unchanged PRISM CLI with process-local fused-selector hooks."""

from __future__ import annotations


def main() -> int:
    from .fused_selector_integration import install

    install()
    from contiguous_fuxian.flexgen_qwen_reprefill import main as original_main

    return int(original_main())


if __name__ == "__main__":
    raise SystemExit(main())
