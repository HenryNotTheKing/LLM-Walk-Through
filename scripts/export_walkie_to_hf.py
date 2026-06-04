"""CLI wrapper for exporting Walkie checkpoints to a HF/vLLM directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from posttrain.utils.hf_export import export_walkie_to_hf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Walkie checkpoint for vLLM evaluation")
    parser.add_argument("--checkpoint", default="runs/walkie_code_0.5b/latest.pt")
    parser.add_argument("--output", default="runs/walkie_code_0.5b_vllm_hf")
    parser.add_argument("--tokenizer", default="data/cache/walkie_code/tokenizer.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = export_walkie_to_hf(
        Path(args.checkpoint),
        Path(args.output),
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
    )
    print(f"exported Walkie checkpoint to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())