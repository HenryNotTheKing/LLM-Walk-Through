"""Merge per-dataset code-bench summaries into one aggregate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge code-bench dataset summaries")
    parser.add_argument("--output", required=True, help="Root directory containing per-dataset subdirs")
    parser.add_argument("--pass-at", default="1,4,8", help="Comma-separated pass@k keys to aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    pass_at = [int(item.strip()) for item in str(args.pass_at).split(",") if item.strip()]

    summaries: dict[str, dict[str, Any]] = {}
    for dataset_dir in sorted(output.iterdir()):
        if not dataset_dir.is_dir():
            continue
        summary_path = dataset_dir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            summaries[dataset_dir.name] = payload

    if not summaries:
        raise RuntimeError(f"no dataset summaries found under {output}")

    macro: dict[str, float] = {}
    for k in pass_at:
        key = f"pass@{k}"
        values = [float(item[key]) for item in summaries.values() if key in item]
        macro[key] = float(sum(values) / len(values)) if values else 0.0

    aggregate = {"datasets": summaries, "macro": macro}
    (output / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
