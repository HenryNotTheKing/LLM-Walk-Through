"""Prepare KodCode-V1-SFT-R1 for bench-aligned Walkie SFT."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Sequence

import pyarrow.parquet as pq

from posttrain.data.kodcode_sft import build_bench_text_index, clean_kodcode_row


READ_COLUMNS = [
    "style",
    "subset",
    "question_id",
    "question",
    "r1_correctness",
    "r1_solution",
    "gpt_difficulty",
    "metadata",
    "test_info",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean KodCode SFT data into minimal bench-aligned JSONL shards")
    parser.add_argument("--input", default="data/sft/KodCode-V1-SFT-R1/data")
    parser.add_argument("--bench-root", default="data/bench")
    parser.add_argument("--output", default="data/sft/kodcode_v1_sft_r1_bench_aligned")
    parser.add_argument("--split", default="train", choices=["train", "incorrect", "use_with_caution"])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-online-judge", action="store_true")
    parser.add_argument("--skip-leak-filter", action="store_true")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input)
    output_root = Path(args.output)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_root}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.glob(f"{args.split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found for split {args.split!r} in {input_root}")

    bench_index = {} if args.skip_leak_filter else build_bench_text_index(args.bench_root)
    allowed_styles = ("instruct", "complete", "online judge") if args.include_online_judge else ("instruct", "complete")
    stats: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    records: list[str] = []

    leakage_writer = (output_root / "leakage_removed.jsonl").open("w", encoding="utf-8")
    try:
        for path in files:
            parquet = pq.ParquetFile(path)
            available_columns = [column for column in READ_COLUMNS if column in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=max(1, int(args.batch_size)), columns=available_columns):
                for row in batch.to_pylist():
                    stats["seen"] += 1
                    decision = clean_kodcode_row(
                        row,
                        bench_index=bench_index,
                        allowed_styles=allowed_styles,
                        drop_leaks=not args.skip_leak_filter,
                    )
                    if decision.record is None:
                        stats[decision.reason] += 1
                        if decision.reason.startswith(("leak:", "near_leak:")):
                            leakage_writer.write(json.dumps({"reason": decision.reason, "question_id": row.get("question_id")}, ensure_ascii=False) + "\n")
                        continue

                    records.append(json.dumps(decision.record, ensure_ascii=False))
                    stats["kept"] += 1
                    style_counts[str(decision.record["style"])] += 1
                    subset_counts[str(decision.record["subset"])] += 1
                    difficulty_counts[str(decision.record["gpt_difficulty"])] += 1
                    if args.limit is not None and stats["kept"] >= int(args.limit):
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        leakage_writer.close()

    if args.shuffle:
        _shuffle_records(records, int(args.seed))
    output_files = _write_shards(output_root, records, int(args.shard_size))

    manifest = {
        "source": str(input_root),
        "split": args.split,
        "bench_root": str(args.bench_root),
        "output_files": output_files,
        "allowed_styles": list(allowed_styles),
        "leak_filter": not args.skip_leak_filter,
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed) if args.shuffle else None,
        "stats": dict(stats),
        "style_counts": dict(style_counts),
        "subset_counts": dict(subset_counts),
        "difficulty_counts": dict(difficulty_counts),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _shuffle_records(records: list[str], seed: int) -> None:
    random.Random(seed).shuffle(records)


def _write_shards(output_root: Path, records: Sequence[str], shard_size: int) -> list[str]:
    output_files: list[str] = []
    shard_size = max(1, shard_size)
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        path = output_root / f"train-{shard_index:05d}.jsonl"
        chunk = records[start : start + shard_size]
        path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        output_files.append(str(path))
    return output_files


if __name__ == "__main__":
    raise SystemExit(main())