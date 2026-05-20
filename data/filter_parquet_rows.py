from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def build_keep_mask(values, *, drop_eq: float | None, drop_le: float | None):
    if drop_eq is not None:
        keep = pc.not_equal(values, drop_eq)
    elif drop_le is not None:
        keep = pc.greater(values, drop_le)
    else:
        raise ValueError("one of drop_eq or drop_le must be provided")
    return pc.fill_null(keep, True)


def filter_file(
    file_path: Path,
    *,
    column: str,
    drop_eq: float | None,
    drop_le: float | None,
    compression: str,
    dry_run: bool,
) -> tuple[int, int]:
    table = pq.read_table(file_path)
    if column not in table.column_names:
        raise KeyError(f"{file_path} missing column: {column}")

    before = len(table)
    keep_mask = build_keep_mask(table[column], drop_eq=drop_eq, drop_le=drop_le)
    filtered = table.filter(keep_mask)
    after = len(filtered)

    if not dry_run and after < before:
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        pq.write_table(filtered, tmp_path, compression=compression)
        tmp_path.replace(file_path)

    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter parquet rows in a directory.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing parquet files.")
    parser.add_argument("--column", required=True, help="Numeric column to filter on.")
    parser.add_argument("--drop-eq", type=float, default=None, help="Drop rows where column == value.")
    parser.add_argument("--drop-le", type=float, default=None, help="Drop rows where column <= value.")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N parquet files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print files even when no rows are removed.")
    args = parser.parse_args()

    if (args.drop_eq is None) == (args.drop_le is None):
        raise ValueError("exactly one of --drop-eq or --drop-le must be provided")

    files = sorted(args.root.glob("*.parquet"))
    if args.limit is not None:
        files = files[: args.limit]

    total_before = 0
    total_after = 0
    for file_path in files:
        before, after = filter_file(
            file_path,
            column=args.column,
            drop_eq=args.drop_eq,
            drop_le=args.drop_le,
            compression=args.compression,
            dry_run=args.dry_run,
        )
        total_before += before
        total_after += after
        if args.verbose or before != after:
            print(f"{file_path.name}: {before} -> {after}")

    print(f"Files processed: {len(files)}")
    print(f"Total before: {total_before}")
    print(f"Total after: {total_after}")
    print(f"Removed: {total_before - total_after}")


if __name__ == "__main__":
    main()