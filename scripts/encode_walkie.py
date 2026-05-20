from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm.auto import tqdm


TEXT_CANDIDATES = ("text", "content", "markdown", "completion", "output")
DOC_KEY_GROUPS = (
    ("max_stars_repo_name", "max_stars_repo_path"),
    ("repo_name", "path"),
    ("url",),
    ("id",),
    ("blob_id",),
    ("content_id",),
)
DOC_KEY_FALLBACK_FIELDS = (
    "max_stars_repo_name",
    "max_stars_repo_path",
    "repo_name",
    "path",
    "url",
    "id",
    "blob_id",
    "content_id",
)

_WORKER_ENCODE_BATCH = None
_WORKER_EOS_ID: int | None = None
_WORKER_DTYPE: np.dtype | None = None
_WORKER_BATCH_ROWS: int | None = None
_WORKER_WRITE_CHUNK_BYTES: int | None = None
_WORKER_TRUST_TEXT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Walkie pretraining sources into main.bin and anneal.bin."
    )
    parser.add_argument("--datasets-json", type=Path, default=None)
    parser.add_argument("--tasks-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/walkie_code"))
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--main-bin", type=Path, default=None)
    parser.add_argument("--anneal-bin", type=Path, default=None)
    parser.add_argument("--main-val-bin", type=Path, default=None)
    parser.add_argument("--anneal-val-bin", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--shards-dir", type=Path, default=None)
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rayon-threads", type=int, default=2)
    parser.add_argument("--arrow-threads", type=int, default=1)
    parser.add_argument("--write-chunk-mb", type=int, default=64)
    parser.add_argument("--copy-buffer-mb", type=int, default=1)
    parser.add_argument("--main-val-ratio", type=float, default=0.002)
    parser.add_argument("--anneal-val-ratio", type=float, default=0.01)
    parser.add_argument("--val-salt", default="walkie-val-v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--trust-text",
        action="store_true",
        help="Skip per-row type/empty checks. Use only after notebook exploration confirms clean text columns.",
    )
    args = parser.parse_args()
    if (args.datasets_json is None) == (args.tasks_json is None):
        parser.error("exactly one of --datasets-json or --tasks-json must be provided")
    for name, value in (
        ("main", args.main_val_ratio),
        ("anneal", args.anneal_val_ratio),
    ):
        if not 0.0 <= value < 1.0:
            parser.error(f"--{name}-val-ratio must be in [0, 1)")
    return args


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list of dataset configs")
    return payload


def source_files(cache_dir: Path) -> list[Path]:
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.rglob("*.parquet")) + sorted(cache_dir.rglob("*.jsonl"))


def safe_name(value: str) -> str:
    value = value.replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in {"_", ".", "-"} else "_" for ch in value)


def normalize_task(task: dict[str, Any], shards_dir: Path, index: int) -> dict[str, Any]:
    stage = str(task.get("stage", "main"))
    if stage not in {"main", "anneal"}:
        raise ValueError(f"task stage must be main/anneal: {task}")

    src_path = task.get("src_path") or task.get("path")
    if src_path is None:
        raise ValueError(f"task missing src_path/path: {task}")

    src = Path(str(src_path))
    dataset = str(task.get("dataset") or task.get("name") or src.parent.name or "dataset")
    text_field = task.get("text_field")
    row_filter = task.get("row_filter")
    if row_filter is not None and not isinstance(row_filter, dict):
        raise TypeError(f"row_filter must be a dict or null: {task}")
    try:
        size = int(task.get("size", src.stat().st_size))
    except OSError:
        size = int(task.get("size", 0))

    safe_dataset = safe_name(dataset)
    stem = safe_name(src.stem)
    return {
        "dataset": dataset,
        "stage": stage,
        "text_field": text_field,
        "src_path": str(src),
        "size": size,
        "matched_rows": int(task.get("matched_rows", 0) or 0),
        "row_filter": row_filter,
        "train_shard_path": str(
            shards_dir / f"{stage}__train__{safe_dataset}__{index:05d}__{stem}.bin"
        ),
        "val_shard_path": str(
            shards_dir / f"{stage}__val__{safe_dataset}__{index:05d}__{stem}.bin"
        ),
    }


def enumerate_tasks(datasets: list[dict[str, Any]], shards_dir: Path) -> list[dict[str, Any]]:
    raw_tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        stage = dataset.get("stage", "main")
        if stage not in {"main", "anneal"}:
            raise ValueError(f"dataset stage must be main/anneal: {dataset}")
        name = str(dataset.get("name") or dataset.get("repo_id") or "dataset")
        cache_dir = Path(str(dataset["cache_dir"]))
        text_field = dataset.get("text_field")
        for path in source_files(cache_dir):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            raw_tasks.append(
                {
                    "dataset": name,
                    "stage": stage,
                    "text_field": text_field,
                    "src_path": str(path),
                    "size": size,
                    "row_filter": None,
                }
            )
    raw_tasks.sort(key=lambda item: int(item["size"]), reverse=True)
    return [normalize_task(task, shards_dir, index) for index, task in enumerate(raw_tasks)]


def load_tasks(path: Path, shards_dir: Path) -> list[dict[str, Any]]:
    raw_tasks = load_json_list(path)
    raw_tasks.sort(key=lambda item: int(item.get("size", 0)), reverse=True)
    return [normalize_task(task, shards_dir, index) for index, task in enumerate(raw_tasks)]


def infer_text_field(parquet: pq.ParquetFile, src: Path, text_field: str | None) -> str:
    if text_field is not None:
        return text_field
    for candidate in TEXT_CANDIDATES:
        if candidate in parquet.schema_arrow.names:
            return candidate
    raise ValueError(f"{src} cannot infer text_field")


def task_val_ratio(task: dict[str, Any]) -> float:
    return float(task.get("val_ratio", 0.0) or 0.0)


def row_matches_filter(value: Any, row_filter: dict[str, Any] | None) -> bool:
    if row_filter is None:
        return True

    include_null = bool(row_filter.get("include_null", False))
    if value is None:
        return include_null
    if hasattr(value, "item"):
        value = value.item()

    op = row_filter["op"]
    threshold = row_filter["value"]
    if op == "ge":
        keep = value >= threshold
    elif op == "lt":
        keep = value < threshold
    elif op == "eq":
        keep = value == threshold
    elif op == "ne":
        keep = value != threshold
    else:
        raise ValueError(f"unsupported row_filter op: {op}")
    return bool(keep)


def choose_doc_key(row: dict[str, Any], *, src_path: str, row_index: int) -> str:
    for group in DOC_KEY_GROUPS:
        values: list[str] = []
        for field in group:
            value = row.get(field)
            if value is None or value == "":
                values = []
                break
            values.append(f"{field}={value}")
        if values:
            return "|".join(values)

    for field in DOC_KEY_FALLBACK_FIELDS:
        value = row.get(field)
        if value is not None and value != "":
            return f"{field}={value}"

    return f"src={src_path}|row={row_index}"


def is_validation_doc(doc_key: str, *, ratio: float, salt: str) -> bool:
    if ratio <= 0.0:
        return False
    payload = f"{salt}\n{doc_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
    return bucket < ratio


def encode_texts_to_file(texts: list[str], handle, write_buf: bytearray) -> tuple[int, int]:
    if not texts:
        return 0, 0
    encs = _WORKER_ENCODE_BATCH(texts)
    n_tokens, n_docs = pack_encs_to_buffer(encs, write_buf)
    if len(write_buf) >= _WORKER_WRITE_CHUNK_BYTES:
        handle.write(write_buf)
        write_buf.clear()
    return n_tokens, n_docs


def init_encode_worker(
    tokenizer_path: str,
    eos_id: int,
    dtype_name: str,
    batch_rows: int,
    rayon_threads: int,
    arrow_threads: int,
    write_chunk_bytes: int,
    trust_text: bool,
) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    try:
        import pyarrow as _pa

        _pa.set_cpu_count(max(1, arrow_threads))
        _pa.set_io_thread_count(max(1, arrow_threads))
    except Exception:
        pass

    tokenizer = Tokenizer.from_file(tokenizer_path)

    global _WORKER_ENCODE_BATCH, _WORKER_EOS_ID, _WORKER_DTYPE
    global _WORKER_BATCH_ROWS, _WORKER_WRITE_CHUNK_BYTES, _WORKER_TRUST_TEXT
    _WORKER_ENCODE_BATCH = getattr(tokenizer, "encode_batch_fast", tokenizer.encode_batch)
    _WORKER_EOS_ID = int(eos_id)
    _WORKER_DTYPE = np.dtype(dtype_name)
    _WORKER_BATCH_ROWS = int(batch_rows)
    _WORKER_WRITE_CHUNK_BYTES = int(write_chunk_bytes)
    _WORKER_TRUST_TEXT = bool(trust_text)


def pack_encs_to_buffer(encs, write_buf: bytearray) -> tuple[int, int]:
    total = sum(len(enc.ids) + 1 for enc in encs)
    if total == 0:
        return 0, 0

    arr = np.empty(total, dtype=_WORKER_DTYPE)
    pos = 0
    for enc in encs:
        ids = enc.ids
        n = len(ids)
        if n:
            arr[pos : pos + n] = ids
            pos += n
        arr[pos] = _WORKER_EOS_ID
        pos += 1

    write_buf.extend(memoryview(arr).cast("B"))
    return total, len(encs)


def encode_one_file(task: dict[str, Any]) -> dict[str, Any]:
    import json as _json
    import pyarrow.parquet as _pq

    if _WORKER_ENCODE_BATCH is None:
        raise RuntimeError("worker tokenizer is not initialized")

    text_field = task["text_field"]
    src = Path(task["src_path"])
    train_out = Path(task["train_shard_path"])
    val_out = Path(task["val_shard_path"])
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    row_filter = task.get("row_filter")
    val_ratio = task_val_ratio(task)
    val_salt = str(task.get("val_salt", "walkie-val-v1"))

    train_tokens = 0
    train_docs = 0
    val_tokens = 0
    val_docs = 0
    suffix = src.suffix.lower()
    train_buf = bytearray()
    val_buf = bytearray()
    row_index = 0

    with train_out.open("wb") as train_fp, val_out.open("wb") as val_fp:
        if suffix == ".parquet":
            parquet = _pq.ParquetFile(src)
            col = infer_text_field(parquet, src, text_field)
            read_columns = [col]
            if row_filter is not None and row_filter["column"] not in read_columns:
                read_columns.append(str(row_filter["column"]))
            for candidate in DOC_KEY_FALLBACK_FIELDS:
                if candidate in parquet.schema_arrow.names and candidate not in read_columns:
                    read_columns.append(candidate)

            for batch in parquet.iter_batches(batch_size=_WORKER_BATCH_ROWS, columns=read_columns):
                columns = batch.to_pydict()
                texts_train: list[str] = []
                texts_val: list[str] = []
                batch_size = batch.num_rows
                filter_column = row_filter.get("column") if row_filter is not None else None

                for local_index in range(batch_size):
                    text = columns[col][local_index]
                    if not _WORKER_TRUST_TEXT:
                        if not isinstance(text, str) or not text:
                            continue
                    elif not isinstance(text, str):
                        raise TypeError(f"{src} column {col!r} contains non-string value under --trust-text")

                    if filter_column is not None:
                        filter_value = columns[filter_column][local_index]
                        if not row_matches_filter(filter_value, row_filter):
                            continue

                    row = {
                        name: values[local_index]
                        for name, values in columns.items()
                        if name != col
                    }
                    doc_key = choose_doc_key(row, src_path=str(src), row_index=row_index + local_index)
                    if is_validation_doc(doc_key, ratio=val_ratio, salt=val_salt):
                        texts_val.append(text)
                    else:
                        texts_train.append(text)

                batch_tokens, batch_docs = encode_texts_to_file(texts_train, train_fp, train_buf)
                train_tokens += batch_tokens
                train_docs += batch_docs
                batch_tokens, batch_docs = encode_texts_to_file(texts_val, val_fp, val_buf)
                val_tokens += batch_tokens
                val_docs += batch_docs
                row_index += batch_size

        elif suffix == ".jsonl":
            train_texts: list[str] = []
            val_texts: list[str] = []
            with src.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    if isinstance(row, dict):
                        if row_filter is not None and not row_matches_filter(row.get(row_filter["column"]), row_filter):
                            row_index += 1
                            continue
                        if _WORKER_TRUST_TEXT and text_field:
                            value = row[text_field]
                        else:
                            value = row.get(text_field) if text_field else None
                            if not isinstance(value, str) or not value:
                                for candidate in TEXT_CANDIDATES:
                                    candidate_value = row.get(candidate)
                                    if isinstance(candidate_value, str) and candidate_value:
                                        value = candidate_value
                                        break
                    elif isinstance(row, str):
                        value = row
                    else:
                        value = None

                    if _WORKER_TRUST_TEXT and not isinstance(value, str):
                        raise TypeError(f"{src} contains non-string text under --trust-text")
                    if not isinstance(value, str) or (not _WORKER_TRUST_TEXT and not value):
                        row_index += 1
                        continue

                    row_payload = row if isinstance(row, dict) else {}
                    doc_key = choose_doc_key(row_payload, src_path=str(src), row_index=row_index)
                    if is_validation_doc(doc_key, ratio=val_ratio, salt=val_salt):
                        val_texts.append(value)
                    else:
                        train_texts.append(value)

                    if len(train_texts) >= _WORKER_BATCH_ROWS:
                        batch_tokens, batch_docs = encode_texts_to_file(train_texts, train_fp, train_buf)
                        train_tokens += batch_tokens
                        train_docs += batch_docs
                        train_texts.clear()
                    if len(val_texts) >= _WORKER_BATCH_ROWS:
                        batch_tokens, batch_docs = encode_texts_to_file(val_texts, val_fp, val_buf)
                        val_tokens += batch_tokens
                        val_docs += batch_docs
                        val_texts.clear()
                    row_index += 1

            if train_texts:
                batch_tokens, batch_docs = encode_texts_to_file(train_texts, train_fp, train_buf)
                train_tokens += batch_tokens
                train_docs += batch_docs
            if val_texts:
                batch_tokens, batch_docs = encode_texts_to_file(val_texts, val_fp, val_buf)
                val_tokens += batch_tokens
                val_docs += batch_docs
        else:
            raise ValueError(f"unsupported file type: {src}")

        if train_buf:
            train_fp.write(train_buf)
        if val_buf:
            val_fp.write(val_buf)

    return {
        "dataset": task["dataset"],
        "stage": task["stage"],
        "src_path": task["src_path"],
        "row_filter": task.get("row_filter"),
        "train_shard_path": str(train_out),
        "val_shard_path": str(val_out),
        "train_n_tokens": train_tokens,
        "train_n_docs": train_docs,
        "val_n_tokens": val_tokens,
        "val_n_docs": val_docs,
    }


def concat_shards(
    results: list[dict[str, Any]],
    output_paths: dict[tuple[str, str], Path],
    buffer_size: int,
) -> None:
    results_sorted = sorted(results, key=lambda row: (row["stage"], row["dataset"], row["src_path"]))
    handles = {key: path.open("ab") for key, path in output_paths.items()}
    try:
        for row in tqdm(results_sorted, desc="concat shards", unit="shard"):
            for split in ("train", "val"):
                n_tokens = int(row[f"{split}_n_tokens"])
                if n_tokens <= 0:
                    continue
                shard_path = Path(row[f"{split}_shard_path"])
                handle_key = (row["stage"], split)
                with shard_path.open("rb") as src_fp:
                    shutil.copyfileobj(src_fp, handles[handle_key], length=buffer_size)
    finally:
        for handle in handles.values():
            handle.close()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    tokenizer_dir = args.tokenizer_dir or output_dir / "tokenizer.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    main_bin = args.main_bin or output_dir / "main.bin"
    anneal_bin = args.anneal_bin or output_dir / "anneal.bin"
    main_val_bin = args.main_val_bin or output_dir / "main_val.bin"
    anneal_val_bin = args.anneal_val_bin or output_dir / "anneal_val.bin"
    meta_path = args.meta or output_dir / "data_meta.json"
    split_manifest = args.split_manifest or output_dir / "split_manifest.jsonl"
    shards_dir = args.shards_dir or output_dir / "intermediate" / "shards"
    dtype = np.dtype(args.dtype)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_token = "<|endoftext|>"
    pad_token = "<|pad|>"
    eos_id = tokenizer.token_to_id(eos_token)
    pad_id = tokenizer.token_to_id(pad_token)
    if eos_id is None:
        raise RuntimeError(f"tokenizer missing {eos_token}")
    vocab_size = tokenizer.get_vocab_size()
    dtype_limit = int(np.iinfo(dtype).max)
    if vocab_size - 1 > dtype_limit:
        raise RuntimeError(
            f"vocab_size={vocab_size} cannot fit in dtype={dtype}; max id is {dtype_limit}"
        )

    input_mode = "tasks_json" if args.tasks_json is not None else "datasets_json"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        for path in (main_bin, anneal_bin, main_val_bin, anneal_val_bin, meta_path, split_manifest):
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in (main_bin, anneal_bin, main_val_bin, anneal_val_bin)):
        raise FileExistsError("bin files already exist; pass --force to overwrite")

    if args.tasks_json is not None:
        tasks = load_tasks(args.tasks_json, shards_dir)
        source_payload: list[dict[str, Any]] = load_json_list(args.tasks_json)
    else:
        source_payload = load_json_list(args.datasets_json)
        tasks = enumerate_tasks(source_payload, shards_dir)
    if not tasks:
        raise RuntimeError("no parquet/jsonl files found from input spec")

    for task in tasks:
        task["val_ratio"] = args.anneal_val_ratio if task["stage"] == "anneal" else args.main_val_ratio
        task["val_salt"] = args.val_salt

    workers = args.workers or max(1, min(len(tasks), (os.cpu_count() or 4) // args.rayon_threads))
    write_chunk_bytes = max(1, args.write_chunk_mb) * (1 << 20)
    task_payloads = []
    for task in tasks:
        payload = dict(task)
        payload.update(
            {
                "tokenizer_path": str(tokenizer_path),
                "eos_id": int(eos_id),
                "dtype": str(dtype),
                "batch_rows": int(args.batch_rows),
                "rayon_threads": int(args.rayon_threads),
                "trust_text": bool(args.trust_text),
            }
        )
        task_payloads.append(payload)

    print(
        f"workers={workers} rayon_threads/worker={args.rayon_threads} "
        f"tasks={len(task_payloads)} dtype={dtype} vocab={vocab_size} eos_id={eos_id} "
        f"trust_text={args.trust_text} main_val_ratio={args.main_val_ratio} "
        f"anneal_val_ratio={args.anneal_val_ratio}"
    )

    results: list[dict[str, Any]] = []
    done_tokens = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_encode_worker,
        initargs=(
            str(tokenizer_path),
            int(eos_id),
            str(dtype),
            int(args.batch_rows),
            int(args.rayon_threads),
            int(args.arrow_threads),
            int(write_chunk_bytes),
            bool(args.trust_text),
        ),
    ) as pool:
        futures = {pool.submit(encode_one_file, task): task for task in task_payloads}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="encode", unit="file")
        for future in pbar:
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERR] {futures[future]['src_path']}: {exc}")
                raise
            results.append(result)
            done_tokens += int(result["train_n_tokens"]) + int(result["val_n_tokens"])
            pbar.set_postfix(tokens=f"{done_tokens:,}")

    concat_shards(
        results,
        {
            ("main", "train"): main_bin,
            ("anneal", "train"): anneal_bin,
            ("main", "val"): main_val_bin,
            ("anneal", "val"): anneal_val_bin,
        },
        buffer_size=max(1, args.copy_buffer_mb) * (1 << 20),
    )

    itemsize = dtype.itemsize
    expected_main = sum(row["train_n_tokens"] for row in results if row["stage"] == "main") * itemsize
    expected_anneal = sum(row["train_n_tokens"] for row in results if row["stage"] == "anneal") * itemsize
    expected_main_val = sum(row["val_n_tokens"] for row in results if row["stage"] == "main") * itemsize
    expected_anneal_val = sum(row["val_n_tokens"] for row in results if row["stage"] == "anneal") * itemsize
    if main_bin.stat().st_size != expected_main:
        raise RuntimeError(f"main.bin size mismatch: {main_bin.stat().st_size} != {expected_main}")
    if anneal_bin.stat().st_size != expected_anneal:
        raise RuntimeError(
            f"anneal.bin size mismatch: {anneal_bin.stat().st_size} != {expected_anneal}"
        )
    if main_val_bin.stat().st_size != expected_main_val:
        raise RuntimeError(
            f"main_val.bin size mismatch: {main_val_bin.stat().st_size} != {expected_main_val}"
        )
    if anneal_val_bin.stat().st_size != expected_anneal_val:
        raise RuntimeError(
            f"anneal_val.bin size mismatch: {anneal_val_bin.stat().st_size} != {expected_anneal_val}"
        )

    manifest_rows: list[dict[str, Any]] = []
    for row in sorted(results, key=lambda item: (item["stage"], item["dataset"], item["src_path"])):
        for split in ("train", "val"):
            n_tokens = int(row[f"{split}_n_tokens"])
            n_docs = int(row[f"{split}_n_docs"])
            if n_tokens <= 0 and n_docs <= 0:
                continue
            manifest_rows.append(
                {
                    "dataset": row["dataset"],
                    "stage": row["stage"],
                    "split": split,
                    "src_path": row["src_path"],
                    "shard_path": row[f"{split}_shard_path"],
                    "n_tokens": n_tokens,
                    "n_docs": n_docs,
                    "row_filter": row.get("row_filter"),
                }
            )
    write_jsonl(split_manifest, manifest_rows)

    dataset_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in results:
        for split in ("train", "val"):
            key = (row["stage"], split, row["dataset"])
            stats = dataset_stats.setdefault(
                key,
                {
                    "stage": row["stage"],
                    "split": split,
                    "dataset": row["dataset"],
                    "n_tokens": 0,
                    "n_docs": 0,
                },
            )
            stats["n_tokens"] += int(row[f"{split}_n_tokens"])
            stats["n_docs"] += int(row[f"{split}_n_docs"])

    stage_totals: dict[str, dict[str, int]] = {
        "main": {"train_tokens": 0, "val_tokens": 0, "train_docs": 0, "val_docs": 0},
        "anneal": {"train_tokens": 0, "val_tokens": 0, "train_docs": 0, "val_docs": 0},
    }
    for stats in dataset_stats.values():
        split_key = "train" if stats["split"] == "train" else "val"
        stage_totals[stats["stage"]][f"{split_key}_tokens"] += int(stats["n_tokens"])
        stage_totals[stats["stage"]][f"{split_key}_docs"] += int(stats["n_docs"])

    total_tokens = sum(
        values["train_tokens"] + values["val_tokens"] for values in stage_totals.values()
    )
    per_dataset_stats = []
    for stats in sorted(
        dataset_stats.values(), key=lambda item: (item["stage"], item["split"], -item["n_tokens"])
    ):
        stage_total = max(
            1,
            stage_totals[stats["stage"]][
                "train_tokens" if stats["split"] == "train" else "val_tokens"
            ],
        )
        per_dataset_stats.append(
            {
                **stats,
                "pct_in_stage": f"{stats['n_tokens'] / stage_total * 100:.2f}%",
                "pct_overall": f"{stats['n_tokens'] / max(1, total_tokens) * 100:.2f}%",
            }
        )

    meta = {
        "kind": "walkie_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "input_mode": input_mode,
        "tokenizer": {
            "kind": "byte_bpe",
            "vocab_size": vocab_size,
            "path": str(tokenizer_dir),
            "special_tokens": [eos_token, pad_token],
            "eos_token": eos_token,
            "eos_id": eos_id,
            "pad_token": pad_token,
            "pad_id": pad_id,
            "add_eos_per_doc": True,
            "pad_written_to_bin": False,
        },
        "dtype": str(dtype),
        "dtype_max": dtype_limit,
        "validation": {
            "kind": "document_hash",
            "main_val_ratio": args.main_val_ratio,
            "anneal_val_ratio": args.anneal_val_ratio,
            "salt": args.val_salt,
        },
        "stage_totals": stage_totals,
        "dataset_stats": per_dataset_stats,
        "source_payload": source_payload,
        "files": {
            "main_bin": str(main_bin),
            "main_val_bin": str(main_val_bin),
            "anneal_bin": str(anneal_bin),
            "anneal_val_bin": str(anneal_val_bin),
            "split_manifest": str(split_manifest),
        },
        "train_overrides": {
            "data.stages.main.bin": main_bin.as_posix(),
            "data.stages.main.val_bin": main_val_bin.as_posix(),
            "data.stages.main.dtype": str(dtype),
            "data.stages.anneal.bin": anneal_bin.as_posix(),
            "data.stages.anneal.val_bin": anneal_val_bin.as_posix(),
            "data.stages.anneal.dtype": str(dtype),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.rmtree(shards_dir, ignore_errors=True)
    print(f"main train tokens  : {stage_totals['main']['train_tokens']:,}")
    print(f"main val tokens    : {stage_totals['main']['val_tokens']:,}")
    print(f"anneal train tokens: {stage_totals['anneal']['train_tokens']:,}")
    print(f"anneal val tokens  : {stage_totals['anneal']['val_tokens']:,}")
    print(f"total  tokens: {total_tokens:,}")
    print(f"wrote {split_manifest}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()