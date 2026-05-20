"""语料编码工具。

负责从已下载的 HF snapshot 中提取文本、迭代、并编码为 ``train.bin`` / ``val.bin``。

公开接口：
- ``iter_texts(cache_dir, ...)``：流式迭代已下载的 parquet/jsonl，逐条产出文本字符串。
- ``encode_corpus(cache_dir, tokenizer, ...)``：用已训练好的分词器将语料编码为
  ``train.bin`` / ``val.bin``。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def _join_message_list(messages: list[object]) -> str:
    chunks: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            content = "\n".join(parts)
        if isinstance(content, str) and content.strip():
            if isinstance(role, str) and role.strip():
                chunks.append(f"{role}: {content.strip()}")
            else:
                chunks.append(content.strip())
    return "\n".join(chunks).strip()


def _extract_text(example: dict, text_field: str | None = None) -> str:
    if text_field is not None:
        value = example.get(text_field)
        if not isinstance(value, str):
            raise ValueError(
                f"字段 {text_field!r} 不是字符串，无法直接用于预训练文本。"
            )
        return value.strip()

    for field_name in ("text", "content", "markdown", "completion", "output"):
        value = example.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for field_name in ("messages", "conversations", "conversation"):
        value = example.get(field_name)
        if isinstance(value, list):
            text = _join_message_list(value)
            if text:
                return text

    if isinstance(example.get("instruction"), str) and isinstance(
        example.get("output"), str
    ):
        parts = [example["instruction"].strip()]
        if isinstance(example.get("input"), str) and example["input"].strip():
            parts.append(example["input"].strip())
        parts.append(example["output"].strip())
        return "\n".join(parts)

    raise ValueError(
        "无法自动推断文本字段，请在配置里显式指定 data.hf.text_field。"
    )


def _iter_parquet_examples(file_path: Path):
    parquet = pq.ParquetFile(file_path)
    for batch in parquet.iter_batches(batch_size=256):
        table = batch.to_pylist()
        for row in table:
            yield row


def _iter_jsonl_examples(file_path: Path):
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_json_examples(file_path: Path):
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for row in payload["data"]:
                if isinstance(row, dict):
                    yield row
        else:
            yield payload


def _is_hidden_path(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _list_data_files(root: Path) -> list[Path]:
    parquet_files = [
        path for path in sorted(root.rglob("*.parquet")) if not _is_hidden_path(path, root)
    ]
    jsonl_files = [
        path for path in sorted(root.rglob("*.jsonl")) if not _is_hidden_path(path, root)
    ]
    json_files = [
        path
        for path in sorted(root.rglob("*.json"))
        if not _is_hidden_path(path, root)
        and path.name not in {"dataset_infos.json", "data_meta.json"}
    ]
    return parquet_files + jsonl_files + json_files


def _resolve_data_dir(cache_dir: Path) -> Path:
    for candidate in (cache_dir / "hf_snapshot", cache_dir):
        if candidate.exists() and _list_data_files(candidate):
            return candidate

    raise FileNotFoundError(
        f"找不到数据目录 {cache_dir / 'hf_snapshot'}，且在 {cache_dir} 下也没有找到 parquet/jsonl/json 数据文件。"
    )


def _iter_examples(data_dir: Path, split: str):
    split_key = split.replace("[", "_").replace("]", "_").replace(":", "_")
    candidates = _list_data_files(data_dir)

    if not candidates:
        raise ValueError(f"仓库 {data_dir} 中没有找到可读取的数据文件。")

    preferred = [
        path
        for path in candidates
        if split in path.as_posix() or split_key in path.as_posix()
    ]
    files = preferred or candidates

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            yield from _iter_parquet_examples(file_path)
        elif suffix == ".jsonl":
            yield from _iter_jsonl_examples(file_path)
        elif suffix == ".json":
            yield from _iter_json_examples(file_path)


def _read_meta(cache_dir: Path) -> dict:
    """读取 ``cache_dir/data_meta.json``，不存在时返回空字典。"""
    meta_path = cache_dir / "data_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def iter_texts(
    cache_dir: str | Path,
    split: str | None = None,
    text_field: str | None = None,
    max_samples: int | None = None,
    max_chars: int | None = None,
) -> Iterator[str]:
    """流式迭代 ``cache_dir`` 下的语料文本。

    若 ``data_meta.json`` 存在且对应参数为 ``None``，则从其中读取默认值。
    显式传入的参数优先级高于 ``data_meta.json`` 中的值。

    Yields:
        每条样本提取出的非空文本字符串。
    """
    cache_dir = Path(cache_dir)
    data_dir = _resolve_data_dir(cache_dir)

    meta = _read_meta(cache_dir)
    split = split if split is not None else meta.get("split", "train")
    text_field = text_field if text_field is not None else meta.get("text_field")
    max_samples = max_samples if max_samples is not None else meta.get("max_samples")
    max_chars = max_chars if max_chars is not None else meta.get("max_chars")

    seen_samples = 0
    seen_chars = 0
    for example in _iter_examples(data_dir, split=split):
        text = _extract_text(example, text_field=text_field)
        if not text:
            continue
        yield text
        seen_samples += 1
        seen_chars += len(text)
        if max_samples is not None and seen_samples >= max_samples:
            break
        if max_chars is not None and seen_chars >= max_chars:
            break


def encode_corpus(
    cache_dir: str | Path,
    tokenizer,
    val_ratio: float = 0.1,
    max_samples: int | None = None,
    max_chars: int | None = None,
) -> dict:
    """用 ``tokenizer`` 将 ``cache_dir/hf_snapshot`` 下的语料编码为 ``train.bin`` / ``val.bin``。

    - 若 ``train.bin`` / ``val.bin`` 已存在 -> 跳过重新编码（幂等）。
    - 编码采用流式写入：每条样本 encode 后立即追加到 bin 文件，不会一次性加载所有 token。
    - 验证集按样本编号取模分配（每 ``round(1/val_ratio)`` 条划分 1 条到 val）。

    Args:
        tokenizer: 已就绪的分词器实例（如 :class:`core.tokenizer.ByteBPETokenizer`）。
        val_ratio: 验证集占比。
        max_samples: 额外限制最多编码样本数；``None`` 表示不限。
        max_chars: 额外限制最多编码字符数；``None`` 表示不限。

    Returns:
        包含 ``train_bin`` / ``val_bin`` / ``tokenizer_path`` / ``vocab_size`` / ``dtype`` 的字典。
    """
    from core.tokenizer.base import BaseTokenizer

    if not isinstance(tokenizer, BaseTokenizer):
        raise TypeError(
            f"tokenizer 必须是 BaseTokenizer 子类实例， got {type(tokenizer)}"
        )

    cache_dir = Path(cache_dir)
    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"
    tokenizer_path = cache_dir / "tokenizer.json"

    meta = _read_meta(cache_dir)
    max_samples = max_samples if max_samples is not None else meta.get("max_samples")
    max_chars = max_chars if max_chars is not None else meta.get("max_chars")

    vocab_size_out = tokenizer.vocab_size
    dtype: np.dtype = np.dtype(
        np.uint16 if vocab_size_out <= np.iinfo(np.uint16).max + 1 else np.uint32
    )

    if train_bin.exists() and val_bin.exists():
        print(f"[encode] bin 文件已存在，跳过重新编码。")
    else:
        val_period = max(1, int(round(1.0 / max(val_ratio, 1e-6))))
        train_tokens = 0
        val_tokens = 0
        seen_samples = 0
        seen_chars = 0
        print(
            f"[encode] 流式编码 -> {train_bin.name} / {val_bin.name}"
            f"（dtype={dtype}, val 每 {val_period} 条取 1）"
        )
        with open(train_bin, "wb") as train_f, open(val_bin, "wb") as val_f:
            pbar = tqdm(desc="Encoding", unit="sample")
            for text in iter_texts(
                cache_dir, max_samples=max_samples, max_chars=max_chars
            ):
                ids = np.array(tokenizer.encode(text), dtype=dtype)
                if seen_samples % val_period == 0:
                    ids.tofile(val_f)
                    val_tokens += len(ids)
                else:
                    ids.tofile(train_f)
                    train_tokens += len(ids)
                seen_samples += 1
                seen_chars += len(text)
                pbar.update(1)
                pbar.set_postfix(chars=f"{seen_chars:,}")
                if max_samples is not None and seen_samples >= max_samples:
                    break
                if max_chars is not None and seen_chars >= max_chars:
                    break
            pbar.close()
        print(
            f"[encode] 完成: 样本={seen_samples:,} train={train_tokens:,} "
            f"val={val_tokens:,} vocab={vocab_size_out} dtype={dtype}"
        )

    return {
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": vocab_size_out,
        "dtype": str(dtype),
        "snapshot_dir": str(cache_dir / "hf_snapshot"),
    }
