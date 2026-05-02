"""Hugging Face 文本数据集准备工具。

公开接口（可逐步调用）：
- ``download_text(cache_dir, repo_id, ...)``：从 HF 下载语料并写入 ``cache_dir/input.txt``，
  文件已存在则直接跳过，幂等。
- ``tokenize(cache_dir, tokenizer_kind, ...)``：训练/复用分词器，将语料编码为
  ``train.bin`` / ``val.bin``。若已有分词器但 kind 不匹配，会报错并告知修复方法。
- ``prepare(cache_dir, repo_id, ...)``：依次调用上面两步的便捷封装，向后兼容。
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from core.tokenizer import build_tokenizer, load_tokenizer


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
            raise ValueError(f"字段 {text_field!r} 不是字符串，无法直接用于预训练文本。")
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

    if isinstance(example.get("instruction"), str) and isinstance(example.get("output"), str):
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


def _iter_examples(snapshot_dir: Path, split: str):
    split_key = split.replace("[", "_").replace("]", "_").replace(":", "_")
    parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
    jsonl_files = sorted(snapshot_dir.rglob("*.jsonl"))
    json_files = [path for path in sorted(snapshot_dir.rglob("*.json")) if path.name != "dataset_infos.json"]

    candidates = parquet_files + jsonl_files + json_files
    if not candidates:
        raise ValueError(f"仓库 {snapshot_dir} 中没有找到可读取的数据文件。")

    preferred = [path for path in candidates if split in path.as_posix() or split_key in path.as_posix()]
    files = preferred or candidates

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            yield from _iter_parquet_examples(file_path)
        elif suffix == ".jsonl":
            yield from _iter_jsonl_examples(file_path)
        elif suffix == ".json":
            yield from _iter_json_examples(file_path)


def download_text(
    cache_dir: str | Path,
    repo_id: str,
    split: str = "train",
    subset_name: str | None = None,
    text_field: str | None = None,
    max_samples: int | None = None,
    max_chars: int | None = None,
    hf_endpoint: str | None = None,
) -> Path:
    """从 HF 下载文本语料并写入 ``cache_dir/input.txt``。

    幂等：若 ``input.txt`` 已存在则直接返回路径，不重复下载。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / "input.txt"
    if txt_path.exists():
        print(f"[data] 语料已存在，跳过下载: {txt_path}")
        return txt_path

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    print(f"[data] 下载 HF 数据集子集 {repo_id}:{split} -> {txt_path}")
    if subset_name is not None:
        raise ValueError("当前实现暂不支持 dataset config name，请先使用默认配置。")

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir / "hf_snapshot",
            allow_patterns=["*.parquet", "*.jsonl", "*.json", "README.md"],
            max_workers=8,
        )
    )

    texts: list[str] = []
    seen_samples = 0
    seen_chars = 0
    for example in _iter_examples(snapshot_dir, split=split):
        text = _extract_text(example, text_field=text_field)
        if not text:
            continue
        texts.append(text)
        seen_samples += 1
        seen_chars += len(text)

        if max_samples is not None and seen_samples >= max_samples:
            break
        if max_chars is not None and seen_chars >= max_chars:
            break

    if not texts:
        raise ValueError(f"数据集 {repo_id!r} 没有提取到可用文本。")

    txt_path.write_text("\n\n".join(texts), encoding="utf-8")
    print(f"[data] 已抽取 {seen_samples} 条样本，约 {seen_chars:,} 字符")
    return txt_path


def tokenize(
    cache_dir: str | Path,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 1024,
    val_ratio: float = 0.1,
) -> dict:
    """对 ``cache_dir/input.txt`` 进行分词并编码为二进制文件。

    - 若 ``tokenizer.json`` 已存在且 kind 匹配 → 直接复用，不重新训练。
    - 若 ``tokenizer.json`` 已存在但 kind **不匹配** → 抛出 ValueError 并给出修复提示，
      不会自动删除任何文件。
    - 若 ``train.bin`` / ``val.bin`` 已存在且分词器兼容 → 跳过重新编码。
    """
    cache_dir = Path(cache_dir)
    txt_path = cache_dir / "input.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"找不到语料文件 {txt_path}，请先调用 download_text() 下载数据。"
        )

    tokenizer_path = cache_dir / "tokenizer.json"
    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"

    if tokenizer_kind == "gpt2":
        tok = build_tokenizer("gpt2")
    else:
        if tokenizer_path.exists():
            tok = load_tokenizer(tokenizer_path)
            if tok.KIND != tokenizer_kind:
                raise ValueError(
                    f"缓存的分词器 kind={tok.KIND!r} 与当前设置 tokenizer_kind={tokenizer_kind!r} 不一致。\n"
                    f"  方案 A（推荐）: 将 tokenizer_kind 改回 {tok.KIND!r} 以复用现有分词器。\n"
                    f"  方案 B: 手动删除 {tokenizer_path} （以及 train.bin / val.bin）后重新运行。"
                )
            print(f"[data] 复用已有分词器 kind={tok.KIND!r}: {tokenizer_path}")
        else:
            print(f"[data] 训练 tokenizer kind={tokenizer_kind} 目标词表 {vocab_size}")
            cls = type(build_tokenizer(tokenizer_kind))
            tok = cls.train(txt_path.read_text(encoding="utf-8"), vocab_size=vocab_size)
            tok.save(tokenizer_path)

    vocab_size_out = tok.vocab_size
    dtype: np.dtype = np.dtype(np.uint16 if vocab_size_out < 65536 else np.uint32)

    if train_bin.exists() and val_bin.exists():
        print(f"[data] bin 文件已存在，跳过重新编码。")
    else:
        text = txt_path.read_text(encoding="utf-8")
        ids = np.array(tok.encode(text), dtype=dtype)
        split_index = int(len(ids) * (1.0 - val_ratio))
        ids[:split_index].tofile(train_bin)
        ids[split_index:].tofile(val_bin)
        print(
            f"[data] 完成: train={split_index:,} val={len(ids) - split_index:,} "
            f"vocab={vocab_size_out} dtype={dtype}"
        )

    return {
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "tokenizer_path": str(tokenizer_path) if tokenizer_kind != "gpt2" else None,
        "vocab_size": vocab_size_out,
        "dtype": str(dtype),
        "input_txt": str(txt_path),
    }


def prepare(
    cache_dir: str | Path,
    repo_id: str,
    split: str = "train",
    subset_name: str | None = None,
    text_field: str | None = None,
    max_samples: int | None = None,
    max_chars: int | None = None,
    hf_endpoint: str | None = None,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 1024,
    val_ratio: float = 0.1,
) -> dict:
    """便捷封装：依次调用 download_text() 和 tokenize()，向后兼容。"""
    download_text(
        cache_dir=cache_dir,
        repo_id=repo_id,
        split=split,
        subset_name=subset_name,
        text_field=text_field,
        max_samples=max_samples,
        max_chars=max_chars,
        hf_endpoint=hf_endpoint,
    )
    result = tokenize(
        cache_dir=cache_dir,
        tokenizer_kind=tokenizer_kind,
        vocab_size=vocab_size,
        val_ratio=val_ratio,
    )
    result["repo_id"] = repo_id
    return result