"""tiny shakespeare 数据集准备工具。

入口：``prepare(cache_dir, tokenizer_kind, vocab_size)`` 会
    1. 下载原始文本到 ``cache_dir/input.txt``（若已存在则跳过）；
    2. 训练或加载分词器并保存到 ``cache_dir/tokenizer.json``；
    3. 把整个语料编码为 ``train.bin`` / ``val.bin``，9:1 划分（dtype 自动按 vocab 选 uint16/uint32）。

后续 ``train/pretrain.py`` 直接 mmap 这两个 bin 文件做随机采样。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from core.tokenizer import build_tokenizer, load_tokenizer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / "input.txt"
    if txt_path.exists():
        return txt_path
    print(f"[data] 下载 tiny shakespeare -> {txt_path}")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    txt_path.write_text(resp.text, encoding="utf-8")
    return txt_path


def prepare(
    cache_dir: str | Path,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 1024,
    train_on_first_n_chars: int | None = None,
) -> dict:
    """准备数据，返回 ``{'train_bin', 'val_bin', 'tokenizer_path', 'vocab_size', 'dtype'}``。

    支持的 ``tokenizer_kind``：``bpe`` / ``byte_bpe`` / ``wordpiece`` / ``unigram`` / ``gpt2``。
    """
    cache_dir = Path(cache_dir)
    txt_path = _download(cache_dir)
    text = txt_path.read_text(encoding="utf-8")

    tokenizer_path = cache_dir / "tokenizer.json"

    if tokenizer_kind == "gpt2":
        # 预训练词表，无需训练；也不写 tokenizer.json（避免误覆盖之前的自训词表）
        tok = build_tokenizer("gpt2")
    else:
        if tokenizer_path.exists():
            tok = load_tokenizer(tokenizer_path)
            if tok.KIND != tokenizer_kind:
                raise ValueError(
                    f"缓存中的分词器 kind={tok.KIND!r} 与请求的 {tokenizer_kind!r} 不一致；"
                    f"请删除 {tokenizer_path} 后重试。"
                )
        else:
            print(f"[data] 训练 tokenizer kind={tokenizer_kind} 目标词表 {vocab_size}")
            train_text = text if train_on_first_n_chars is None else text[: train_on_first_n_chars]
            cls = type(build_tokenizer(tokenizer_kind))
            tok = cls.train(train_text, vocab_size=vocab_size)
            tok.save(tokenizer_path)

    vocab_size_out = tok.vocab_size
    dtype: np.dtype = np.dtype(np.uint16 if vocab_size_out < 65536 else np.uint32)
    ids = np.array(tok.encode(text), dtype=dtype)

    n = len(ids)
    split = int(n * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    print(
        f"[data] 完成: train={len(train_ids):,} val={len(val_ids):,} "
        f"vocab={vocab_size_out} dtype={dtype}"
    )
    return {
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "tokenizer_path": str(tokenizer_path) if tokenizer_kind != "gpt2" else None,
        "vocab_size": vocab_size_out,
        "dtype": str(dtype),
    }


def load_split(bin_path: str | Path, dtype: str = "uint16") -> np.ndarray:
    """以 mmap 方式加载一个 split。"""
    return np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
