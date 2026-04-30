"""分词器：统一通过 :func:`build_tokenizer` 工厂创建，序列化通过 :func:`load_tokenizer`。

支持的 ``kind``：

- ``"bpe"``：教学版字符级 BPE，见 :mod:`core.tokenizer.bpe`。
- ``"byte_bpe"``：真 GPT-2 风格字节级 BPE，见 :mod:`core.tokenizer.byte_bpe`。
- ``"wordpiece"``：BERT 风格 WordPiece，见 :mod:`core.tokenizer.wordpiece`。
- ``"unigram"``：SentencePiece / LLaMA 风格 Unigram LM，见 :mod:`core.tokenizer.unigram`。
- ``"gpt2"``：包装 ``transformers.GPT2TokenizerFast`` 的官方词表路径，见 :mod:`core.tokenizer.gpt2_bpe`。
"""

from __future__ import annotations

import json
from pathlib import Path

from core.tokenizer.base import BaseTokenizer
from core.tokenizer.bpe import BPETokenizer
from core.tokenizer.byte_bpe import ByteBPETokenizer
from core.tokenizer.unigram import UnigramTokenizer
from core.tokenizer.wordpiece import WordPieceTokenizer

# kind → 类
_TOKENIZER_REGISTRY: dict[str, type[BaseTokenizer]] = {
    "bpe": BPETokenizer,
    "byte_bpe": ByteBPETokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
}


def _get_gpt2_class() -> type[BaseTokenizer]:
    # 延迟导入：依赖 ``transformers``，不放进 registry 顶层
    from core.tokenizer.gpt2_bpe import GPT2BPETokenizer

    return GPT2BPETokenizer


def build_tokenizer(kind: str, **kwargs) -> BaseTokenizer:
    """根据 ``kind`` 返回**未训练**的分词器实例（GPT-2 兼容路径直接构造为已就绪状态）。

    Args:
        kind: ``"bpe" | "byte_bpe" | "wordpiece" | "unigram" | "gpt2"``。
        **kwargs: 传给具体类构造函数的参数（一般可省）。
    """
    if kind == "gpt2":
        return _get_gpt2_class()(**kwargs)
    if kind in _TOKENIZER_REGISTRY:
        return _TOKENIZER_REGISTRY[kind](**kwargs)
    raise ValueError(f"unknown tokenizer kind: {kind!r}; supported: {list(_TOKENIZER_REGISTRY) + ['gpt2']}")


def load_tokenizer(path: str | Path) -> BaseTokenizer:
    """根据 JSON 文件中的 ``kind`` 字段自动分发到对应类的 ``load``。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = payload.get("kind", "bpe")  # 兼容 V0 之前没写 kind 的旧文件
    if kind == "gpt2":
        return _get_gpt2_class().load(path)
    if kind not in _TOKENIZER_REGISTRY:
        raise ValueError(f"unknown tokenizer kind in {path}: {kind!r}")
    return _TOKENIZER_REGISTRY[kind].load(path)


__all__ = [
    "BaseTokenizer",
    "BPETokenizer",
    "ByteBPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "build_tokenizer",
    "load_tokenizer",
]
