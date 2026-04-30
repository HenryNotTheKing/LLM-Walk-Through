"""分词器抽象基类。

所有分词器实现都应当继承 :class:`BaseTokenizer`，以保证：

- 上层（训练 / 生成 / 数据预处理）可以用统一接口换装；
- 序列化格式可识别（``save`` 写出的 ``json`` 一定带 ``"kind"`` 字段，``load`` 据此分发）。

可训练的子类**还应**额外提供 ``classmethod train(cls, corpus, vocab_size, **kw) -> Self``；
预训练（如 HF GPT-2）的子类无需 ``train``，调用时抛 ``NotImplementedError``。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class BaseTokenizer(ABC):
    """所有分词器的统一接口。

    约定：
        - ``encode(text, add_eos)`` 返回 ``list[int]``。
        - ``decode(ids)`` 返回 ``str``。
        - ``save(path)`` / ``load(path)`` 序列化 / 反序列化为单个 JSON 文件，文件中含 ``"kind"`` 字段。
        - ``vocab_size`` 为属性。
        - ``eos_token_id`` 为属性；若实现没有专门的 EOS，可指向最常用的"分隔" token。
    """

    #: 该子类对应的 ``kind`` 字符串，与 :func:`core.tokenizer.build_tokenizer` 的 ``kind`` 参数一致。
    KIND: str = "base"

    # ----- 必须实现的抽象接口 -----
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str: ...

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseTokenizer": ...

    # ----- 默认行为 -----
    def encode_batch(self, texts: Iterable[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(t, add_eos=add_eos) for t in texts]


def _peek_kind(path: str | Path) -> str:
    """读 JSON 文件首部的 ``kind`` 字段，用于工厂决定走哪个 ``load``。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("kind", "bpe")  # 兼容老格式（首版 BPE 没写 kind）
