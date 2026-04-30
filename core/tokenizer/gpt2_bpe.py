"""GPT-2 官方 BPE 兼容路径。

借助 ``transformers`` / ``tokenizers`` 包加载官方 ``vocab.json`` 与 ``merges.txt``，
专门用于和 HuggingFace GPT-2 权重做 logits 对齐（自训 BPE 得到的词表与官方不一致）。
仅在用户显式调用时才依赖 ``transformers``，不进入项目的强依赖。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from core.tokenizer.base import BaseTokenizer


class GPT2BPETokenizer(BaseTokenizer):
    """对 ``transformers.GPT2TokenizerFast`` 的薄包装，统一接口风格。"""

    KIND = "gpt2"

    def __init__(self, model_name: str = "gpt2") -> None:
        from transformers import GPT2TokenizerFast  # 延迟导入

        self._model_name = model_name
        self._tok = GPT2TokenizerFast.from_pretrained(model_name)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"kind": self.KIND, "model_name": self._model_name}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GPT2BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(model_name=payload.get("model_name", "gpt2"))
