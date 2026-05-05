"""教学版 BPE：从零实现一个最小可用的 Byte-Pair Encoding 分词器。

设计要点：
    - 基于"字符序列 + 合并规则"的最朴素 BPE，便于教学，不追求与 GPT-2 官方词表完全一致；
      若需要与 GPT-2 官方权重对齐，请使用 :mod:`core.tokenizer.byte_bpe` 并训练同等大小的词表。
    - 训练算法：参考 Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units"。
    - 单词内部以单字符为初始 token，结尾追加 ``</w>`` 标记词尾，避免跨词合并。
    - 词表 = 所有出现过的字符 ∪ 所有学到的合并产物 ∪ 特殊 token。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tqdm.auto import tqdm

from core.tokenizer.base import BaseTokenizer

# 与 GPT-2 一致的预切分正则：把空白、标点、数字、字母等粗分一下，再做 BPE。
_PRETOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

END_OF_WORD = "</w>"


def _pretokenize(text: str) -> list[str]:
    return [m.group(0) for m in _PRETOKEN_RE.finditer(text) if m.group(0)]


def _word_to_symbols(word: str) -> tuple[str, ...]:
    return tuple(list(word) + [END_OF_WORD])


def _get_pair_counts(vocab: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge_in_word(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    a, b = pair
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer(BaseTokenizer):
    """教学版 BPE。

    特殊 token：
        - ``<|endoftext|>``：序列结束/分隔符。
        - ``<|unk|>``：训练词表中没出现过的字符的兜底（实践中如果训练语料覆盖足够通常用不到）。
    """

    KIND = "bpe"
    SPECIAL_TOKENS = ("<|endoftext|>", "<|unk|>")

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
    ) -> None:
        self.token_to_id: dict[str, int] = dict(token_to_id or {})
        self.merges: list[tuple[str, str]] = list(merges or [])
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self._merge_rank: dict[tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}

    # ----- properties -----
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<|endoftext|>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<|unk|>"]

    # ----- training -----
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """在 ``text`` 上训练 BPE，目标词表大小为 ``vocab_size``。"""
        words: Counter = Counter()
        for piece in _pretokenize(text):
            words[_word_to_symbols(piece)] += 1

        # 初始词表：所有字符 + END_OF_WORD + 特殊 token
        chars: set[str] = set()
        for symbols in words:
            chars.update(symbols)
        token_to_id: dict[str, int] = {}
        for tok in cls.SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)
        for ch in sorted(chars):
            if ch not in token_to_id:
                token_to_id[ch] = len(token_to_id)

        merges: list[tuple[str, str]] = []
        vocab = dict(words)
        pbar = tqdm(total=vocab_size, initial=len(token_to_id), desc="Training BPE", unit="token")
        while len(token_to_id) < vocab_size:
            pair_counts = _get_pair_counts(vocab)
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in token_to_id:
                token_to_id[merged_token] = len(token_to_id)
                pbar.update(1)
            merges.append(best_pair)
            vocab = {
                _merge_in_word(symbols, best_pair): freq
                for symbols, freq in vocab.items()
            }
            if verbose and len(merges) % 100 == 0:
                pbar.set_postfix(merges=len(merges), top=f"{best_pair}({best_freq})")
        pbar.close()

        return cls(token_to_id=token_to_id, merges=merges)

    # ----- encode / decode -----
    def _bpe(self, word: str) -> list[str]:
        symbols = list(_word_to_symbols(word))
        while True:
            best_rank = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self._merge_rank.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_rank is None:
                break
            a, b = symbols[best_i], symbols[best_i + 1]
            symbols = symbols[:best_i] + [a + b] + symbols[best_i + 2:]
        return symbols

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        unk_id = self.unk_token_id
        for piece in _pretokenize(text):
            for tok in self._bpe(piece):
                ids.append(self.token_to_id.get(tok, unk_id))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.id_to_token.get(int(i), "") for i in ids]
        text = "".join(toks)
        return text.replace(END_OF_WORD, "")

    # ----- save / load -----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.KIND,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in payload["merges"]]
        return cls(token_to_id=payload["token_to_id"], merges=merges)
