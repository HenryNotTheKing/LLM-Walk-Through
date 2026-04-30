"""Byte-level BPE：与真正的 GPT-2 / GPT-3 / RoBERTa 一致的算法。

与 :mod:`core.tokenizer.bpe`（字符级教学版）的关键区别：

- **输入是字节**：先把文本 utf-8 编码成字节序列，再走 BPE。
  这样任何字符串都可以被分词，**不可能 OOV**。
- **bytes ↔ unicode 映射**：把 0–255 字节映射到一组可打印的 unicode 字符，
  避免训练时空白/控制字符干扰，也方便日志查看。映射表与 OpenAI 官方 ``encoder.py`` 一致。
- **预切分仍用 GPT-2 那条正则**：先把文本粗切成单词/标点段，再做 BPE，避免跨段合并。

这是从零造轮子的实现：不依赖 ``transformers`` / ``tokenizers``，可以独立训练。
若只是想加载官方 ``gpt2`` 的 50257 词表做权重对齐，请用 :class:`core.tokenizer.gpt2_bpe.GPT2BPETokenizer`。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from core.tokenizer.base import BaseTokenizer

_PRETOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """OpenAI 官方实现的字节↔可打印 unicode 双射。

    覆盖 ``!``–``~``、``¡``–``¬``、``®``–``ÿ`` 共 188 个"安全"字符，
    其余 68 个字节顺序映射到 ``256+i``。
    """
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _pretokenize(text: str) -> list[str]:
    return [m.group(0) for m in _PRETOKEN_RE.finditer(text) if m.group(0)]


def _get_pair_counts(words: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter = Counter()
    for symbols, freq in words.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
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


class ByteBPETokenizer(BaseTokenizer):
    """字节级 BPE 分词器（GPT-2 风格）。"""

    KIND = "byte_bpe"
    SPECIAL_TOKENS = ("<|endoftext|>",)

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
    ) -> None:
        self.token_to_id: dict[str, int] = dict(token_to_id or {})
        self.merges: list[tuple[str, str]] = [tuple(p) for p in (merges or [])]
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self._merge_rank: dict[tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}
        self._b2u = bytes_to_unicode()
        self._u2b = {v: k for k, v in self._b2u.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<|endoftext|>"]

    # ----- training -----
    @classmethod
    def train(cls, corpus: str | Iterable[str], vocab_size: int, verbose: bool = False) -> "ByteBPETokenizer":
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        b2u = bytes_to_unicode()

        words: Counter = Counter()
        for piece in _pretokenize(text):
            uchars = tuple(b2u[b] for b in piece.encode("utf-8"))
            words[uchars] += 1

        token_to_id: dict[str, int] = {}
        for tok in cls.SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)
        # 初始词表：bytes_to_unicode 的全部 256 个 unicode 字符（保证任何字节都能查表）
        for ch in b2u.values():
            if ch not in token_to_id:
                token_to_id[ch] = len(token_to_id)

        merges: list[tuple[str, str]] = []
        vocab = dict(words)
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
            merges.append(best_pair)
            vocab = {_merge(syms, best_pair): freq for syms, freq in vocab.items()}
            if verbose and len(merges) % 100 == 0:
                print(f"  merges={len(merges)} vocab={len(token_to_id)} top={best_pair}({best_freq})")

        return cls(token_to_id=token_to_id, merges=merges)

    # ----- encode / decode -----
    def _bpe(self, uchars: list[str]) -> list[str]:
        symbols = list(uchars)
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
        for piece in _pretokenize(text):
            uchars = [self._b2u[b] for b in piece.encode("utf-8")]
            for tok in self._bpe(uchars):
                if tok in self.token_to_id:
                    ids.append(self.token_to_id[tok])
                else:
                    # 任何字节级合并失败的 fallback：拆回单字符
                    for ch in tok:
                        ids.append(self.token_to_id[ch])
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.id_to_token.get(int(i), "") for i in ids]
        # 跳过特殊 token，免得字节恢复时遇到 "<|endoftext|>" 这种"非映射字符"
        special = set(self.SPECIAL_TOKENS)
        text_chars = "".join(t for t in toks if t not in special)
        try:
            byts = bytes([self._u2b[ch] for ch in text_chars])
        except KeyError:
            # 出现非映射字符（理论上不该发生），逐字节兜底
            byts = bytes(self._u2b.get(ch, ord("?")) for ch in text_chars)
        return byts.decode("utf-8", errors="replace")

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
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in payload["merges"]]
        return cls(token_to_id=payload["token_to_id"], merges=merges)
