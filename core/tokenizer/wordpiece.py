"""WordPiece：BERT 系列的子词算法。

与 BPE 的核心差异：
    - **合并准则**：不取频次最高的 pair，而是取 ``score = freq(pair) / (freq(left) * freq(right))``
      最大的 pair（这是对"合并能多大程度提升语料似然"的近似）。
    - **续接前缀** ``##``：词内部的非首子词带 ``##`` 前缀，例如 ``"playing"`` 可能被切成 ``"play"`` + ``"##ing"``。
    - **编码**：贪心 **最长前缀匹配**；遇到无法匹配的前缀则整个词输出 ``[UNK]``（与 BPE 的"逐字符回退"不同）。

本实现是教学版：保留算法核心，简化工程细节（无 BasicTokenizer 全套 NFD/CJK 处理等）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from core.tokenizer.base import BaseTokenizer

_WORD_RE = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def _pretokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class WordPieceTokenizer(BaseTokenizer):
    """教学版 WordPiece。"""

    KIND = "wordpiece"
    SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
    CONTINUATION_PREFIX = "##"

    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id: dict[str, int] = dict(token_to_id or {})
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def eos_token_id(self) -> int:
        # WordPiece 没有原生 EOS；BERT 通常用 [SEP] 当分隔
        return self.token_to_id["[SEP]"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["[UNK]"]

    # ----- training -----
    @classmethod
    def train(cls, corpus: str | Iterable[str], vocab_size: int, verbose: bool = False) -> "WordPieceTokenizer":
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        word_freq: Counter = Counter(_pretokenize(text))

        # 初始词表 = 特殊 token + 出现过的字符（首字符不带 ##，非首带 ##）
        token_to_id: dict[str, int] = {}
        for tok in cls.SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)
        first_chars: set[str] = set()
        cont_chars: set[str] = set()
        for w in word_freq:
            if not w:
                continue
            first_chars.add(w[0])
            for ch in w[1:]:
                cont_chars.add(ch)
        for ch in sorted(first_chars):
            token_to_id.setdefault(ch, len(token_to_id))
        for ch in sorted(cont_chars):
            token_to_id.setdefault(cls.CONTINUATION_PREFIX + ch, len(token_to_id))

        # 把每个词初始化为 [c0, ##c1, ##c2, ...]
        word_splits: dict[str, list[str]] = {}
        for w in word_freq:
            if not w:
                continue
            word_splits[w] = [w[0]] + [cls.CONTINUATION_PREFIX + c for c in w[1:]]

        while len(token_to_id) < vocab_size:
            # 统计 pair 频次与各 token 频次
            pair_freq: Counter = Counter()
            tok_freq: Counter = Counter()
            for w, freq in word_freq.items():
                pieces = word_splits[w]
                for t in pieces:
                    tok_freq[t] += freq
                for i in range(len(pieces) - 1):
                    pair_freq[(pieces[i], pieces[i + 1])] += freq
            if not pair_freq:
                break

            # 选 score 最大的 pair；score = pf / (lf * rf)
            best_pair = None
            best_score = -1.0
            for pair, pf in pair_freq.items():
                left, right = pair
                lf = tok_freq[left]
                rf = tok_freq[right]
                if lf == 0 or rf == 0:
                    continue
                s = pf / (lf * rf)
                if s > best_score:
                    best_score = s
                    best_pair = pair
            if best_pair is None:
                break

            # 合并：左 token 是首子词时直接拼接；如果左 token 已带 ## 则保留 ##；右 token 始终去掉 ## 前缀
            left, right = best_pair
            merged = left + (right[len(cls.CONTINUATION_PREFIX):] if right.startswith(cls.CONTINUATION_PREFIX) else right)
            if merged in token_to_id:
                # 已存在，避免死循环
                # 这种情况下以训练终止收尾——通常意味着 vocab 已收敛
                break
            token_to_id[merged] = len(token_to_id)

            for w, pieces in list(word_splits.items()):
                new_pieces: list[str] = []
                i = 0
                while i < len(pieces):
                    if i < len(pieces) - 1 and pieces[i] == left and pieces[i + 1] == right:
                        new_pieces.append(merged)
                        i += 2
                    else:
                        new_pieces.append(pieces[i])
                        i += 1
                word_splits[w] = new_pieces

            if verbose and (len(token_to_id) % 200 == 0):
                print(f"  vocab={len(token_to_id)} top={best_pair} score={best_score:.4f}")

        return cls(token_to_id=token_to_id)

    # ----- encode / decode -----
    def _encode_word(self, word: str) -> list[int]:
        if not word:
            return []
        ids: list[int] = []
        start = 0
        first = True
        while start < len(word):
            end = len(word)
            cur_token: str | None = None
            while start < end:
                sub = word[start:end]
                cand = sub if first else self.CONTINUATION_PREFIX + sub
                if cand in self.token_to_id:
                    cur_token = cand
                    break
                end -= 1
            if cur_token is None:
                # 整词降级为 [UNK]
                return [self.unk_token_id]
            ids.append(self.token_to_id[cur_token])
            start = end
            first = False
        return ids

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        for w in _pretokenize(text):
            ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.id_to_token.get(int(i), "") for i in ids]
        out: list[str] = []
        special = set(self.SPECIAL_TOKENS)
        for t in toks:
            if t in special:
                continue
            if t.startswith(self.CONTINUATION_PREFIX):
                out.append(t[len(self.CONTINUATION_PREFIX):])
            else:
                if out:
                    out.append(" ")
                out.append(t)
        return "".join(out)

    # ----- save / load -----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": self.KIND, "token_to_id": self.token_to_id}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordPieceTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=payload["token_to_id"])
