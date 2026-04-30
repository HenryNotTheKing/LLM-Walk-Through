"""Unigram Language Model 分词器（SentencePiece / LLaMA / T5 系列使用）。

与 BPE / WordPiece 的"自下而上合并"不同，Unigram 的训练是**自上而下裁剪**：

1. 用语料里的高频子串构造一个**很大**的种子词表（远大于目标 vocab_size）；
2. 用 EM 估出每个 token 的概率 ``p(token)``；
3. 对每个 token 计算"如果删了它，整体最大似然损失会增加多少"——损失增加越小越该删；
4. 反复迭代：E 步 Viterbi 求最优分段并累加期望计数 → M 步重估概率 → 裁剪一部分低贡献 token；
5. 直到 vocab 收敛到目标大小。

编码（推理）时用 **Viterbi** 求一句话的最大概率分段。

本实现是**教学版**：算法骨架与 Kudo 2018 一致，但简化了若干工程细节
（例如不做 BOS/EOS 插入、不做 NFD 归一化、种子词表只取高频前缀等）。
若需要工业级实现请用 :mod:`sentencepiece`。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from core.tokenizer.base import BaseTokenizer

# Unigram 通常把空格替换为 ``▁`` 作为词首标记（SentencePiece 风格），便于无损还原原文
SPACE_MARKER = "\u2581"  # ▁


def _normalize(text: str) -> str:
    """SentencePiece 风格：空格 → ``▁``，并在文本开头补一个 ``▁``。"""
    if not text:
        return text
    return SPACE_MARKER + text.replace(" ", SPACE_MARKER)


class UnigramTokenizer(BaseTokenizer):
    """教学版 Unigram LM 分词器。"""

    KIND = "unigram"
    SPECIAL_TOKENS = ("<unk>", "<s>", "</s>")

    def __init__(
        self,
        token_to_id: dict[str, int] | None = None,
        log_probs: dict[str, float] | None = None,
    ) -> None:
        self.token_to_id: dict[str, int] = dict(token_to_id or {})
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        # log p(token)；越大表示越倾向于在分段里使用这个 token
        self.log_probs: dict[str, float] = dict(log_probs or {})

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["</s>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    # ============== Viterbi: 求最大对数概率分段 ==============
    def _viterbi(self, text: str) -> list[str]:
        n = len(text)
        if n == 0:
            return []
        # dp[i] = 解码到 text[:i] 的最大 log-prob；back[i] = 选择的最后一个 token 长度
        neg_inf = -1e18
        dp = [neg_inf] * (n + 1)
        back = [0] * (n + 1)
        dp[0] = 0.0
        unk_logp = self.log_probs.get("<unk>", math.log(1e-6))
        for i in range(1, n + 1):
            # 尝试所有以 i 结尾的子串
            best = neg_inf
            best_j = i - 1
            for j in range(0, i):
                sub = text[j:i]
                lp = self.log_probs.get(sub)
                if lp is None:
                    # 单字符 fallback：按 unk 处理
                    if i - j != 1:
                        continue
                    lp = unk_logp
                cand = dp[j] + lp
                if cand > best:
                    best = cand
                    best_j = j
            dp[i] = best
            back[i] = i - best_j
        # 回溯
        out: list[str] = []
        i = n
        while i > 0:
            length = back[i]
            out.append(text[i - length:i])
            i -= length
        out.reverse()
        return out

    # ============== 训练 ==============
    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int,
        seed_size: int = 10_000,
        max_substr_len: int = 16,
        n_iter: int = 4,
        prune_ratio: float = 0.2,
        verbose: bool = False,
    ) -> "UnigramTokenizer":
        text = corpus if isinstance(corpus, str) else "\n".join(corpus)
        text = _normalize(text)

        # 1. 构造种子词表：所有出现过的字符 + 高频子串前 seed_size 个
        chars: Counter = Counter(text)
        substr_freq: Counter = Counter()
        # 按"行"统计，避免跨行子串
        for line in text.split("\n"):
            L = len(line)
            for i in range(L):
                for k in range(2, min(max_substr_len, L - i) + 1):
                    substr_freq[line[i:i + k]] += 1

        seed_tokens: dict[str, float] = {ch: float(freq) for ch, freq in chars.items() if freq > 0}
        for sub, freq in substr_freq.most_common(seed_size):
            if sub not in seed_tokens:
                seed_tokens[sub] = float(freq)

        # 转 log p
        total = sum(seed_tokens.values())
        log_probs: dict[str, float] = {t: math.log(c / total) for t, c in seed_tokens.items()}

        def make_tokenizer(lps: dict[str, float]) -> "UnigramTokenizer":
            t2i: dict[str, int] = {}
            for s in cls.SPECIAL_TOKENS:
                t2i[s] = len(t2i)
            for tok in lps:
                if tok not in t2i:
                    t2i[tok] = len(t2i)
            # 给特殊 token 一个很小的概率（编码时一般不会选到它们）
            for s in cls.SPECIAL_TOKENS:
                lps.setdefault(s, math.log(1e-9))
            return cls(token_to_id=t2i, log_probs=lps)

        # 2. EM + 裁剪迭代
        lines = [ln for ln in text.split("\n") if ln]
        for it in range(n_iter):
            # 终止条件：词表已经接近目标
            current_vocab = len(log_probs) + len(cls.SPECIAL_TOKENS)
            if current_vocab <= vocab_size:
                break

            tok = make_tokenizer(dict(log_probs))

            # E 步：对每行做 Viterbi，累加每个 token 的使用次数
            counts: Counter = Counter()
            total_log_lik = 0.0
            for ln in lines:
                pieces = tok._viterbi(ln)
                counts.update(pieces)
                for p in pieces:
                    total_log_lik += tok.log_probs.get(p, math.log(1e-12))

            # M 步：用 (count + 1) 平滑，更新概率
            denom = sum(counts.values()) + len(counts)
            new_lps: dict[str, float] = {}
            for t, c in counts.items():
                if t in cls.SPECIAL_TOKENS:
                    continue
                new_lps[t] = math.log((c + 1) / denom)

            # 永远保留单字符（保证任何 utf-8 串都有可行分段）
            for ch in chars:
                if ch not in new_lps:
                    new_lps[ch] = math.log(1e-9)

            # 裁剪：除去最不重要的 prune_ratio。重要性 ≈ count（教学近似；正式版按"删掉它后似然下降量"算）
            if len(new_lps) > vocab_size - len(cls.SPECIAL_TOKENS):
                # 对多字符 token 排序裁剪；单字符不裁
                multi = [(t, counts.get(t, 0)) for t in new_lps if len(t) > 1]
                multi.sort(key=lambda x: x[1])
                target_extra = max(0, len(new_lps) - (vocab_size - len(cls.SPECIAL_TOKENS)))
                drop_n = min(int(prune_ratio * len(multi)) + 1, target_extra, len(multi))
                for t, _ in multi[:drop_n]:
                    new_lps.pop(t, None)

            log_probs = new_lps
            if verbose:
                print(f"  iter={it} vocab={len(log_probs)} log_lik={total_log_lik:.1f}")

        return make_tokenizer(log_probs)

    # ============== encode / decode ==============
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        norm = _normalize(text)
        pieces = self._viterbi(norm) if norm else []
        ids = [self.token_to_id.get(p, self.unk_token_id) for p in pieces]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.id_to_token.get(int(i), "") for i in ids]
        special = set(self.SPECIAL_TOKENS)
        text = "".join(t for t in toks if t not in special)
        return text.replace(SPACE_MARKER, " ").lstrip(" ")

    # ============== save / load ==============
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.KIND,
            "token_to_id": self.token_to_id,
            "log_probs": self.log_probs,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnigramTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=payload["token_to_id"], log_probs=payload["log_probs"])
