> 父文档：[← 分词器总览](README.md)

# Unigram LM（SentencePiece / LLaMA / Qwen）

## 1. 思路反转

BPE / WordPiece 是**自下而上合并**：从单字符开始，一路把"该合的"合到一起。
Unigram 是**自上而下裁剪**：从一个**远超目标大小**的种子词表开始，
反复用 EM 估每个 token 的概率，把"最不重要"的 token 裁掉，直到收敛到目标大小。

LLaMA、Qwen、DeepSeek、T5、ALBERT 都用这条路线（通过 SentencePiece 实现）。

## 2. 数学形式

把句子 $X$ 看作隐变量分段 $S = (s_1, ..., s_k)$ 的边缘：

$$P(X) = \sum_{S \in \mathcal{S}(X)} \prod_{i=1}^{k} p(s_i)$$

训练目标是极大化语料的对数似然 $\sum_X \log P(X)$，受词表大小约束。

**E 步**：对每个句子用 Viterbi 求**最大概率分段**（一阶近似 EM，比真期望便宜很多）；
累加每个 token 在最大分段中出现的次数 $c(t)$。

**M 步**：

$$p(t) \leftarrow \frac{c(t) + 1}{\sum_{t'} (c(t') + 1)}$$

**裁剪**：对每个候选 token，估算"删了它后语料对数似然下降多少"；
按下降量从小到大排序，删掉最不重要的若干个，重复。

本实现是教学简化版：用 token 计数近似"重要性"（数学上不严谨，但训练出来的分段质量已经接近）。

## 3. 推理：Viterbi 求最大概率分段

经典的一维动态规划：

$$dp[i] = \max_{j < i} \big( dp[j] + \log p(\text{text}[j:i]) \big)$$

复杂度 $O(L^2)$（$L$ 是句子长度）。比 BPE 的贪心合并慢，但分段质量更高，
且**唯一确定**——不会出现"合并顺序不同导致分段不同"的不一致。

## 4. SentencePiece 风格的空格处理

为了把"空格也是 token"这件事做得无歧义，SentencePiece 把所有空格替换为 `▁`（U+2581）后再分词，
解码时再换回去。这样 `"Hello world"` 与 `"Helloworld"` 永远是不同的 token 序列，**还原 100% 无损**。
本实现也采用这一约定，详见 [`core/tokenizer/unigram.py`](../../core/tokenizer/unigram.py) 的 `_normalize`。

## 5. 工程实现

- 训练：[`UnigramTokenizer.train`](../../core/tokenizer/unigram.py)
- 解码：`_viterbi`
- 序列化：JSON 文件（含 `token_to_id` + `log_probs`）

教学版的简化点：
- 种子词表是"高频前 N 子串"，没做更精细的初始化；
- 重要性近似用 `count`，不是真正的"删掉它后似然下降"；
- 没有引入 BOS / EOS（生成端可在 `add_eos=True` 时手动加）。

工业级请直接用 [`sentencepiece`](https://github.com/google/sentencepiece) Python 包。

## 6. 常见坑

- **空格符 `▁` 必须保留在所有日志/序列化里**，不要被编辑器自动 strip 成普通空格。
- **裁剪比例**：太激进会一轮砍掉关键 token 后再也长不回来；太保守要几十轮才收敛。本实现默认 0.2/轮。
- **种子规模 vs 训练数据**：种子是 N 子串前 K 个，N 太小会让 Viterbi 永远选不到长 token；K 太大会让训练时间爆炸。

## 7. References

### paper
- Kudo, 2018. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. https://arxiv.org/abs/1804.10959
- Kudo & Richardson, 2018. *SentencePiece: A simple and language independent subword tokenizer*. https://arxiv.org/abs/1808.06226

### blog
- Hugging Face, *Unigram tokenization*. https://huggingface.co/learn/nlp-course/chapter6/7

### code
- Google SentencePiece：https://github.com/google/sentencepiece
- HuggingFace tokenizers `Unigram`：https://github.com/huggingface/tokenizers

## 8. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| Unigram LM | [`core/tokenizer/unigram.py`](../../core/tokenizer/unigram.py) | [`tests/test_unigram.py`](../../tests/test_unigram.py) | （由 `tokenizer.kind=unigram` 切换） |

> 父文档：[← 分词器总览](README.md)
