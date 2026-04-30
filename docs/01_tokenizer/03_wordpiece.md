> 父文档：[← 分词器总览](README.md)

# WordPiece（BERT 系列）

## 1. 与 BPE 的核心差异

| | BPE | WordPiece |
| --- | --- | --- |
| 合并准则 | 取**频次**最高的 pair | 取 `score = freq(pair) / (freq(left) * freq(right))` 最大的 pair |
| 续接标记 | 用 `</w>` 标记词**尾** | 用 `##` 标记词内**非首**子词 |
| OOV 处理 | 拆回单字符 / `<|unk|>` | 整词降级为 `[UNK]`（更激进） |
| 编码策略 | 按 merges 顺序贪心合并 | 最长前缀匹配 |

BPE 选"最常一起出现"的对；WordPiece 选"互信息最大"的对——
也就是说，WordPiece 倾向于合并那些**一起出现远超独立出现**的 pair（共现强相关）。
这是对 *合并后的语言模型似然提升* 的一阶近似，因此称之为"似然驱动"。

## 2. 算法

**训练**：
1. 初始化词表 = 特殊 token（`[PAD] [UNK] [CLS] [SEP] [MASK]`）+ 出现过的字符
   （首字符不带 `##`，其它带 `##`）。
2. 把每个词拆成 `[c0, ##c1, ##c2, ...]`；
3. 反复扫所有相邻 pair 算 score，选 score 最大的 pair 合并；
4. 直到 vocab 达到目标大小。

**编码**（推理）：对每个词做最长前缀匹配，第一段不带 `##`，后续段带 `##`；
任一段无法匹配则整词输出 `[UNK]`。

## 3. 工程实现

源码：[`core/tokenizer/wordpiece.py`](../../core/tokenizer/wordpiece.py)。教学版：

- 预切分用简单的 `\w+|[^\w\s]+` 正则并小写化；与 BERT 全套 `BasicTokenizer`（NFD/CJK/重音号去除）相比更朴素。
- 训练时省略了"似然真值"的精确推导，用上面给出的 score 近似；这是 HF NLP Course 推荐的教学公式，与 BERT 论文方向一致。

## 4. 常见坑

- **大小写**：BERT 有 cased / uncased 两版，搞错会导致词表不匹配。
- **`##` 是约定不是常量**：实现时整个 pipeline 必须保持一致，乱改会破坏编码。
- **整词降 `[UNK]` vs 部分降**：原生 WordPiece 是整词降级；某些"修正版"会逐字符降，不再算 WordPiece。

## 5. References

### paper
- Schuster & Nakajima, 2012. *Japanese and Korean Voice Search.* IEEE ICASSP.
- Wu et al., 2016. *Google's Neural Machine Translation System*. https://arxiv.org/abs/1609.08144
- Devlin et al., 2018. *BERT*. https://arxiv.org/abs/1810.04805

### blog
- Hugging Face, *WordPiece tokenization*. https://huggingface.co/learn/nlp-course/chapter6/6

### code
- HuggingFace tokenizers `WordPiece`：https://github.com/huggingface/tokenizers
- Google BERT 官方：https://github.com/google-research/bert/blob/master/tokenization.py

## 6. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| WordPiece | [`core/tokenizer/wordpiece.py`](../../core/tokenizer/wordpiece.py) | [`tests/test_wordpiece.py`](../../tests/test_wordpiece.py) | （由 `tokenizer.kind=wordpiece` 切换） |

> 父文档：[← 分词器总览](README.md)
