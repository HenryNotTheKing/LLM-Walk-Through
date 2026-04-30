> 父文档：[← 分词器总览](README.md)

# 教学版 BPE 分词器

> 把任意文本切成一串子词单元，让"常见词"进入词表，"罕见词/未见词"用更细粒度拼出来。

## 1. 问题背景

字符级建模序列太长、词级建模 OOV（未登录词）严重；BPE 用频次驱动的合并算法，
让最高频的子串成为单一 token，平衡序列长度与覆盖率。GPT-2 / GPT-3 / LLaMA 系列
都使用 BPE 或其字节级变体。

## 2. 算法

1. 初始化：每个词拆成字符序列，词尾补 `</w>`。
2. 反复统计相邻 token 对的全局频次，把最高频的对合并为一个新 token，写入 merges。
3. 重复直到词表达到目标大小或没有可合并对。

编码时按 merges 的顺序贪心合并；解码时把所有 token 拼接，再去掉 `</w>`。

## 3. 与 GPT-2 官方 BPE 的关系

教学版自训 BPE **不会**得到与 GPT-2 官方相同的 50257 词表，且本实现是**字符级**而非字节级。
要做 HuggingFace GPT-2 logits 对齐，请使用 [`GPT2BPETokenizer`](../../core/tokenizer/gpt2_bpe.py)
（直接加载官方 `vocab.json` / `merges.txt`），或参见同目录下的 [Byte-level BPE](02_byte_level_bpe.md)
来理解"真正的 GPT-2 算法"。

## 4. 工程实现

- 训练 / 编码 / 解码：[`core.tokenizer.bpe.BPETokenizer`](../../core/tokenizer/bpe.py)
- 预切分用与 GPT-2 一致的正则把文本粗分为单词/标点段，再做 BPE 合并。
- 特殊 token：`<|endoftext|>`、`<|unk|>`。

## 5. 常见坑

- 未做预切分时，BPE 会跨词合并，得到很奇怪的 token；GPT-2 也是先做预切分。
- 训练语料过小会导致词表达不到目标大小（合并频次降到 1 时退出）。
- decode 时若直接 `"".join(tokens)` 而不去掉 `</w>` 标记，会留下伪迹。

## 6. References

### paper
- Sennrich et al., 2016. *Neural Machine Translation of Rare Words with Subword Units.* https://arxiv.org/abs/1508.07909
- Radford et al., 2019. *Language Models are Unsupervised Multitask Learners* (GPT-2 tech report).

### blog
- Hugging Face, *Byte-Pair Encoding tokenization*. https://huggingface.co/learn/nlp-course/chapter6/5
- Karpathy, *Let's build the GPT Tokenizer*. https://www.youtube.com/watch?v=zduSFxRajkE

### code
- 官方参考：https://github.com/openai/gpt-2/blob/master/src/encoder.py
- 工程实现：https://github.com/huggingface/tokenizers

## 7. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| 教学版 BPE | [`core/tokenizer/bpe.py`](../../core/tokenizer/bpe.py) | [`tests/test_bpe.py`](../../tests/test_bpe.py) | [`configs/train/pretrain_tiny.yaml`](../../configs/train/pretrain_tiny.yaml) |
| GPT-2 官方 BPE 兼容 | [`core/tokenizer/gpt2_bpe.py`](../../core/tokenizer/gpt2_bpe.py) | [`tests/test_hf_gpt2_parity.py`](../../tests/test_hf_gpt2_parity.py) | — |

> 父文档：[← 分词器总览](README.md)
