> 父文档：[← 分词器总览](README.md)

# Byte-level BPE（真 GPT-2 / GPT-3 算法）

## 1. 与字符级 BPE 的差异

[字符级 BPE](01_bpe.md) 在**字符序列**上做合并，遇到训练时没见过的字符就走 `<|unk|>`。
**字节级 BPE 在字节序列上做合并**，初始词表是 256 个字节的双射 unicode 表示，因此**不可能 OOV**——
任何 utf-8 字符串都可以被无损编码 / 解码。

GPT-2 / GPT-3 / RoBERTa / OPT / Mistral 都使用这一变体。

## 2. bytes ↔ unicode 映射

直接把 `bytes` 当 token 训练有两个问题：
1. 大量字节是不可打印控制字符（`\x00`、`\t`、`\n`），调试日志非常痛苦；
2. 训练 BPE 时空白本身也是合并对象，会和 GPT-2 的"前导空格当词的一部分"约定打架。

OpenAI 官方做法：把 256 个字节双射到一组**可打印**的 unicode 字符
（`!`–`~` 的 ASCII 可见区 + Latin 扩展区 + 偏移到 `256+i` 的 68 个字符）。
实现就在 [`core/tokenizer/byte_bpe.py`](../../core/tokenizer/byte_bpe.py) 的 `bytes_to_unicode()`，
与 OpenAI `encoder.py` 完全一致。

## 3. 训练流程

```
text
 │ utf-8 编码
 ▼
bytes
 │ 查 bytes_to_unicode 映射
 ▼
unicode 字符序列（每段对应一个预切分 piece）
 │ 与字符级 BPE 完全相同的合并循环
 ▼
{vocab, merges}
```

预切分用与 GPT-2 完全一致的正则把文本粗分为单词/标点段；段内做 BPE，段间不合并。

## 4. 推理流程

```
text → utf-8 bytes → unicode chars → BPE 编码 → token ids
                                                   │
                                                   ▼
                                            token ids → unicode chars → bytes → utf-8 → text
```

由于映射是双射、字节集是闭集，往返**无损**。

## 5. 与 `gpt2` 官方权重的关系

**自训** Byte-BPE 不会得到与 GPT-2 官方相同的 50257 词表（语料和合并顺序都不同）。
要做权重对齐请用 [`GPT2BPETokenizer`](../../core/tokenizer/gpt2_bpe.py)，它直接加载 HF 官方 `vocab.json` / `merges.txt`。
但**算法本身**与本文件实现的字节级 BPE 是同一个东西。

## 6. 常见坑

- **字节映射表必须与训练时一致**：自己改了 `bytes_to_unicode` 之后，老 checkpoint 的 token 含义会全部错位。
- **decode 时跳过特殊 token**：特殊 token（如 `<|endoftext|>`）的字符不在字节映射表里，混入字符流会导致字节恢复失败；本实现里 `decode()` 显式跳过。
- **预切分的领导空格**：`" world"` 与 `"world"` 是不同的 token；这正是 GPT-2 把空格当词一部分的设计，不要"贴心"地 strip 掉。

## 7. References

### paper
- Sennrich et al., 2016. *Neural Machine Translation of Rare Words with Subword Units.* https://arxiv.org/abs/1508.07909
- Radford et al., 2019. *Language Models are Unsupervised Multitask Learners* (GPT-2 tech report).

### blog
- Karpathy, *Let's build the GPT Tokenizer*. https://www.youtube.com/watch?v=zduSFxRajkE
- Hugging Face, *Byte-Pair Encoding tokenization*. https://huggingface.co/learn/nlp-course/chapter6/5

### code
- 官方参考实现：https://github.com/openai/gpt-2/blob/master/src/encoder.py
- minBPE（教学复刻）：https://github.com/karpathy/minbpe

## 8. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| Byte-level BPE | [`core/tokenizer/byte_bpe.py`](../../core/tokenizer/byte_bpe.py) | [`tests/test_byte_bpe.py`](../../tests/test_byte_bpe.py) | （由 `tokenizer.kind=byte_bpe` 切换） |
| 字节↔unicode 映射 | `bytes_to_unicode()` in [`core/tokenizer/byte_bpe.py`](../../core/tokenizer/byte_bpe.py) | [`tests/test_byte_bpe.py`](../../tests/test_byte_bpe.py) | — |

> 父文档：[← 分词器总览](README.md)
