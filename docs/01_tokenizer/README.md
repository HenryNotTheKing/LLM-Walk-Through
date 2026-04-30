# 分词器（Tokenizer）总览

> 把任意文本切成离散单元（token）的算法。LLM 性能、上下文长度、跨语言能力都与之强相关。

本父文档给出**全景对照与决策建议**；具体算法各占一篇子文档。

## 子文档

| 子文档 | 算法 | 代表模型 | 教学定位 |
| --- | --- | --- | --- |
| [01 · BPE（教学版）](01_bpe.md) | 字符级 BPE + `</w>` 词尾标记 | — | 用最简洁的代码讲清 BPE 的合并循环 |
| [02 · Byte-level BPE](02_byte_level_bpe.md) | **真正的** GPT-2 / GPT-3 算法 | GPT-2/3, RoBERTa | 字节→Unicode 映射，0 OOV |
| [03 · WordPiece](03_wordpiece.md) | 似然最大化合并 + `##` 续接前缀 | BERT / DistilBERT | 与 BPE 的"合并准则"差异 |
| [04 · Unigram LM (SentencePiece)](04_unigram.md) | EM 训练 + Viterbi 编码 | T5 / **LLaMA / Qwen / DeepSeek** | LLaMA 系列实际算法 |

## 一张表对比

| 维度 | BPE（字符） | Byte-BPE | WordPiece | Unigram |
| --- | --- | --- | --- | --- |
| 训练目标 | 频次最大合并 | 频次最大合并（在字节空间） | 似然最大合并 | 似然最大保留（删词表） |
| OOV | 取决于字符表 | **不可能** OOV（字节是闭集） | `[UNK]` | 通过子词回退 |
| 编码 | 贪心按 merges 顺序合并 | 同 BPE | 最长匹配优先（前缀 `##`） | Viterbi 取最优分段 |
| 序列化 | `vocab + merges` | `vocab + merges + bytes_map` | `vocab` | `vocab + scores` |

## 在本项目中如何选

| 场景 | 用谁 |
| --- | --- |
| tiny shakespeare 教学跑通 | [BPE 教学版](01_bpe.md)（默认） |
| 与 HuggingFace `gpt2` logits 对齐 | [`GPT2BPETokenizer`](../../core/tokenizer/gpt2_bpe.py)（基于 HF 官方） |
| 复刻"真 GPT-2 训练数据流" | [Byte-level BPE](02_byte_level_bpe.md) |
| 后续做现代 LLM 风格预训练 | [Unigram](04_unigram.md) |
| 学习 BERT 系列 | [WordPiece](03_wordpiece.md) |

## 统一接口

所有 tokenizer 均继承 [`core.tokenizer.base.BaseTokenizer`](../../core/tokenizer/base.py)，
通过 [`build_tokenizer(kind, ...)`](../../core/tokenizer/__init__.py) 工厂创建。

```python
from core.tokenizer import build_tokenizer

tok = build_tokenizer("byte_bpe", vocab_size=2048)
tok.train(["some corpus..."])
ids = tok.encode("hello world")
text = tok.decode(ids)
tok.save("path/to/tokenizer.json")
```

## Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| 抽象基类 | [`core/tokenizer/base.py`](../../core/tokenizer/base.py) | [`tests/test_tokenizer_base.py`](../../tests/test_tokenizer_base.py) | — |
| 工厂 | [`core/tokenizer/__init__.py`](../../core/tokenizer/__init__.py) | [`tests/test_tokenizer_base.py`](../../tests/test_tokenizer_base.py) | [`configs/train/pretrain_tiny.yaml`](../../configs/train/pretrain_tiny.yaml) |

## References

### paper
- 见各子文档。

### blog
- HuggingFace 课程，*Summary of the tokenizers*. https://huggingface.co/learn/nlp-course/chapter6/

### code
- HuggingFace `tokenizers`：https://github.com/huggingface/tokenizers
- Google SentencePiece：https://github.com/google/sentencepiece
