# 原始 GPT-2 baseline

本文是 V0 实现的总览。它对应：[`core/model/gpt2.py`](../core/model/gpt2.py)。

## 1. 结构总览

```
tokens ──► wte (token embedding)
           +
           wpe (learned absolute PE)
           │
           ▼
    [Block × N]
       │
       ▼
    LayerNorm (ln_f)
       │
       ▼
    lm_head (与 wte 共享权重)
```

每个 Block 是 **Pre-LN 残差结构**：

$$x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$$
$$x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$$

## 2. 子模块

| 子模块 | 实现 |
| --- | --- |
| Token Embedding | `nn.Embedding(vocab, n_embd)` |
| Positional Embedding | [`LearnedPositionalEmbedding`](../core/position/learned.py) |
| Norm | [`LayerNorm`](../core/norm/layer_norm.py)（Pre-LN） |
| Attention | [`CausalSelfAttention`](../core/attention/mha.py)（SDPA 默认 / eager 备选） |
| MLP | [`GeluMLP`](../core/ffn/mlp.py)（4× hidden, GELU-tanh 近似） |
| LM 头 | `nn.Linear(n_embd, vocab, bias=False)`，与 wte 共享权重 |

## 3. 与 HuggingFace GPT-2 的对齐

`GPT2LMHeadModel.from_pretrained_hf("gpt2")` 会：

1. 用 HF 的 `GPT2Config` 推出本项目的 `GPT2Config`；
2. 加载 HF 权重；
3. 把 HF 的 `Conv1D` 权重（形状 `(in, out)`）转置回 `nn.Linear` 的 `(out, in)`；
4. 拷到本模型的 state dict（`lm_head.weight` 走 tied 共享）。

logits 对齐测试见 [`tests/test_hf_gpt2_parity.py`](../tests/test_hf_gpt2_parity.py)。

## 4. 默认尺寸

| 配置 | layer | head | embd | block |
| --- | --- | --- | --- | --- |
| [`gpt2_tiny`](../configs/model/gpt2_tiny.yaml) | 2 | 2 | 64 | 64 |
| [`gpt2_124m`](../configs/model/gpt2_124m.yaml) | 12 | 12 | 768 | 1024 |
| [`gpt2_350m`](../configs/model/gpt2_350m.yaml) | 24 | 16 | 1024 | 1024 |
| [`gpt2_774m`](../configs/model/gpt2_774m.yaml) | 36 | 20 | 1280 | 1024 |
| [`gpt2_1558m`](../configs/model/gpt2_1558m.yaml) | 48 | 25 | 1600 | 1024 |

## 5. References

### paper
- Radford et al., 2019. *Language Models are Unsupervised Multitask Learners*.
- Vaswani et al., 2017. *Attention Is All You Need*. https://arxiv.org/abs/1706.03762

### blog
- Karpathy, *nanoGPT*. https://github.com/karpathy/nanoGPT
- *The Illustrated GPT-2*. https://jalammar.github.io/illustrated-gpt2/

### code
- OpenAI 官方：https://github.com/openai/gpt-2
- HuggingFace 实现：https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

## 6. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| GPT2 模型类 | [`core/model/gpt2.py`](../core/model/gpt2.py) | [`tests/test_gpt2_model.py`](../tests/test_gpt2_model.py) | [`configs/model/gpt2_124m.yaml`](../configs/model/gpt2_124m.yaml) |
| HF 权重对齐 | [`core/model/gpt2.py`](../core/model/gpt2.py)（`from_pretrained_hf`） | [`tests/test_hf_gpt2_parity.py`](../tests/test_hf_gpt2_parity.py) | — |
| 训练入口 | [`train/pretrain.py`](../train/pretrain.py) | [`tests/test_training_step.py`](../tests/test_training_step.py) | [`configs/train/pretrain_tiny.yaml`](../configs/train/pretrain_tiny.yaml) |
