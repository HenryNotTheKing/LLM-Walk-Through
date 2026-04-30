# Causal Multi-Head Self-Attention

> Decoder-only Transformer 的核心：让每个位置的输出只看见自己和左侧的 token。

## 1. 计算

把输入投影到 $Q, K, V \in \mathbb{R}^{B \times H \times T \times d_h}$，然后

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V$$

$M$ 是因果 mask（上三角为 $-\infty$）。多头之后 concat 投回 $n\_embd$。

## 2. 三种实现路径

| `attn_impl` | 路径 | 备注 |
| --- | --- | --- |
| `sdpa`（默认） | `F.scaled_dot_product_attention(q,k,v, is_causal=True)` | CUDA 上自动选 Flash / MemEff 内核；CPU/MPS 走数学路径 |
| `eager` | 手写 softmax + mask | 教学/对齐用，便于打印中间张量 |
| `flash`（V1+） | `flash_attn` 包 | 仅 CUDA 上；首版未启用 |

`sdpa` 与 `eager` 在 dropout=0 下应数值一致，见 [`tests/test_gpt2_model.py::test_attn_impl_eager_matches_sdpa`](../tests/test_gpt2_model.py)。

## 3. 与 GPT-2 实现的几个细节

- `c_attn`：把 q/k/v 三个投影合并为一次 `nn.Linear(n_embd, 3*n_embd)`。
- `c_proj`：输出投影；GPT-2 论文对 `c_proj.weight` 做 $1/\sqrt{2N}$ 缩放初始化。
- HuggingFace 的 `Conv1D` 等价于 transpose 过的 `nn.Linear`，权重加载时需要转置。

## 4. 后续将替换为什么

- **MQA / GQA**：把 K/V 头数减少为 1 / 少数组，对推理 KV 显存最友好。
- **RoPE 替换 absolute PE**：把位置注入到 q/k 的旋转里，不再加到输入 embedding 上。
- **Flash-Attention-2**：在 CUDA 上替换 SDPA，进一步降低显存与提速。

## 5. References

### paper
- Vaswani et al., 2017. *Attention Is All You Need*.
- Dao et al., 2022/2023. *FlashAttention / FlashAttention-2*. https://arxiv.org/abs/2205.14135 / https://arxiv.org/abs/2307.08691
- Shazeer, 2019. *Multi-Query Attention*. https://arxiv.org/abs/1911.02150
- Ainslie et al., 2023. *GQA: Training Generalized Multi-Query Transformer Models*. https://arxiv.org/abs/2305.13245

### blog
- Lilian Weng, *The Transformer Family*. https://lilianweng.github.io/posts/2018-06-24-attention/
- PyTorch, *scaled_dot_product_attention*. https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### code
- HuggingFace GPT-2 attention：https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
- flash-attn：https://github.com/Dao-AILab/flash-attention

## 6. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| Causal MHA | [`core/attention/mha.py`](../core/attention/mha.py) | [`tests/test_gpt2_model.py`](../tests/test_gpt2_model.py) | [`configs/model/gpt2_124m.yaml`](../configs/model/gpt2_124m.yaml) |
