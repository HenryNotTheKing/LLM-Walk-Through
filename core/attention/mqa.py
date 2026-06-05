"""Multi-Query Attention（MQA）因果自注意力。

来源：Shazeer, 2019（多查询注意力）；PaLM (Chowdhery et al., 2022) 大规模采用。

MQA 是 GQA 的极端情形：所有 ``n_head`` 个 Q 头共享**唯一**一组 K/V 头
（``n_head_kv = 1``）。KV cache 从 O(n_head · T · d_h) 降至 O(T · d_h)，
推理显存与带宽压力最小，但表达能力弱于 MHA/GQA。

本模块在 ``WalkieCausalSelfAttention`` 上固定 ``n_head_kv=1``，保留
QK-Norm、RoPE 与 ``attn_impl`` 多后端，便于与 GQA notebook 对照学习。
"""

from __future__ import annotations

from core.attention.walkie_attention import WalkieCausalSelfAttention


class MQACausalSelfAttention(WalkieCausalSelfAttention):
    """MQA：``n_head_kv=1`` 的 Walkie 风格因果自注意力。"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int | None = None,
        max_seq_len: int = 16384,
        dropout: float = 0.0,
        bias: bool = False,
        attn_impl: str = "sdpa",
        qk_norm: bool = True,
        rope_theta: float = 1e6,
        rope_scaling_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        rope=None,
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            n_head_kv=1,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bias=bias,
            attn_impl=attn_impl,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rms_norm_eps=rms_norm_eps,
            rope=rope,
        )
