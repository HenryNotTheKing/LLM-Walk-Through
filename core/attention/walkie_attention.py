"""Walkie 专用的因果自注意力。

特性：
    - **GQA**：``n_head`` 个 Q 头共享 ``n_head_kv`` 个 KV 头。
    - **QK-Norm**：对每头 Q/K 各做一次 RMSNorm，提升训练稳定性。
    - **RoPE**：在 attention 内部对 Q/K 应用旋转位置编码，不污染 token embedding。
    - **bias=False**：默认全部线性层无 bias。
    - 优先走 ``torch.nn.functional.scaled_dot_product_attention``，并保留 eager 路径作教学/对齐用。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.norm.rmsnorm import RMSNorm
from core.position.rope import RotaryPositionalEmbedding, apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头沿头维度复制 ``n_rep`` 次以匹配 Q 头数。

    输入 ``(B, H_kv, T, D)``，输出 ``(B, H_kv * n_rep, T, D)``。
    """
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class WalkieCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int | None = None,
        max_seq_len: int = 16384,
        dropout: float = 0.0,
        bias: bool = False,
        attn_impl: str = "sdpa",
        qk_norm: bool = True,
        rope_theta: float = 1e6,
        rope_scaling_factor: float = 1.0,
        rms_norm_eps: float = 1e-6,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if head_dim is None:
            if n_embd % n_head != 0:
                raise ValueError(f"n_embd={n_embd} 不能被 n_head={n_head} 整除")
            head_dim = n_embd // n_head
        if n_head % n_head_kv != 0:
            raise ValueError(f"n_head={n_head} 必须是 n_head_kv={n_head_kv} 的整数倍")

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.n_rep = n_head // n_head_kv
        self.attn_impl = attn_impl
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.max_seq_len = max_seq_len

        q_dim = n_head * head_dim
        kv_dim = n_head_kv * head_dim
        self.q_proj = nn.Linear(n_embd, q_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, kv_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_norm:
            # 对每头独立的 head_dim 维做 RMSNorm（按头独立学习缩放）
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # 允许多层共享同一个 RoPE 模块以省显存
        if rope is None:
            rope = RotaryPositionalEmbedding(
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                base=rope_theta,
                scaling_factor=rope_scaling_factor,
            )
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rope(T, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA：把 KV 头扩展到 Q 头数
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.attn_impl == "eager":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            raise ValueError(f"未知 attn_impl: {self.attn_impl}")

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        return y
