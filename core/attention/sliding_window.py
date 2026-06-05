"""滑动窗口因果自注意力（Sliding Window Attention）。

来源：Mistral 7B 技术报告；Longformer (Beltagy et al., 2020, arXiv:2004.05150)。

每个 query 位置 ``i`` 只能 attend 到窗口
``[max(0, i - window_size + 1), i]`` 内的 key，复杂度从 O(T²) 降至
O(T · window_size)，适合长上下文推理。

实现基于 Walkie GQA 投影 + QK-Norm + RoPE，通过 banded causal mask 限制
注意力范围。支持 ``attn_impl='eager'`` 与 ``'sdpa'``。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention.walkie_attention import repeat_kv
from core.norm.rmsnorm import RMSNorm
from core.position.rope import RotaryPositionalEmbedding, apply_rope


def make_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """构造滑动窗口因果掩码，形状 ``(seq_len, seq_len)``，True 表示允许 attend。"""
    idx = torch.arange(seq_len, device=device)
    row = idx.unsqueeze(1)
    col = idx.unsqueeze(0)
    causal = col <= row
    in_window = (row - col) < window_size
    return causal & in_window


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        window_size: int = 4096,
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
        self.window_size = window_size
        self.attn_impl = attn_impl
        self.dropout = dropout

        q_dim = n_head * head_dim
        kv_dim = n_head_kv * head_dim
        self.q_proj = nn.Linear(n_embd, q_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, kv_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, kv_dim, bias=bias)
        self.o_proj = nn.Linear(q_dim, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

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

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        window_mask = make_sliding_window_mask(T, self.window_size, x.device)

        if self.attn_impl == "sdpa":
            # SDPA bool mask: True = keep
            attn_mask = window_mask.unsqueeze(0).unsqueeze(0)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        elif self.attn_impl == "eager":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~window_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            raise ValueError(f"未知 attn_impl: {self.attn_impl}")

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        return self.resid_dropout(y)
