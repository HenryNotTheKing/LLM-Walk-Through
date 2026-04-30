"""Causal Multi-Head Self-Attention（GPT-2 原版结构）。

实现要点：
    - 一次 ``Linear`` 把 q/k/v 一起算出来（与 GPT-2 ``c_attn`` 对齐）。
    - 默认走 ``torch.nn.functional.scaled_dot_product_attention``，
      在 CUDA 上会自动选择 Flash/MemEff 内核；CPU/MPS 也能正确运行。
    - 提供 ``attn_impl='eager'`` 的手写路径，用于教学和 sanity check。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = True,
        attn_impl: str = "sdpa",
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) 必须能被 n_head ({n_head}) 整除")
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.attn_impl = attn_impl

        # GPT-2 ``c_attn``：一次性投影出 q/k/v
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # eager 路径用的下三角 mask（sdpa 路径用 is_causal=True，不需要这个）
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.attn_impl == "eager":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~self._causal_mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            raise ValueError(f"未知 attn_impl: {self.attn_impl}")

        # (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
