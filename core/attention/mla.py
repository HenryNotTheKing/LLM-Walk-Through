"""Multi-Head Latent Attention（MLA，DeepSeek-V2 风格教学实现）。

来源：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model* (2024, arXiv:2405.04434)。

核心思想：将 K/V 联合压缩到低秩潜空间再展开，推理时 KV cache 只需存储
潜向量（维度 ``kv_lora_rank``），而非完整的 per-head K/V。

教学简化：
    - 保留 Q 低秩分解（``q_down`` → ``q_up``）与 KV 低秩压缩；
    - K 的内容维（nope）与 RoPE 维（rope）解耦，RoPE 仅作用于 rope 部分；
    - 首版不做 weight absorption（推理时融合投影矩阵的优化 trick）；
    - 支持 ``attn_impl='eager'``（教学）与 ``'sdpa'``。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.norm.rmsnorm import RMSNorm
from core.position.rope import RotaryPositionalEmbedding, apply_rope


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        kv_lora_rank: int = 64,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        bias: bool = False,
        attn_impl: str = "sdpa",
        rope_theta: float = 1e4,
        rms_norm_eps: float = 1e-6,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.attn_impl = attn_impl
        self.dropout = dropout

        qk_rope_total = qk_rope_head_dim  # 单个共享 rope 头（广播到所有 Q 头）
        self.kv_down_proj = nn.Linear(
            n_embd, kv_lora_rank + qk_rope_total, bias=bias
        )
        self.kv_down_norm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        kv_up_dim = n_head * (qk_nope_head_dim + v_head_dim)
        self.kv_up_proj = nn.Linear(kv_lora_rank, kv_up_dim, bias=bias)

        if q_lora_rank is not None:
            self.q_down_proj = nn.Linear(n_embd, q_lora_rank, bias=bias)
            self.q_down_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_up_proj = nn.Linear(
                q_lora_rank, n_head * self.q_head_dim, bias=bias
            )
        else:
            self.q_down_proj = None
            self.q_up_proj = nn.Linear(n_embd, n_head * self.q_head_dim, bias=bias)

        self.o_proj = nn.Linear(n_head * v_head_dim, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if rope is None:
            rope = RotaryPositionalEmbedding(
                head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                base=rope_theta,
            )
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        if self.q_down_proj is not None:
            q_latent = self.q_down_norm(self.q_down_proj(x))
            q = self.q_up_proj(q_latent)
        else:
            q = self.q_up_proj(x)
        q = q.view(B, T, self.n_head, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv_down = self.kv_down_proj(x)
        c_kv, k_rope_in = torch.split(
            kv_down, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        c_kv = self.kv_down_norm(c_kv)
        kv_up = self.kv_up_proj(c_kv)
        kv_up = kv_up.view(
            B, T, self.n_head, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = torch.split(
            kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rope = k_rope_in.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = self.rope(T, device=x.device, dtype=q.dtype)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)
        k_rope = k_rope.expand(-1, self.n_head, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        scale = math.sqrt(self.q_head_dim)

        if self.attn_impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 / scale,
            )
        elif self.attn_impl == "eager":
            att = (q @ k.transpose(-2, -1)) / scale
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        else:
            raise ValueError(f"未知 attn_impl: {self.attn_impl}")

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)
        y = self.o_proj(y)
        return self.resid_dropout(y)

    def kv_cache_size_per_token(self) -> int:
        """每个 token 的 KV cache 元素数（潜空间 + rope 部分）。"""
        return self.kv_lora_rank + self.qk_rope_head_dim
