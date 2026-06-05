"""线性因果自注意力（Linear Transformer 风格）。

来源：Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive
Transformers with Linear Attention* (2020, arXiv:2006.16236)。

用正定特征映射 φ 替代 softmax，使注意力可写为：
    Attn(Q, K, V) = φ(Q) (φ(K)ᵀ V) / (φ(Q) (φ(K)ᵀ 1))

因果性通过前缀累积实现，单步推理只需维护状态 (KV_state, K_sum)，
复杂度 O(T) 而非 O(T²)。

默认核：φ(x) = elu(x) + 1（保证非负，便于归一化）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_attention_feature_map(x: torch.Tensor) -> torch.Tensor:
    """默认特征映射 φ(x) = elu(x) + 1。"""
    return F.elu(x) + 1.0


class LinearCausalAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        attn_impl: str = "eager",
    ) -> None:
        super().__init__()
        if head_dim is None:
            if n_embd % n_head != 0:
                raise ValueError(f"n_embd={n_embd} 不能被 n_head={n_head} 整除")
            head_dim = n_embd // n_head

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout = dropout
        self.attn_impl = attn_impl

        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_head * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def _linear_attention_eager(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """因果线性注意力：按时间步前缀累积。"""
        B, H, T, D = q.shape
        q_feat = linear_attention_feature_map(q)
        k_feat = linear_attention_feature_map(k)

        outputs = []
        kv_state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        k_sum = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)

        for t in range(T):
            kt = k_feat[:, :, t, :]  # (B, H, D)
            vt = v[:, :, t, :]
            qt = q_feat[:, :, t, :]

            kv_state = kv_state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            k_sum = k_sum + kt

            num = torch.matmul(qt.unsqueeze(-2), kv_state).squeeze(-2)
            denom = (qt * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            outputs.append(num / denom)

        return torch.stack(outputs, dim=2)

    def _linear_attention_recurrent(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """与 eager 等价，显式 recurrent 路径（教学用）。"""
        return self._linear_attention_eager(q, k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)

        if self.attn_impl in ("eager", "recurrent"):
            y = self._linear_attention_eager(q, k, v)
        else:
            raise ValueError(
                f"未知 attn_impl: {self.attn_impl}；线性注意力仅支持 eager/recurrent"
            )

        B, T, _ = x.shape
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.o_proj(y)
        return self.resid_dropout(y)

    @torch.no_grad()
    def forward_step(
        self,
        x: torch.Tensor,
        kv_state: torch.Tensor | None = None,
        k_sum: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步递推推理：输入 ``(B, 1, C)``，返回 (out, new_kv_state, new_k_sum)。"""
        q, k, v = self._project(x)
        q_feat = linear_attention_feature_map(q)
        k_feat = linear_attention_feature_map(k)

        B, H, _, D = q.shape
        if kv_state is None:
            kv_state = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        if k_sum is None:
            k_sum = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)

        kt = k_feat[:, :, 0, :]
        vt = v[:, :, 0, :]
        qt = q_feat[:, :, 0, :]

        kv_state = kv_state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        k_sum = k_sum + kt

        num = torch.matmul(qt.unsqueeze(-2), kv_state).squeeze(-2)
        denom = (qt * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        y = (num / denom).unsqueeze(2)

        y = y.transpose(1, 2).contiguous().view(B, 1, self.n_head * self.head_dim)
        y = self.o_proj(y)
        return y, kv_state, k_sum
