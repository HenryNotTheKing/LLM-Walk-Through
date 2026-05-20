"""Rotary Position Embedding（RoPE，Walkie-Code-1B 专用）。

实现要点：
    - 频率 :math:`\theta_i = base^{-2i/d}`，i ∈ [0, d/2)。
    - 仅在 attention 内部对 Q/K 应用旋转，不参与 token embedding 加和。
    - cos/sin 表按需扩展并缓存到 ``register_buffer``，避免每个 forward 重算。
    - 支持 ``rope_scaling_factor`` 做线性 NTK-style 拉伸（位置除以 factor），
      首版默认 1.0；做长上下文外推时再调整。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_rope_freqs(head_dim: int, base: float = 1e6) -> torch.Tensor:
    """返回长度 ``head_dim // 2`` 的频率向量（fp32）。"""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim 必须是偶数，得到 {head_dim}")
    half = head_dim // 2
    idx = torch.arange(half, dtype=torch.float32)
    return 1.0 / (base ** (2.0 * idx / head_dim))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将最后一维拆成两半并做 (x1, x2) -> (-x2, x1)。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """对 ``x`` 的最后一维做旋转。

    ``x`` 形状 ``(B, H, T, D)``，``cos``/``sin`` 形状 ``(T, D)`` 或可广播形状。
    """
    # 把 cos/sin 广播到 (1, 1, T, D)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    half = x.size(-1) // 2
    x1, x2 = x.chunk(2, dim=-1)
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    return torch.cat(
        (x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half),
        dim=-1,
    )


class RotaryPositionalEmbedding(nn.Module):
    """生成并缓存 RoPE 的 cos/sin 表。"""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 1e6,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        freqs = compute_rope_freqs(head_dim, base=base)
        self.register_buffer("inv_freq", freqs, persistent=False)
        self._cached_seq_len: int = 0
        self._build_cache(max_seq_len, device=freqs.device, dtype=torch.float32)

    def _build_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
        # (T, D/2)
        freqs = torch.einsum("t,d->td", t, self.inv_freq.to(device=device))
        # 拼成 (T, D)：前半与后半同频，使 _rotate_half 与 cos/sin 对齐
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._cached_seq_len = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.inv_freq.device
        dtype = dtype or torch.float32
        if (
            seq_len > self._cached_seq_len
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            self._build_cache(max(seq_len, self._cached_seq_len), device=device, dtype=dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
