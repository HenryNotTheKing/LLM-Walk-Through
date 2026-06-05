"""ALiBi — Attention with Linear Biases.

Press et al., 2021. *Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation*.

Instead of adding position information to embeddings, ALiBi adds a
**negative linear bias** directly to attention scores:

    score'(i, j) = score(i, j) - m * |i - j|

where ``m`` is a head-specific slope. The bias penalizes distant tokens
linearly, giving the model a strong inductive bias toward locality.
Because the bias depends only on relative distance |i-j|, ALiBi
extrapolates to arbitrary sequence lengths without any training-time
modification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Return head-specific slopes of shape ``(n_heads,)``.

    The original paper uses a geometric sequence of slopes:
    ``2^(-8 * i / n_heads)`` for i = 1, ..., n_heads.
    """
    # i = 1 .. n_heads
    i = torch.arange(1, n_heads + 1, dtype=torch.float32)
    return (2 ** (-8.0 * i / n_heads))


class ALiBiPositionalBias(nn.Module):
    """Generate and cache ALiBi bias matrices.

    Args:
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length to pre-cache.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.slopes = get_alibi_slopes(n_heads)  # (n_heads,)

        # Pre-compute and cache the distance matrix.
        # bias[h, i, j] = -slope[h] * |i - j|
        self._cached_seq_len: int = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build ``(n_heads, seq_len, seq_len)`` bias tensor."""
        # Distance matrix: (seq_len, seq_len)
        positions = torch.arange(seq_len, dtype=torch.float32)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        dist = dist.abs()

        # Scale by slopes: (n_heads, 1, 1) * (1, seq_len, seq_len)
        slopes = self.slopes.view(-1, 1, 1)
        bias = -(slopes * dist.unsqueeze(0))  # (n_heads, seq_len, seq_len)

        self.register_buffer("bias_cached", bias, persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return ALiBi bias of shape ``(n_heads, seq_len, seq_len)``."""
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        return self.bias_cached[:, :seq_len, :seq_len]

    def apply_to_attention(
        self, attn_scores: torch.Tensor
    ) -> torch.Tensor:
        """Add ALiBi bias to attention scores in-place.

        Args:
            attn_scores: Shape ``(batch, n_heads, seq_len, seq_len)``.

        Returns:
            Bias-added scores of the same shape.
        """
        seq_len = attn_scores.size(-1)
        bias = self.forward(seq_len)  # (n_heads, seq_len, seq_len)
        # Broadcast across batch dimension.
        return attn_scores + bias.unsqueeze(0)
