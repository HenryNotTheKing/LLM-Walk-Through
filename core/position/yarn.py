"""YaRN — Yet another RoPE eNtension.

Peng et al., 2023. *YaRN: Efficient Context Window Extension of Large
Language Models*.

YaRN combines two ideas to extend RoPE's context window:
1.  NTK-aware frequency interpolation ("by-parts"): high-frequency components
    are interpolated more aggressively than low-frequency ones.
2.  Attention temperature scaling: scales attention logits by a constant to
    counteract the soft-max entropy collapse caused by longer contexts.

This module provides a drop-in replacement for ``RotaryPositionalEmbedding``
with a configurable ``scale_factor``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from core.position.rope import apply_rope, _rotate_half


def _yarn_find_correction_dim(
    num_rotations: float,
    d_model: int,
    base: float = 1e6,
    max_position_embeddings: int = 2048,
) -> float:
    """Inverse of the RoPE frequency formula to find the dimension index
    that corresponds to a given number of rotations.
    """
    return (d_model * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    d_model: int,
    base: float = 1e6,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, d_model, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, d_model, base, max_position_embeddings))
    return max(low, 0), min(high, d_model - 1)


class YarnRotaryPositionalEmbedding(nn.Module):
    """YaRN-extended RoPE with NTK-by-parts interpolation.

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to pre-cache.
        base: RoPE base (θ).
        scale_factor: Context extension factor, e.g. 4.0 to extend 2k → 8k.
        orig_max_seq_len: Original training-time max sequence length.
        beta_fast: Fast wavelength threshold (default 32).
        beta_slow: Slow wavelength threshold (default 1).
        attn_factor: Temperature scaling factor for attention logits.
                     If ``None``, auto-computed as ``0.1 * math.log(scale_factor) + 1.0``.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 8192,
        base: float = 1e6,
        scale_factor: float = 1.0,
        orig_max_seq_len: int = 2048,
        beta_fast: int = 32,
        beta_slow: int = 1,
        attn_factor: float | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self.scale_factor = scale_factor
        self.orig_max_seq_len = orig_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Attention temperature scaling factor.
        if attn_factor is None:
            # Heuristic from the paper: ~0.1*log(scale) + 1.0
            self.attn_factor = 0.1 * math.log(scale_factor) + 1.0 if scale_factor > 1.0 else 1.0
        else:
            self.attn_factor = attn_factor

        # Build NTK-by-parts scaled frequencies.
        self._build_scaled_freqs()
        self._cached_seq_len: int = 0
        self._build_cache(max_seq_len, device=self.inv_freq.device, dtype=torch.float32)

    def _build_scaled_freqs(self) -> None:
        """Compute per-dimension interpolation ramps and the scaled inv_freq."""
        half = self.head_dim // 2
        freqs = 1.0 / (
            self.base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / self.head_dim)
        )

        if self.scale_factor <= 1.0:
            self.register_buffer("inv_freq", freqs, persistent=False)
            self.register_buffer("freq_scale", torch.ones(half, dtype=torch.float32), persistent=False)
            return

        # Find the dimension range that should be interpolated.
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            half,
            base=self.base,
            max_position_embeddings=self.orig_max_seq_len,
        )

        # Linear ramp: 0 at `low`, 1 at `high`.
        ramp = torch.zeros(half, dtype=torch.float32)
        if high > low:
            ramp[low : high + 1] = torch.linspace(0.0, 1.0, high - low + 1, dtype=torch.float32)
        elif half > 0:
            # Degenerate case: set all to 1.0 if the window covers everything.
            ramp.fill_(1.0)
        ramp = torch.clamp(ramp, 0.0, 1.0)

        # Apply the interpolation factor.
        freq_scale = 1.0 / self.scale_factor * (1.0 - ramp) + ramp
        scaled_freqs = freqs * freq_scale

        self.register_buffer("inv_freq", scaled_freqs, persistent=False)
        self.register_buffer("freq_scale", freq_scale, persistent=False)

    def _build_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", t, self.inv_freq.to(device=device))
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

    def apply_attention_scaling(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Apply YaRN attention temperature scaling.

        Call this on the raw attention logits (before softmax) when using YaRN.
        """
        return attn_scores / self.attn_factor
