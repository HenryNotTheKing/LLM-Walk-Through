"""Sinusoidal (fixed) positional encoding.

Vaswani et al. (2017) *Attention Is All You Need*.

Unlike learned embeddings, sinusoidal PE uses pre-defined trigonometric
functions of varying frequencies, allowing the model to attend to relative
positions via linear combinations of sines and cosines.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Computes:
        PE(pos, 2i)   = sin(pos / base^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / base^(2i / d_model))

    The encoding is added directly to token embeddings, i.e. it is an
    *absolute* position representation (same family as learned PE).
    """

    def __init__(self, d_model: int, max_seq_len: int = 8192, base: float = 1e4) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute and register as non-persistent buffer so it moves with
        # .to(device) but is not saved in the state_dict (fixed values).
        pe = self._build_pe(max_seq_len)
        self.register_buffer("pe", pe, persistent=False)

    def _build_pe(self, seq_len: int) -> torch.Tensor:
        """Return shape (seq_len, d_model)."""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(self.base) / self.d_model)
        )  # (d_model // 2,)

        pe = torch.zeros(seq_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """Return positional encoding of shape ``(seq_len, d_model)``."""
        if seq_len > self.pe.size(0):
            # Lazily expand the buffer if requested sequence is longer.
            self.pe = self._build_pe(seq_len).to(self.pe.device)
        pe = self.pe[:seq_len]
        if device is not None:
            pe = pe.to(device)
        return pe

    def get_pe(self, seq_len: int) -> torch.Tensor:
        """Alias for forward, convenient when used as a drop-in replacement."""
        return self.forward(seq_len)
