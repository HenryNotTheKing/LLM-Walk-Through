"""Learned absolute position embedding（GPT-2 原版做法）。

继承 ``nn.Embedding`` 让 state_dict 的 key 直接是 ``wpe.weight``，
便于与 HuggingFace 权重对齐与 checkpoint 互转。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Embedding):
    """与 token embedding 并列加在 input 上的可学习位置向量。

    ``forward(seq_len, device)`` 返回形状 ``(seq_len, n_embd)`` 的位置向量。
    """

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__(block_size, n_embd)
        self.block_size = block_size

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:  # type: ignore[override]
        if seq_len > self.block_size:
            raise ValueError(
                f"序列长度 {seq_len} 超过位置编码最大长度 {self.block_size}"
            )
        pos = torch.arange(seq_len, device=device or self.weight.device)
        return super().forward(pos)
