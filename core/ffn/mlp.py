"""GPT-2 GELU MLP。

结构: ``Linear -> GELU(approx='tanh') -> Linear -> dropout``，隐藏维度为 ``4 * n_embd``。
GPT-2 原始实现使用 ``tanh`` 近似的 GELU；为了与 HF 权重对齐，这里也使用同样的近似。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeluMLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
