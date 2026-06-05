"""ReGLU FFN。

Shazeer (2020) *GLU Variants Improve Transformer* 中 ReGLU 变体：
用 ReLU 替代 SiLU 作为门控激活：

    ReGLU(x) = (relu(W_gate x) * (W_up x)) W_down

ReGLU 是 GLU 家族中最"硬"的门控——负值区域完全截断为 0，
使 gate 起到更明确的开关作用。参数量和计算开销与 SwiGLU / GEGLU 相同。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReGLUMLP(nn.Module):
    """ReGLU 前馈网络。

    Args:
        n_embd: 输入/输出隐藏维度。
        d_ffn: 中间隐藏维度。
        dropout: 输出 dropout 概率。
        bias: 是否带 bias。
    """

    def __init__(
        self,
        n_embd: int,
        d_ffn: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, d_ffn, bias=bias)
        self.up_proj = nn.Linear(n_embd, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
