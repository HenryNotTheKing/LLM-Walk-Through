"""SwiGLU FFN（Walkie-Code-1B 专用）。

公式::

    SwiGLU(x) = (silu(W_gate x) * (W_up x)) W_down

参考 LLaMA / Mistral 的实现：``gate`` 与 ``up`` 两路并联，
``down`` 投影回 ``n_embd``。隐藏维度 ``d_ffn`` 显式给出，
通常取约 ``8/3 * n_embd`` 以匹配标准 4× GELU 的参数量。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """Walkie 默认的 FFN。

    Args:
        n_embd: 输入/输出隐藏维度。
        d_ffn: 中间隐藏维度。
        dropout: 输出 dropout 概率。
        bias: 是否带 bias，Walkie 默认为 ``False``。
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
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
