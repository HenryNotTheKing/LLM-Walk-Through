"""GEGLU FFN (PaLM 风格)。

Shazeer (2020) *GLU Variants Improve Transformer* 中提出 GLU 门控机制，
PaLM (Chowdhery et al., 2022) 采用 **GEGLU** 变体——用 GELU 替代 SiLU 作为门控激活：

    GEGLU(x) = (gelu(W_gate x) * (W_up x)) W_down

与 SwiGLU 结构完全一致，仅将门控激活函数由 SiLU 替换为 GELU。
实验表明 GEGLU 在某些任务上的训练稳定性优于 SwiGLU，且 GELU 在原点的
平滑性使其对输入分布偏移更不敏感。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLUMLP(nn.Module):
    """PaLM 采用的 GEGLU 前馈网络。

    Args:
        n_embd: 输入/输出隐藏维度。
        d_ffn: 中间隐藏维度。PaLM 使用约 ``2 * n_embd``（注意 GEGLU
            已天然包含 up 投影，参数量与 4× GELU MLP 等价）。
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
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))
