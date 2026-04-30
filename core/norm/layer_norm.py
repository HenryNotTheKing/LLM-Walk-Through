"""LayerNorm。

GPT-2 沿用标准 LayerNorm（带 affine 参数与 bias）。这里直接基于 ``torch.nn.LayerNorm``
做一层薄包装，统一与项目其他归一化模块的接口（后续 RMSNorm 等会放在同目录）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """带可选 bias 的 LayerNorm。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )
