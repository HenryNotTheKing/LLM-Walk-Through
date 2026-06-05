"""Derf — Dynamic erf（DyT 的增强版，使用误差函数）。

来源：Chen et al., *Stronger Normalization-Free Transformers* (2025,
arXiv:2512.10938)。

核心改进：在 DyT 的 ``tanh(α·x)`` 基础上增加可学习偏移 ``s``，并改用误差函数
``erf``。erf 在边界处的梯度衰减速率比 tanh 更温和，且 learnable shift 提供了
额外的自由度，使其在多个领域（视觉、生成、DNA、语音）的泛化性能优于 DyT 和
LayerNorm。

公式：
    Derf(x) = erf(α · x + s) · γ

训练建议：
    - 默认 α_init = 0.3（比 DyT 更小），可降低深层网络的饱和风险。
    - 若使用 Muon 优化器，建议保持 α ≤ 0.3 以维持近线性区。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Derf(nn.Module):
    """Dynamic erf 归一化替代层。

    Args:
        normalized_shape: 特征维度，用于生成逐通道 γ。
        alpha_init: α 的初始值，默认 0.3。
        shift_init: 偏移 s 的初始值，默认 0.0。
        gamma_init: γ 的初始值，默认 1.0。
        use_gamma: 是否使用逐通道 γ。
    """

    def __init__(
        self,
        normalized_shape: int,
        alpha_init: float = 0.3,
        shift_init: float = 0.0,
        gamma_init: float = 1.0,
        use_gamma: bool = True,
    ) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(shift_init, dtype=torch.float32))
        if use_gamma:
            self.gamma = nn.Parameter(torch.full((normalized_shape,), gamma_init, dtype=torch.float32))
        else:
            self.register_parameter("gamma", None)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        alpha = self.alpha.to(x_fp32.dtype)
        shift = self.shift.to(x_fp32.dtype)
        out = torch.erf(alpha * x_fp32 + shift)
        if self.gamma is not None:
            out = out * self.gamma.to(x_fp32.dtype)
        return out.to(orig_dtype)
