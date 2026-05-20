"""RMSNorm（Walkie-Code-1B 专用）。

参考 LLaMA / T5 的实现：
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

与 LayerNorm 相比少了减均值与可选 bias，参数减半、计算更快，
对训练稳定性的影响在 1B 量级模型上已被广泛验证。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """无中心化、仅有可学习缩放向量的归一化。

    Args:
        normalized_shape: 最后一维的特征维度。
        eps: 数值稳定项。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 fp32 计算 RMS 提高数值稳定性，再回到原 dtype
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_fp32 * rms).to(orig_dtype)
        return out * self.weight.to(orig_dtype)
