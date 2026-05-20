"""RMSNorm 的基础测试。"""

from __future__ import annotations

import torch

from core.norm.rmsnorm import RMSNorm


def test_rmsnorm_shape_and_dtype():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rmsnorm_preserves_low_precision_dtype_with_fp32_weight():
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16


def test_rmsnorm_unit_scale_when_weight_one():
    norm = RMSNorm(64)
    x = torch.randn(8, 64) * 5.0
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    # weight 全 1：归一化后每行 RMS 应当≈1
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_weight_scaling():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(3, 8)
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-3)
