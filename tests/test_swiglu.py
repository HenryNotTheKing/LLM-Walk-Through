"""SwiGLU MLP 的基础测试。"""

from __future__ import annotations

import torch

from core.ffn.swiglu import SwiGLUMLP


def test_swiglu_forward_shape():
    mlp = SwiGLUMLP(n_embd=16, d_ffn=32)
    x = torch.randn(2, 4, 16)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_swiglu_no_bias_by_default():
    mlp = SwiGLUMLP(n_embd=8, d_ffn=16)
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


def test_swiglu_param_count():
    n_embd, d_ffn = 16, 32
    mlp = SwiGLUMLP(n_embd, d_ffn)
    expected = 3 * n_embd * d_ffn  # 三个 Linear 各 n_embd*d_ffn
    got = sum(p.numel() for p in mlp.parameters())
    assert got == expected
