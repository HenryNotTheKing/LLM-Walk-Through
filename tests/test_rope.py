"""RoPE 的基础测试。"""

from __future__ import annotations

import torch

from core.position.rope import RotaryPositionalEmbedding, apply_rope, compute_rope_freqs


def test_compute_rope_freqs_length_and_decreasing():
    freqs = compute_rope_freqs(64, base=1e6)
    assert freqs.shape == (32,)
    # 频率应单调递减
    assert (freqs[:-1] >= freqs[1:]).all()


def test_rope_module_cache_grows_when_needed():
    rope = RotaryPositionalEmbedding(head_dim=16, max_seq_len=8, base=10000.0)
    cos, sin = rope(8)
    assert cos.shape == (8, 16) and sin.shape == (8, 16)
    cos2, sin2 = rope(32)
    assert cos2.shape == (32, 16)


def test_apply_rope_preserves_shape_and_norm_per_pair():
    rope = RotaryPositionalEmbedding(head_dim=8, max_seq_len=16, base=10000.0)
    cos, sin = rope(4, dtype=torch.float32)
    x = torch.randn(2, 3, 4, 8)  # (B, H, T, D)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # 旋转保范数：x 的每个 (前半, 后半) 配对二范数应与 y 相同
    half = 4
    nx = x[..., :half] ** 2 + x[..., half:] ** 2
    ny = y[..., :half] ** 2 + y[..., half:] ** 2
    assert torch.allclose(nx, ny, atol=1e-5)


def test_apply_rope_identity_at_position_zero():
    rope = RotaryPositionalEmbedding(head_dim=8, max_seq_len=4, base=10000.0)
    cos, sin = rope(1, dtype=torch.float32)
    x = torch.randn(1, 1, 1, 8)
    y = apply_rope(x, cos, sin)
    # 位置 0 时 sin=0 cos=1，应近似恒等
    assert torch.allclose(x, y, atol=1e-5)
