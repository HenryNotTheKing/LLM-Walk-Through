"""Walkie 注意力（GQA + QK-Norm + RoPE）的基础测试。"""

from __future__ import annotations

import torch

from core.attention.walkie_attention import WalkieCausalSelfAttention, repeat_kv


def test_repeat_kv_shape():
    x = torch.randn(2, 3, 5, 8)  # (B, H_kv, T, D)
    y = repeat_kv(x, 4)
    assert y.shape == (2, 12, 5, 8)
    # 每 4 个相邻头应一致
    assert torch.allclose(y[:, 0], y[:, 1])
    assert torch.allclose(y[:, 4], y[:, 5])


def test_repeat_kv_noop_when_one():
    x = torch.randn(1, 2, 3, 4)
    assert torch.equal(repeat_kv(x, 1), x)


def test_walkie_attn_forward_shape():
    attn = WalkieCausalSelfAttention(
        n_embd=64, n_head=8, n_head_kv=2, head_dim=8, max_seq_len=32,
        dropout=0.0, attn_impl="sdpa",
    )
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == x.shape


def test_walkie_attn_eager_matches_sdpa_shape():
    torch.manual_seed(0)
    args = dict(
        n_embd=32, n_head=4, n_head_kv=2, head_dim=8, max_seq_len=16,
        dropout=0.0, qk_norm=False,
    )
    attn_sdpa = WalkieCausalSelfAttention(attn_impl="sdpa", **args)
    attn_eager = WalkieCausalSelfAttention(attn_impl="eager", **args)
    # 同步权重
    attn_eager.load_state_dict(attn_sdpa.state_dict())
    x = torch.randn(2, 8, 32)
    y_sdpa = attn_sdpa(x)
    y_eager = attn_eager(x)
    assert torch.allclose(y_sdpa, y_eager, atol=1e-4)


def test_walkie_attn_causal():
    """改变靠后的 token 不应影响靠前的输出（因果性）。"""
    torch.manual_seed(1)
    attn = WalkieCausalSelfAttention(
        n_embd=16, n_head=2, n_head_kv=1, head_dim=8, max_seq_len=8,
        dropout=0.0, attn_impl="sdpa",
    )
    attn.eval()
    x1 = torch.randn(1, 6, 16)
    x2 = x1.clone()
    x2[:, -1, :] += 10.0  # 改最后一位
    y1 = attn(x1)
    y2 = attn(x2)
    # 前 5 位应保持一致
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-4)
