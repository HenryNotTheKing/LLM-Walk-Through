"""Walkie 模型主体的基础测试。"""

from __future__ import annotations

import torch

from core.model.walkie import WalkieConfig, WalkieForCausalLM


def make_tiny() -> WalkieForCausalLM:
    cfg = WalkieConfig(
        vocab_size=128, block_size=64,
        n_embd=32, n_layer=2, n_head=4, n_head_kv=2, head_dim=8, d_ffn=64,
        dropout=0.0, bias=False, tie_weights=True, attn_impl="sdpa",
    )
    return WalkieForCausalLM(cfg)


def test_walkie_forward_and_loss():
    model = make_tiny()
    B, T = 2, 8
    idx = torch.randint(0, 128, (B, T))
    tgt = torch.randint(0, 128, (B, T))
    logits, loss = model(idx, tgt)
    assert logits.shape == (B, T, 128)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_walkie_inference_returns_last_step():
    model = make_tiny()
    logits, loss = model(torch.randint(0, 128, (2, 5)))
    assert loss is None
    assert logits.shape == (2, 1, 128)


def test_walkie_backward():
    model = make_tiny()
    idx = torch.randint(0, 128, (2, 6))
    tgt = torch.randint(0, 128, (2, 6))
    _, loss = model(idx, tgt)
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None


def test_walkie_tied_weights():
    model = make_tiny()
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()


def test_walkie_generate_shape():
    model = make_tiny()
    out = model.generate(torch.randint(0, 128, (1, 3)), max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 8)


def test_walkie_block_size_overflow_raises():
    import pytest
    model = make_tiny()
    too_long = torch.zeros(1, model.cfg.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        model(too_long)


def test_walkie_1b_param_count_under_one_billion():
    """通过 num_parameters 解析 1B 配置不超过 1B（不实际分配权重）。"""
    cfg = WalkieConfig()  # 默认即 Walkie-Code-1B
    # 直接做参数量估算，避免实例化 ~964M 模型
    V, D, L = cfg.vocab_size, cfg.n_embd, cfg.n_layer
    H, Hkv, Hd, F = cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.d_ffn
    embed = V * D  # tied，lm_head 不再算
    # 每层：q(D, H*Hd), k(D, Hkv*Hd), v(D, Hkv*Hd), o(H*Hd, D), gate/up/down(D,F + D,F + F,D)
    per_layer = (
        D * H * Hd
        + 2 * D * Hkv * Hd
        + H * Hd * D
        + 3 * D * F
        + 2 * D  # 两个 RMSNorm
        + (2 * Hd if cfg.qk_norm else 0)  # q_norm/k_norm
    )
    total = embed + L * per_layer + D  # final norm
    assert total < 1_000_000_000, f"Walkie-Code-1B 估算参数 {total/1e6:.1f}M 超过 1B"
