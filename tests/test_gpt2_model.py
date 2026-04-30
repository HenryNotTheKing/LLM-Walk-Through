"""GPT-2 模型基本前向 / loss / generate / 边界条件。"""

from __future__ import annotations

import pytest
import torch

from core.model import GPT2Config, GPT2LMHeadModel


def make_tiny() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=64, n_layer=2, n_head=2, n_embd=32,
        block_size=16, dropout=0.0, bias=True, tie_weights=True, attn_impl="sdpa",
    )
    return GPT2LMHeadModel(cfg)


def test_forward_shapes_and_loss():
    model = make_tiny()
    B, T = 3, 8
    idx = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, 64)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_inference_only_returns_last_step():
    model = make_tiny()
    idx = torch.randint(0, 64, (2, 5))
    logits, loss = model(idx)
    assert loss is None
    assert logits.shape == (2, 1, 64)


def test_loss_backward():
    model = make_tiny()
    idx = torch.randint(0, 64, (2, 6))
    tgt = torch.randint(0, 64, (2, 6))
    _, loss = model(idx, tgt)
    loss.backward()
    # 至少 lm_head/wte（共享）应该有梯度
    assert model.wte.weight.grad is not None


def test_tied_weights():
    model = make_tiny()
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()


def test_block_size_overflow_raises():
    model = make_tiny()
    too_long = torch.zeros(1, model.cfg.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        model(too_long)


def test_generate_shapes_greedy_and_sampling():
    torch.manual_seed(0)
    model = make_tiny()
    idx = torch.zeros(2, 1, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=0.0)  # greedy
    assert out.shape == (2, 6)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10, top_p=0.9)
    assert out.shape == (2, 6)


def test_attn_impl_eager_matches_sdpa():
    """sdpa 与 eager 在 dropout=0 时输出应一致（数值容差较宽，因为算子路径不同）。"""
    torch.manual_seed(0)
    cfg_sdpa = GPT2Config(vocab_size=64, n_layer=2, n_head=2, n_embd=32,
                          block_size=16, dropout=0.0, attn_impl="sdpa")
    cfg_eager = GPT2Config(vocab_size=64, n_layer=2, n_head=2, n_embd=32,
                           block_size=16, dropout=0.0, attn_impl="eager")
    m1 = GPT2LMHeadModel(cfg_sdpa)
    m2 = GPT2LMHeadModel(cfg_eager)
    m2.load_state_dict(m1.state_dict())
    m1.eval(); m2.eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        l1, _ = m1(x, x)
        l2, _ = m2(x, x)
    assert torch.allclose(l1, l2, atol=1e-4, rtol=1e-4)
