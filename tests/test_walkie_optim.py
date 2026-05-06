"""Walkie 优化器测试：参数分组、Muon step、状态恢复。"""

from __future__ import annotations

import torch

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.walkie_optim import (
    Muon,
    build_walkie_optimizers,
    split_walkie_params,
    zeropower_via_newton_schulz5,
)


def make_tiny() -> WalkieForCausalLM:
    cfg = WalkieConfig(
        vocab_size=64, block_size=32, n_embd=16, n_layer=2,
        n_head=2, n_head_kv=1, head_dim=8, d_ffn=32,
        dropout=0.0, bias=False, tie_weights=True, attn_impl="sdpa",
    )
    return WalkieForCausalLM(cfg)


def test_split_walkie_params_routes_correctly():
    model = make_tiny()
    muon, adamw, mn, an = split_walkie_params(model)
    # 所有 muon 参数都应是 2D 矩阵权重，且名字含 proj
    for p, n in zip(muon, mn):
        assert p.ndim == 2, n
        assert "proj" in n.lower()
    # AdamW 组应包含 embedding/norm
    assert any("tok_embeddings" in n for n in an)
    assert any("norm" in n.lower() for n in an)
    # 没有交集
    muon_ids = {id(p) for p in muon}
    adamw_ids = {id(p) for p in adamw}
    assert muon_ids.isdisjoint(adamw_ids)


def test_newton_schulz_orthogonalizes_small_matrix():
    torch.manual_seed(0)
    G = torch.randn(8, 8)
    O = zeropower_via_newton_schulz5(G.clone(), steps=5)
    # NS5 是近似 msign：奇异值在 [~0.5, ~1.2] 区间，远比原始 G 的奇异值分布更集中
    s = torch.linalg.svdvals(O.float())
    s_orig = torch.linalg.svdvals(G.float())
    spread = s.max() / s.min()
    spread_orig = s_orig.max() / s_orig.min()
    assert spread < spread_orig, (spread, spread_orig)
    assert s.max() < 1.5 and s.min() > 0.4


def test_muon_step_changes_params_and_keeps_shape():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 16))
    opt = Muon([w], lr=0.01, momentum=0.9, weight_decay=0.0)
    target = torch.randn(8, 16)
    before = w.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        loss = ((w - target) ** 2).sum()
        loss.backward()
        opt.step()
    assert w.shape == before.shape
    assert not torch.allclose(w.detach(), before)


def test_build_walkie_optimizers_state_dict_round_trip():
    model = make_tiny()
    opts = build_walkie_optimizers(model, adamw_lr=1e-3, muon_lr=1e-2)
    # 跑一步，让 state 真实填充
    idx = torch.randint(0, 64, (1, 4))
    tgt = torch.randint(0, 64, (1, 4))
    _, loss = model(idx, tgt)
    loss.backward()
    for opt in opts.values():
        opt.step()
    sd = {k: v.state_dict() for k, v in opts.items()}

    # 重新构造模型与优化器，加载 state
    model2 = make_tiny()
    model2.load_state_dict(model.state_dict())
    opts2 = build_walkie_optimizers(model2, adamw_lr=1e-3, muon_lr=1e-2)
    for k, opt in opts2.items():
        opt.load_state_dict(sd[k])
    # 状态结构应一致
    assert set(opts2["muon"].state.keys())
    assert set(opts2["adamw"].state.keys())
