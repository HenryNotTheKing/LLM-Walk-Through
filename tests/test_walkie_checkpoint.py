"""Walkie checkpoint 保存/加载/恢复测试。"""

from __future__ import annotations

import torch

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.walkie_checkpoint import (
    WALKIE_CKPT_VERSION,
    apply_walkie_checkpoint,
    load_walkie_checkpoint,
    resolve_resume_path,
    save_walkie_checkpoint,
    unwrap_model,
)
from core.utils.walkie_optim import build_walkie_optimizers
from core.utils.walkie_schedule import WalkieWSDSchedule


def _make():
    cfg = WalkieConfig(
        vocab_size=64, block_size=32, n_embd=16, n_layer=2,
        n_head=2, n_head_kv=1, head_dim=8, d_ffn=32,
        dropout=0.0, bias=False, tie_weights=True,
    )
    model = WalkieForCausalLM(cfg)
    opts = build_walkie_optimizers(model, adamw_lr=1e-3, muon_lr=1e-2)
    sched = WalkieWSDSchedule.from_config(
        total_steps=10, warmup_steps=2, anneal_start_ratio=0.8, decay_shape="sqrt",
        tracks={"adamw": {"peak_lr": 1e-3, "final_lr": 1e-4},
                "muon": {"peak_lr": 1e-2, "final_lr": 1e-3}},
    )
    return cfg, model, opts, sched


def test_unwrap_model_strips_compile_orig_mod():
    _, model, _, _ = _make()
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m
    wrapped = Wrap(model)
    assert unwrap_model(wrapped) is model


def test_save_and_load_round_trip(tmp_path):
    cfg, model, opts, sched = _make()
    # 跑一步让优化器/模型权重发生变化
    idx = torch.randint(0, 64, (1, 4))
    tgt = torch.randint(0, 64, (1, 4))
    _, loss = model(idx, tgt)
    loss.backward()
    for opt in opts.values():
        opt.step()
    sched.step_to(3)

    path = save_walkie_checkpoint(
        tmp_path,
        model=model, optimizers=opts, scaler=None,
        schedule_state=sched.state_dict(),
        step=3, stage="main", best_metric=1.23,
        model_cfg=cfg.to_dict(), train_cfg={"foo": "bar"},
        format="latest",
    )
    assert path.exists()
    assert path.name == "latest.pt"

    # 重新构造模型 / 优化器 / 调度器
    cfg2, model2, opts2, sched2 = _make()
    payload = load_walkie_checkpoint(path, expected_model_cfg=cfg2.to_dict())
    assert payload["version"] == WALKIE_CKPT_VERSION

    info = apply_walkie_checkpoint(payload, model=model2, optimizers=opts2, scaler=None)
    sched2.load_state_dict(payload["schedule"])

    assert info["step"] == 3
    assert info["best_metric"] == 1.23
    assert sched2.step == 3
    # 权重应一致
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.equal(v1, v2), k1


def test_resolve_resume_path_prefers_latest(tmp_path):
    cfg, model, opts, sched = _make()
    save_walkie_checkpoint(
        tmp_path, model=model, optimizers=opts, scaler=None,
        schedule_state=sched.state_dict(), step=1, stage="main", best_metric=None,
        model_cfg=cfg.to_dict(), train_cfg={}, format="step", tag=1,
    )
    save_walkie_checkpoint(
        tmp_path, model=model, optimizers=opts, scaler=None,
        schedule_state=sched.state_dict(), step=2, stage="main", best_metric=None,
        model_cfg=cfg.to_dict(), train_cfg={}, format="latest",
    )
    p = resolve_resume_path(tmp_path)
    assert p.name == "latest.pt"


def test_arch_mismatch_raises(tmp_path):
    cfg, model, opts, sched = _make()
    save_walkie_checkpoint(
        tmp_path, model=model, optimizers=opts, scaler=None,
        schedule_state=sched.state_dict(), step=0, stage="main", best_metric=None,
        model_cfg=cfg.to_dict(), train_cfg={}, format="latest",
    )
    bad_cfg = cfg.to_dict() | {"n_layer": 99}
    import pytest
    with pytest.raises(RuntimeError):
        load_walkie_checkpoint(tmp_path / "latest.pt", expected_model_cfg=bad_cfg)
