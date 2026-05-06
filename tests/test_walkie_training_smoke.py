"""Walkie 训练冒烟测试：直接调用 train 函数，跑 tiny 配置 2 步。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.utils.config import load_config


def test_walkie_pretrain_smoke(tmp_path, monkeypatch):
    # 训练脚本会写到 cfg.train.out_dir，这里用临时目录
    cfg = load_config(
        "configs/train/pretrain_walkie_tiny.yaml",
        overrides=[
            f"train.out_dir={tmp_path.as_posix()}",
            "train.total_steps=2",
            "train.eval_interval=1",
            "train.ckpt_interval=1",
            "train.eval_iters=1",
            "train.warmup_steps=1",
        ],
    )
    from train.walkie_pretrain import train as walkie_train

    walkie_train(cfg)
    # 至少 latest.pt 存在
    assert (tmp_path / "latest.pt").exists()


def test_walkie_pretrain_resume(tmp_path):
    cfg = load_config(
        "configs/train/pretrain_walkie_tiny.yaml",
        overrides=[
            f"train.out_dir={tmp_path.as_posix()}",
            "train.total_steps=2",
            "train.eval_interval=1",
            "train.ckpt_interval=1",
            "train.eval_iters=1",
            "train.warmup_steps=1",
        ],
    )
    from train.walkie_pretrain import train as walkie_train

    walkie_train(cfg)
    ckpt = tmp_path / "latest.pt"
    assert ckpt.exists()

    # 用同一 ckpt resume 再跑 2 步
    cfg2 = load_config(
        "configs/train/pretrain_walkie_tiny.yaml",
        overrides=[
            f"train.out_dir={tmp_path.as_posix()}",
            "train.total_steps=4",
            "train.eval_interval=1",
            "train.ckpt_interval=1",
            "train.eval_iters=1",
            "train.warmup_steps=1",
        ],
    )
    walkie_train(cfg2, resume=str(ckpt))
