"""配置加载与 OmegaConf 覆盖。"""

from __future__ import annotations

from core.utils.config import load_config


def test_load_pretrain_tiny_with_overrides():
    cfg = load_config(
        "configs/train/pretrain_tiny.yaml",
        overrides=["train.batch_size=4", "train.max_steps=5"],
    )
    # 子配置应被合并到 model 字段
    assert cfg.model.name == "gpt2_tiny"
    assert cfg.model.n_layer == 2
    # 覆盖生效
    assert int(cfg.train.batch_size) == 4
    assert int(cfg.train.max_steps) == 5
    # distributed 字段存在
    assert cfg.distributed.backend == "none"
