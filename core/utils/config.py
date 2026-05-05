"""配置加载工具。

约定：
    - 模型配置位于 ``configs/model/*.yaml``
    - 训练配置位于 ``configs/train/*.yaml``，可通过 ``defaults`` 字段引用模型配置。
    - 命令行可通过 ``key=value`` 形式覆盖任意字段（OmegaConf dotlist）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        # 相对仓库根目录
        p = (REPO_ROOT / p).resolve()
    return p


def load_yaml(path: str | Path) -> DictConfig:
    """加载单个 YAML 配置文件。"""
    cfg = OmegaConf.load(_resolve_path(path))
    assert isinstance(cfg, DictConfig)
    return cfg


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """加载主配置，并按需合并 ``defaults`` 字段。

    支持的 ``defaults`` 形式::

        defaults:
          model: configs/model/gpt2_tiny.yaml

    被引用的子配置会写入相应字段（如上例中的 ``model``）。
    命令行 dotlist 覆盖（例如 ``train.batch_size=8``）会在最后合并。
    """
    cfg = load_yaml(path)

    def _apply_defaults(cfg: DictConfig) -> DictConfig:
        if "defaults" in cfg:
            defaults = cfg.pop("defaults")
            merged = OmegaConf.create({})
            for key, sub_path in defaults.items():
                sub = load_yaml(sub_path)
                merged[key] = sub
            cfg = OmegaConf.merge(cfg, merged)
        return cfg

    cfg = _apply_defaults(cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg = _apply_defaults(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg
