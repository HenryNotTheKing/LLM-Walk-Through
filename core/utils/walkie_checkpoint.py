"""Walkie 训练 checkpoint：保存/加载/恢复路径解析。

保存内容：
    - model state_dict（自动剥离 DDP/torch.compile 包装）
    - 各优化器（AdamW、Muon）独立 state_dict
    - GradScaler 状态（若启用 AMP）
    - 学习率调度器状态（含两阶段连续推进信息）
    - 当前 step / stage / 最佳指标
    - 模型/训练 OmegaConf 序列化为 dict 的快照
    - 随机数状态（Python / NumPy / torch CPU + CUDA）
    - 版本号 ``WALKIE_CKPT_VERSION``，用于兼容性检测

落盘格式：
    runs/walkie/<run_name>/
        latest.pt
        best.pt
        step_000XXXX.pt   # 可选周期性快照
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch

WALKIE_CKPT_VERSION = 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """剥离 DDP / torch.compile 等包装，得到原始 ``nn.Module``。"""
    inner = model
    # DDP
    if hasattr(inner, "module") and isinstance(getattr(inner, "module"), torch.nn.Module):
        inner = inner.module
    # torch.compile (>=2.0) 把原模型放在 _orig_mod
    if hasattr(inner, "_orig_mod") and isinstance(getattr(inner, "_orig_mod"), torch.nn.Module):
        inner = inner._orig_mod
    return inner


def _collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if "torch_cuda_all" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
        except Exception:
            pass


def save_walkie_checkpoint(
    out_dir: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    scaler: torch.cuda.amp.GradScaler | None,
    schedule_state: dict[str, Any] | None,
    step: int,
    stage: str,
    best_metric: float | None,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    extra: dict[str, Any] | None = None,
    format: str = "latest",
    tag: str | int | None = None,
) -> Path:
    """保存 checkpoint 到 ``out_dir``，返回写入的文件路径。

    ``format`` 可取 ``"latest"`` / ``"best"`` / ``"step"``。当为 ``"step"`` 时
    使用 ``tag`` 作为步数后缀（如 ``step_00010000.pt``）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if format == "latest":
        path = out_dir / "latest.pt"
    elif format == "best":
        path = out_dir / "best.pt"
    elif format == "step":
        if tag is None:
            tag = step
        if isinstance(tag, int):
            path = out_dir / f"step_{tag:08d}.pt"
        else:
            path = out_dir / f"step_{tag}.pt"
    else:
        raise ValueError(f"未知 format={format}")

    payload: dict[str, Any] = {
        "version": WALKIE_CKPT_VERSION,
        "model_name": model_cfg.get("model_name", "Walkie"),
        "model": unwrap_model(model).state_dict(),
        "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        "scaler": scaler.state_dict() if scaler is not None else None,
        "schedule": schedule_state,
        "step": int(step),
        "stage": stage,
        "best_metric": best_metric,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "rng_state": _collect_rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_walkie_checkpoint(
    path: str | os.PathLike,
    *,
    map_location: str | torch.device | None = "cpu",
    strict_arch: bool = True,
    expected_model_cfg: dict[str, Any] | None = None,
    weights_only: bool = False,
) -> dict[str, Any]:
    """加载 checkpoint 字典。校验版本号与（可选）关键架构字段。

    ``weights_only`` 默认 ``False`` 以兼容含 RNG/optimizer state 的完整 ckpt；
    若仅需加载自己产生的、可信的权重，建议显式传 ``True`` 以规避反序列化风险。
    """
    payload = torch.load(path, map_location=map_location, weights_only=weights_only)
    version = payload.get("version", 0)
    if version > WALKIE_CKPT_VERSION:
        raise RuntimeError(
            f"checkpoint 版本 {version} 比当前代码 {WALKIE_CKPT_VERSION} 更高，"
            "请升级 Walkie 代码再加载。"
        )
    if strict_arch and expected_model_cfg is not None:
        loaded = payload.get("model_cfg", {})
        keys = (
            "vocab_size",
            "n_layer",
            "n_embd",
            "n_head",
            "n_head_kv",
            "head_dim",
            "d_ffn",
            "tie_weights",
        )
        for k in keys:
            if k in loaded and k in expected_model_cfg and loaded[k] != expected_model_cfg[k]:
                raise RuntimeError(
                    f"模型架构字段 {k} 不一致：ckpt={loaded[k]} vs cfg={expected_model_cfg[k]}"
                )
    return payload


def apply_walkie_checkpoint(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """把 ``payload`` 应用到现有模型/优化器/scaler 上，返回恢复后的元信息。"""
    inner = unwrap_model(model)
    missing, unexpected = inner.load_state_dict(payload["model"], strict=strict)
    if optimizers is not None:
        for name, opt in optimizers.items():
            sd = payload.get("optimizers", {}).get(name)
            if sd is not None:
                opt.load_state_dict(sd)
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng_state(payload.get("rng_state", {}))
    return {
        "step": payload.get("step", 0),
        "stage": payload.get("stage", "main"),
        "best_metric": payload.get("best_metric"),
        "schedule": payload.get("schedule"),
        "missing": list(missing) if isinstance(missing, Iterable) else missing,
        "unexpected": list(unexpected) if isinstance(unexpected, Iterable) else unexpected,
    }


def resolve_resume_path(path_or_dir: str | os.PathLike) -> Path:
    """``path_or_dir`` 可以是具体文件，也可以是目录；目录则按 latest > best > 最大 step 顺序解析。"""
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        latest = p / "latest.pt"
        if latest.exists():
            return latest
        best = p / "best.pt"
        if best.exists():
            return best
        candidates = sorted(p.glob("step_*.pt"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"无法在 {path_or_dir} 找到可恢复的 Walkie checkpoint")


def prune_step_checkpoints(out_dir: str | os.PathLike, keep: int) -> list[Path]:
    """只保留最近 ``keep`` 个 ``step_*.pt`` 快照，返回被删除的路径。"""
    if keep < 0:
        raise ValueError(f"keep 必须非负，得到 {keep}")
    out = Path(out_dir)
    candidates = sorted(out.glob("step_*.pt"))
    if keep == 0:
        stale = candidates
    else:
        stale = candidates[:-keep]
    removed: list[Path] = []
    for path in stale:
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed
