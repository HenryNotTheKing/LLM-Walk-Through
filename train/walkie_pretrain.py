"""Walkie-Code-1B 预训练入口（多卡 DDP + 两阶段连续 WSD + 完善 checkpoint）。

特性：
    - **不修改** ``train/pretrain.py``：本脚本独立完整实现。
    - 单卡：``python -m train.walkie_pretrain --config configs/train/pretrain_walkie_tiny.yaml``
    - 多卡：``torchrun --nproc_per_node=N -m train.walkie_pretrain --config ...``
    - 两阶段（main / anneal）共用 **同一个 step 计数器** 与 **同一组优化器/调度器**，
      在 ``anneal_start_ratio`` 处切换数据流，学习率连续无跳变。
    - AdamW + Muon 双优化器；checkpoint 同步保存两者状态、scaler、调度器、RNG。

数据流水线占位：
    本仓库的数据处理由用户后续负责，这里仅约定通过 ``data.stages.<main|anneal>.bin``
    指向 ``np.memmap`` 文件（dtype 由 ``data.stages.<x>.dtype`` 给出）。如果路径不存在
    则会用一段内存里随机生成的 token 流兜底，方便冒烟测试。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.config import load_config
from core.utils.device import amp_enabled, select_device, select_dtype
from core.utils.distributed import cleanup_distributed, setup_distributed
from core.utils.walkie_checkpoint import (
    apply_walkie_checkpoint,
    load_walkie_checkpoint,
    resolve_resume_path,
    save_walkie_checkpoint,
    unwrap_model,
)
from core.utils.walkie_optim import build_walkie_optimizers
from core.utils.walkie_schedule import WalkieWSDSchedule, apply_lrs_to_optimizers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="checkpoint 文件或目录；目录则解析 latest.pt > best.pt > step_*.pt",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        type=str,
        help="只加载模型权重作为初始化（不恢复优化器/调度器）",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist 覆盖")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def _load_stage_data(
    stage_cfg, vocab_size: int, block_size: int
) -> np.ndarray:
    """加载一个阶段的 ``np.memmap``；不存在则生成随机 token 兜底（仅 smoke 用）。"""
    bin_path = stage_cfg.get("bin", None) if stage_cfg is not None else None
    dtype_name = (stage_cfg.get("dtype", "uint16") if stage_cfg is not None else "uint16")
    dtype = np.dtype(dtype_name)
    if bin_path and Path(bin_path).exists():
        return np.memmap(bin_path, dtype=dtype, mode="r")
    # 兜底：内存里生成几百万 token 的随机数据，保证脚本能 smoke
    rng = np.random.default_rng(0)
    n = max(block_size * 64, 4096)
    return rng.integers(0, vocab_size, size=n, dtype=np.int64).astype(dtype)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(np.asarray(data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def train(cfg, resume: str | None = None, init_from: str | None = None) -> None:
    dist_info = setup_distributed(cfg.distributed.backend)

    train_cfg = cfg.train
    model_cfg_dict = dict(cfg.model)

    torch.manual_seed(int(train_cfg.seed) + dist_info.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(train_cfg.seed) + dist_info.rank)

    device = select_device(train_cfg.device)
    dtype = select_dtype(train_cfg.dtype, device)
    use_amp = amp_enabled(device, dtype, bool(train_cfg.amp))

    if dist_info.is_main:
        print(
            f"[walkie/setup] device={device} dtype={dtype} amp={use_amp} "
            f"world_size={dist_info.world_size}"
        )

    # ----- 模型 -----
    model_cfg_dict.setdefault("block_size", int(train_cfg.block_size))
    model_cfg = WalkieConfig.from_dict(model_cfg_dict)
    model = WalkieForCausalLM(model_cfg).to(device)
    if bool(getattr(train_cfg, "compile", False)):
        model = torch.compile(model)  # type: ignore[assignment]

    if dist_info.enabled:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
        )

    raw_model = unwrap_model(model)
    if dist_info.is_main:
        n_params = raw_model.num_parameters()
        print(f"[walkie/setup] model={model_cfg.model_name} params={n_params/1e6:.2f}M")

    # ----- 优化器 + 调度器 -----
    optimizers = build_walkie_optimizers(
        raw_model,
        adamw_lr=float(train_cfg.adamw.peak_lr),
        muon_lr=float(train_cfg.muon.peak_lr),
        adamw_betas=(float(train_cfg.adamw.beta1), float(train_cfg.adamw.beta2)),
        adamw_eps=float(train_cfg.adamw.eps),
        adamw_weight_decay=float(train_cfg.adamw.weight_decay),
        muon_momentum=float(train_cfg.muon.momentum),
        muon_nesterov=bool(train_cfg.muon.nesterov),
        muon_weight_decay=float(train_cfg.muon.weight_decay),
        muon_ns_steps=int(train_cfg.muon.ns_steps),
    )

    schedule = WalkieWSDSchedule.from_config(
        total_steps=int(train_cfg.total_steps),
        warmup_steps=int(train_cfg.warmup_steps),
        anneal_start_ratio=float(train_cfg.anneal_start_ratio),
        decay_shape=str(train_cfg.decay_shape),
        tracks={
            "adamw": {
                "peak_lr": float(train_cfg.adamw.peak_lr),
                "final_lr": float(train_cfg.adamw.final_lr),
            },
            "muon": {
                "peak_lr": float(train_cfg.muon.peak_lr),
                "final_lr": float(train_cfg.muon.final_lr),
            },
        },
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    # ----- 恢复 -----
    start_step = 0
    if init_from is not None:
        if dist_info.is_main:
            print(f"[walkie/init-from] {init_from}")
        payload = load_walkie_checkpoint(
            resolve_resume_path(init_from),
            map_location="cpu",
            strict_arch=True,
            expected_model_cfg=model_cfg.to_dict(),
        )
        apply_walkie_checkpoint(payload, model=model, optimizers=None, scaler=None, restore_rng=False)
    if resume is not None:
        resume_path = resolve_resume_path(resume)
        if dist_info.is_main:
            print(f"[walkie/resume] {resume_path}")
        payload = load_walkie_checkpoint(
            resume_path,
            map_location="cpu",
            strict_arch=True,
            expected_model_cfg=model_cfg.to_dict(),
        )
        info = apply_walkie_checkpoint(
            payload, model=model, optimizers=optimizers, scaler=scaler, restore_rng=True
        )
        if payload.get("schedule") is not None:
            schedule.load_state_dict(payload["schedule"])
        start_step = int(info.get("step", 0))

    # ----- 数据：两阶段 -----
    stages_cfg = cfg.data.stages
    main_data = _load_stage_data(stages_cfg.main, model_cfg.vocab_size, int(train_cfg.block_size))
    anneal_data = _load_stage_data(
        stages_cfg.anneal, model_cfg.vocab_size, int(train_cfg.block_size)
    )

    # ----- 训练循环 -----
    out_dir = Path(train_cfg.out_dir)
    if dist_info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    total_steps = int(train_cfg.total_steps)
    log_interval = int(train_cfg.log_interval)
    eval_interval = int(train_cfg.eval_interval)
    ckpt_interval = int(getattr(train_cfg, "ckpt_interval", 0)) or eval_interval
    grad_accum = int(train_cfg.grad_accum_steps)

    best_metric: float | None = None
    t0 = time.time()
    last_stage = schedule.current_stage(start_step)
    if dist_info.is_main:
        print(f"[walkie/train] start_step={start_step} stage={last_stage} total_steps={total_steps}")

    for step in range(start_step, total_steps + 1):
        # 1) 学习率
        lrs = schedule.step_to(step)
        apply_lrs_to_optimizers(optimizers, lrs)

        # 2) 阶段切换日志
        stage = schedule.current_stage(step)
        if stage != last_stage and dist_info.is_main:
            print(
                f"[walkie/stage] step={step} switch {last_stage} -> {stage} "
                f"(lrs={lrs})"
            )
            last_stage = stage
        cur_data = anneal_data if stage == "anneal" else main_data

        # 3) eval / ckpt（用主进程一并完成）
        if step % eval_interval == 0 and dist_info.is_main and step > start_step:
            raw_model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(int(train_cfg.eval_iters)):
                    x, y = get_batch(
                        cur_data, int(train_cfg.block_size), int(train_cfg.batch_size), device
                    )
                    if use_amp:
                        with torch.autocast(device_type=device.type, dtype=dtype):
                            _, loss = raw_model(x, y)
                    else:
                        _, loss = raw_model(x, y)
                    losses.append(loss.item())
            val = float(np.mean(losses))
            print(f"[walkie/eval] step={step} stage={stage} val_loss={val:.4f} lrs={lrs}")
            raw_model.train()

            if best_metric is None or val < best_metric:
                best_metric = val
                save_walkie_checkpoint(
                    out_dir,
                    model=model,
                    optimizers=optimizers,
                    scaler=scaler,
                    schedule_state=schedule.state_dict(),
                    step=step,
                    stage=stage,
                    best_metric=best_metric,
                    model_cfg=model_cfg.to_dict(),
                    train_cfg=dict(train_cfg),
                    format="best",
                )

        if step % ckpt_interval == 0 and dist_info.is_main and step > start_step:
            save_walkie_checkpoint(
                out_dir,
                model=model,
                optimizers=optimizers,
                scaler=scaler,
                schedule_state=schedule.state_dict(),
                step=step,
                stage=stage,
                best_metric=best_metric,
                model_cfg=model_cfg.to_dict(),
                train_cfg=dict(train_cfg),
                format="latest",
            )

        if step == total_steps:
            break

        # 4) 一个 step（梯度累积）
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x, y = get_batch(
                cur_data, int(train_cfg.block_size), int(train_cfg.batch_size), device
            )
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()

        if float(train_cfg.grad_clip) > 0:
            if use_amp and dtype == torch.float16:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), float(train_cfg.grad_clip)
            )

        if use_amp and dtype == torch.float16:
            for opt in optimizers.values():
                scaler.step(opt)
            scaler.update()
        else:
            for opt in optimizers.values():
                opt.step()

        if step % log_interval == 0 and dist_info.is_main:
            dt = time.time() - t0
            print(
                f"[walkie/step] step={step} stage={stage} "
                f"loss={loss.item() * grad_accum:.4f} "
                f"lr_adamw={lrs['adamw']:.2e} lr_muon={lrs['muon']:.2e} "
                f"dt={dt:.1f}s"
            )

    # 收尾：写一次 latest，确保最后一步状态落盘
    if dist_info.is_main:
        save_walkie_checkpoint(
            out_dir,
            model=model,
            optimizers=optimizers,
            scaler=scaler,
            schedule_state=schedule.state_dict(),
            step=total_steps,
            stage=schedule.current_stage(total_steps),
            best_metric=best_metric,
            model_cfg=model_cfg.to_dict(),
            train_cfg=dict(train_cfg),
            format="latest",
        )
        print(f"[walkie/done] saved to {out_dir}")

    cleanup_distributed()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    train(cfg, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
