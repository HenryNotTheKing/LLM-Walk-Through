"""最小预训练脚本。

用法：
    单卡:    ``python -m train.pretrain --config configs/train/pretrain_tiny.yaml``
    DDP:     ``torchrun --nproc_per_node=2 -m train.pretrain \
                  --config configs/train/pretrain_tiny.yaml distributed.backend=ddp``
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from core.model import GPT2Config, GPT2LMHeadModel
from core.utils.config import load_config
from core.utils.device import amp_enabled, select_device, select_dtype
from core.utils.distributed import cleanup_distributed, setup_distributed
from data.tiny_shakespeare import prepare as prepare_tiny_shakespeare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist 覆盖，如 train.batch_size=8")
    return parser.parse_args()


def get_lr(step: int, cfg) -> float:
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    # cosine with warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    dist_info = setup_distributed(cfg.distributed.backend)

    train_cfg = cfg.train
    model_cfg_dict = cfg.model

    torch.manual_seed(train_cfg.seed + dist_info.rank)

    device = select_device(train_cfg.device)
    dtype = select_dtype(train_cfg.dtype, device)
    use_amp = amp_enabled(device, dtype, bool(train_cfg.amp))

    if dist_info.is_main:
        print(f"[setup] device={device} dtype={dtype} amp={use_amp} world_size={dist_info.world_size}")

    # ----- 数据 -----
    data_info = prepare_tiny_shakespeare(
        cache_dir=cfg.data.cache_dir,
        tokenizer_kind=cfg.data.tokenizer.kind,
        vocab_size=int(cfg.data.tokenizer.vocab_size),
        train_on_first_n_chars=int(cfg.data.tokenizer.train_on_first_n_chars),
    )
    train_data = np.memmap(data_info["train_bin"], dtype=data_info["dtype"], mode="r")
    val_data = np.memmap(data_info["val_bin"], dtype=data_info["dtype"], mode="r")

    # ----- 模型 -----
    # 让模型词表与实际分词器对齐
    model_cfg_dict = dict(model_cfg_dict)
    model_cfg_dict["vocab_size"] = data_info["vocab_size"]
    model_cfg_dict["block_size"] = int(train_cfg.block_size)
    model_cfg = GPT2Config.from_dict(model_cfg_dict)
    model = GPT2LMHeadModel(model_cfg).to(device)
    if train_cfg.compile:
        model = torch.compile(model)  # type: ignore[assignment]

    if dist_info.enabled:
        model = DDP(model, device_ids=[dist_info.local_rank] if device.type == "cuda" else None)

    raw_model = model.module if dist_info.enabled else model
    n_params = sum(p.numel() for p in raw_model.parameters())
    if dist_info.is_main:
        print(f"[setup] model params = {n_params/1e6:.2f}M")

    # ----- 优化器 -----
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    # ----- 训练 -----
    out_dir = Path(train_cfg.out_dir)
    if dist_info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    for step in range(int(train_cfg.max_steps) + 1):
        # 设学习率
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 评估
        if step % int(train_cfg.eval_interval) == 0 and dist_info.is_main:
            metrics = estimate_loss(raw_model, train_data, val_data, train_cfg, device)
            print(f"[eval] step={step} train={metrics['train']:.4f} val={metrics['val']:.4f}")
            ckpt = {
                "model": raw_model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "step": step,
            }
            torch.save(ckpt, out_dir / "ckpt.pt")

        if step == int(train_cfg.max_steps):
            break

        # 一个 step（含梯度累积）
        optimizer.zero_grad(set_to_none=True)
        for micro in range(int(train_cfg.grad_accum_steps)):
            x, y = get_batch(train_data, int(train_cfg.block_size), int(train_cfg.batch_size), device)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, loss = model(x, y)
                loss = loss / int(train_cfg.grad_accum_steps)
                scaler.scale(loss).backward()
            else:
                _, loss = model(x, y)
                loss = loss / int(train_cfg.grad_accum_steps)
                loss.backward()

        if train_cfg.grad_clip > 0:
            if use_amp and dtype == torch.float16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(train_cfg.grad_clip))

        if use_amp and dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % int(train_cfg.log_interval) == 0 and dist_info.is_main:
            dt = time.time() - t0
            print(f"[train] step={step} loss={loss.item()*int(train_cfg.grad_accum_steps):.4f} lr={lr:.2e} dt={dt:.1f}s")

    cleanup_distributed()


if __name__ == "__main__":
    main()
