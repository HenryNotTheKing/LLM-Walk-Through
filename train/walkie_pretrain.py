"""Walkie-Code-1B 预训练入口（多卡 DDP + 两阶段连续 WSD + 完善 checkpoint）。

特性：
    - **不修改** ``train/pretrain.py``：本脚本独立完整实现。
    - 单卡：``python -m train.walkie_pretrain --config configs/train/pretrain_walkie_tiny.yaml``
    - 多卡：``torchrun --nproc_per_node=N -m train.walkie_pretrain --config ...``
    - 两阶段（main / anneal）共用 **同一个 step 计数器** 与 **同一组优化器/调度器**，
      在 ``anneal_start_ratio`` 处切换数据流，学习率连续无跳变。
    - AdamW + Muon 双优化器；checkpoint 同步保存两者状态、scaler、调度器、RNG。

数据流水线占位：
    本仓库的数据处理由用户后续负责，这里仅约定通过
    ``data.stages.<main|anneal>.(bin|val_bin)`` 指向 ``np.memmap`` 文件
    （dtype 由 ``data.stages.<x>.dtype`` 给出）。如果路径不存在，验证集会回退到
    对应训练 bin，训练集则会用一段内存里随机生成的 token 流兜底，方便冒烟测试。
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.config import load_config
from core.utils.device import amp_enabled, select_device, select_dtype
from core.utils.distributed import cleanup_distributed, setup_distributed
from core.utils.walkie_checkpoint import (
    apply_walkie_checkpoint,
    load_walkie_checkpoint,
    prune_step_checkpoints,
    resolve_resume_path,
    save_walkie_checkpoint,
    unwrap_model,
)
from core.utils.walkie_optim import build_walkie_optimizers
from core.utils.walkie_schedule import WalkieWSDSchedule, apply_lrs_to_optimizers


# ---------------------------------------------------------------------------
# 日志时间格式化
# ---------------------------------------------------------------------------
def _fmt_elapsed(seconds: float) -> str:
    """格式化为 ``[+HH:MM:SS]``，作为日志时间戳前缀。"""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"[+{h:02d}:{m:02d}:{s:02d}]"


def _fmt_eta(seconds: float) -> str:
    """格式化剩余时间为 ``1h23m`` / ``23m45s`` / ``45s``；非数/负数返回 ``?``。"""
    if seconds != seconds or seconds < 0:
        return "?"
    total = int(seconds)
    if total >= 3600:
        h, rem = divmod(total, 3600)
        m = rem // 60
        return f"{h}h{m:02d}m"
    if total >= 60:
        m, s = divmod(total, 60)
        return f"{m}m{s:02d}s"
    return f"{total}s"


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
def _validate_token_dtype(dtype: np.dtype, vocab_size: int) -> None:
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"token bin dtype 必须是整数类型，得到 {dtype}")
    info = np.iinfo(dtype)
    if vocab_size - 1 > info.max:
        raise ValueError(
            f"vocab_size={vocab_size} 需要能表示 token id {vocab_size - 1}，"
            f"但 dtype={dtype} 最大值为 {info.max}"
        )


def _load_stage_data(
    stage_cfg,
    vocab_size: int,
    block_size: int,
    *,
    field: str = "bin",
    fallback_data: np.ndarray | None = None,
) -> np.ndarray:
    """加载一个阶段的 ``np.memmap``；不存在则生成随机 token 兜底（仅 smoke 用）。"""
    bin_path = stage_cfg.get(field, None) if stage_cfg is not None else None
    dtype_name = (stage_cfg.get("dtype", "uint16") if stage_cfg is not None else "uint16")
    dtype = np.dtype(dtype_name)
    _validate_token_dtype(dtype, vocab_size)
    if bin_path and Path(bin_path).exists():
        data = np.memmap(bin_path, dtype=dtype, mode="r")
        if len(data) > block_size:
            return data
        if fallback_data is None:
            raise ValueError(
                f"{bin_path} token 数={len(data)}，不足以构造 block_size={block_size} 的 batch"
            )
    if fallback_data is not None:
        return fallback_data
    # 兜底：内存里生成几百万 token 的随机数据，保证脚本能 smoke
    rng = np.random.default_rng(0)
    n = max(block_size * 64, 4096)
    return rng.integers(0, vocab_size, size=n, dtype=np.int64).astype(dtype)


def _batch_from_starts(
    data: np.ndarray,
    starts: np.ndarray,
    block_size: int,
    device: torch.device,
    *,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = starts[:, None] + np.arange(block_size, dtype=np.int64)[None, :]
    x = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64))
    y = torch.from_numpy(np.asarray(data[offsets + 1], dtype=np.int64))
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_positions = len(data) - block_size
    if n_positions <= 0:
        raise ValueError(f"data token 数={len(data)}，不足以构造 block_size={block_size} 的 batch")
    ix = torch.randint(n_positions, (batch_size,), generator=generator).numpy()
    return _batch_from_starts(data, ix.astype(np.int64, copy=False), block_size, device, pin_memory=pin_memory)


class ShuffledBlockSampler:
    """按 block 构造样本索引，shuffle 后无放回顺序读取。"""

    def __init__(
        self,
        data: np.ndarray,
        block_size: int,
        batch_size: int,
        device: torch.device,
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        name: str = "train",
        pin_memory: bool = False,
    ) -> None:
        self.data = data
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.name = str(name)
        self.pin_memory = bool(pin_memory)
        self.global_batch_size = self.batch_size * self.world_size
        self.num_samples = (len(data) - 1) // self.block_size
        if self.num_samples <= 0:
            raise ValueError(
                f"{self.name} token 数={len(data)}，不足以构造 block_size={self.block_size} 的样本"
            )
        if self.num_samples < self.global_batch_size:
            raise ValueError(
                f"{self.name} 样本数={self.num_samples} 小于 global batch 样本数="
                f"{self.global_batch_size}；请减小 batch_size 或 world_size"
            )
        dtype = np.uint32 if self.num_samples <= np.iinfo(np.uint32).max else np.int64
        self.order = np.arange(self.num_samples, dtype=dtype)
        self.epoch = 0
        self.cursor = 0
        self._reshuffle()

    @property
    def tokens_per_epoch(self) -> int:
        return self.num_samples * self.block_size

    def _reshuffle(self) -> None:
        self.order[:] = np.arange(self.num_samples, dtype=self.order.dtype)
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.order)

    def _ensure_room_for_global_batch(self) -> None:
        if self.cursor + self.global_batch_size <= self.num_samples:
            return
        self.epoch += 1
        self.cursor = 0
        self._reshuffle()

    def next_starts(self) -> np.ndarray:
        self._ensure_room_for_global_batch()
        begin = self.cursor + self.rank * self.batch_size
        end = begin + self.batch_size
        starts = self.order[begin:end].astype(np.int64, copy=False) * self.block_size
        self.cursor += self.global_batch_size
        return starts

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _batch_from_starts(
            self.data,
            self.next_starts(),
            self.block_size,
            self.device,
            pin_memory=self.pin_memory,
        )

    def skip_batches(self, n_batches: int) -> None:
        for _ in range(int(n_batches)):
            self._ensure_room_for_global_batch()
            self.cursor += self.global_batch_size

    def state_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "epoch": int(self.epoch),
            "cursor": int(self.cursor),
            "seed": int(self.seed),
            "num_samples": int(self.num_samples),
            "block_size": int(self.block_size),
            "batch_size": int(self.batch_size),
            "world_size": int(self.world_size),
            "global_batch_size": int(self.global_batch_size),
        }

    def state_mismatches(self, state: dict[str, object]) -> list[str]:
        mismatches: list[str] = []
        for field, expected in (
            ("num_samples", self.num_samples),
            ("block_size", self.block_size),
            ("batch_size", self.batch_size),
            ("world_size", self.world_size),
            ("global_batch_size", self.global_batch_size),
        ):
            if field not in state:
                mismatches.append(f"{field}=<missing> vs 当前={expected}")
                continue
            actual = int(state[field])
            if actual != expected:
                mismatches.append(f"{field}: ckpt={actual} vs 当前={expected}")
        return mismatches

    def load_state_dict(self, state: dict[str, object]) -> None:
        mismatches = self.state_mismatches(state)
        if mismatches:
            raise RuntimeError(
                f"恢复 {self.name} sampler 失败：" + "；".join(mismatches)
            )
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        if not (0 <= self.cursor <= self.num_samples):
            raise RuntimeError(
                f"恢复 {self.name} sampler 失败：cursor={self.cursor} 超出样本数={self.num_samples}"
            )
        self._reshuffle()


def _dist_barrier(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _restore_or_seed_data_generators(
    batch_rng: torch.Generator,
    eval_rng: torch.Generator,
    *,
    seed: int,
    rank: int,
    step: int,
    extra: dict[str, object] | None,
) -> None:
    batch_state = extra.get("batch_rng_state") if extra else None
    eval_state = extra.get("eval_rng_state") if extra else None

    if batch_state is not None:
        batch_rng.set_state(batch_state)
    else:
        batch_rng.manual_seed(int(seed) + 1_000_003 * rank + 97_003 * step)

    if eval_state is not None:
        eval_rng.set_state(eval_state)
    else:
        eval_rng.manual_seed(int(seed) + 2_000_003 * rank + 193_003 * step + 17)


def _checkpoint_extra(
    batch_rng: torch.Generator,
    eval_rng: torch.Generator,
    train_samplers: dict[str, ShuffledBlockSampler] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "batch_rng_state": batch_rng.get_state(),
        "eval_rng_state": eval_rng.get_state(),
    }
    if train_samplers is not None:
        extra["train_sampler_states"] = {
            name: sampler.state_dict() for name, sampler in train_samplers.items()
        }
    return extra


def _train_sampling_mode(train_cfg: Any) -> str:
    sampling_cfg = train_cfg.get("sampling", "shuffled_sequential")
    if sampling_cfg is None:
        return "shuffled_sequential"
    if isinstance(sampling_cfg, str):
        return sampling_cfg
    return str(sampling_cfg.get("mode", "shuffled_sequential"))


def _train_sampler_resume_policy(train_cfg: Any) -> str:
    sampling_cfg = train_cfg.get("sampling")
    if sampling_cfg is None or isinstance(sampling_cfg, str):
        return "auto"
    return str(sampling_cfg.get("resume_policy", "auto"))


def _fast_forward_train_samplers(
    train_samplers: dict[str, ShuffledBlockSampler],
    *,
    schedule: WalkieWSDSchedule,
    start_step: int,
    grad_accum: int,
) -> None:
    for step in range(int(start_step)):
        sampler = train_samplers[schedule.current_stage(step)]
        sampler.skip_batches(int(grad_accum))


def _restore_or_fast_forward_train_samplers(
    train_samplers: dict[str, ShuffledBlockSampler],
    *,
    schedule: WalkieWSDSchedule,
    start_step: int,
    grad_accum: int,
    extra: dict[str, object] | None,
    resume_policy: str,
) -> str:
    if resume_policy not in {"auto", "strict", "fast_forward", "reset"}:
        raise ValueError(
            "train.sampling.resume_policy 只支持 auto/strict/fast_forward/reset，"
            f"得到 {resume_policy}"
        )

    if int(start_step) <= 0:
        return "sampler 从头开始"

    if resume_policy == "reset":
        return "sampler 按 reset 策略从头开始，模型/优化器/调度器仍从 checkpoint 恢复"

    saved_states = extra.get("train_sampler_states") if extra else None
    if resume_policy != "fast_forward" and isinstance(saved_states, dict):
        mismatch_lines: list[str] = []
        for name, sampler in train_samplers.items():
            state = saved_states.get(name)
            if isinstance(state, dict):
                mismatches = sampler.state_mismatches(state)
                if mismatches:
                    mismatch_lines.append(f"{name}: " + "；".join(mismatches))
            else:
                mismatch_lines.append(f"{name}: checkpoint 缺少 sampler 状态")
        if not mismatch_lines:
            for name, sampler in train_samplers.items():
                state = saved_states.get(name)
                if isinstance(state, dict):
                    sampler.load_state_dict(state)
            return "sampler 精确恢复自 checkpoint"
        if resume_policy == "strict":
            raise RuntimeError("恢复 sampler 失败：" + " | ".join(mismatch_lines))
        _fast_forward_train_samplers(
            train_samplers,
            schedule=schedule,
            start_step=start_step,
            grad_accum=grad_accum,
        )
        return "sampler 状态与当前卡数/batch 不兼容，已按当前配置快进：" + " | ".join(mismatch_lines)

    if resume_policy == "strict":
        raise RuntimeError("恢复 sampler 失败：checkpoint 中没有可用的 train_sampler_states")

    _fast_forward_train_samplers(
        train_samplers,
        schedule=schedule,
        start_step=start_step,
        grad_accum=grad_accum,
    )
    if resume_policy == "fast_forward":
        return "sampler 按 fast_forward 策略忽略 checkpoint 状态并快进到当前 step"
    return "checkpoint 中没有可用 sampler 状态，已按当前配置快进到当前 step"


def _plain_cfg_dict(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        plain = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(plain, dict):
            return plain
        return {"value": plain}
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"value": cfg}


def _restore_schedule_state(
    schedule: WalkieWSDSchedule,
    payload_schedule: dict[str, object] | None,
    *,
    current_total_steps: int,
    expected_step: int,
) -> None:
    if payload_schedule is None:
        return

    saved_step = int(payload_schedule.get("step", expected_step))
    if saved_step != expected_step:
        raise RuntimeError(
            f"checkpoint step 不一致：schedule.step={saved_step} vs step={expected_step}"
        )

    current_state = schedule.state_dict()
    for field in ("warmup_steps", "anneal_start_ratio", "decay_shape", "tracks"):
        if payload_schedule.get(field) != current_state[field]:
            raise RuntimeError(
                f"resume 时 schedule 字段 {field} 不一致："
                f"ckpt={payload_schedule.get(field)} vs cfg={current_state[field]}"
            )

    schedule.load_state_dict(payload_schedule)
    if current_total_steps < schedule.step:
        raise RuntimeError(
            f"train.total_steps={current_total_steps} 小于已恢复 step={schedule.step}"
        )
    schedule.total_steps = int(current_total_steps)


def _init_swanlab_run(
    *,
    train_cfg,
    model_cfg: WalkieConfig,
    out_dir: Path,
    resume_path: str | None,
    resume_run_id: str | None,
) -> Any | None:
    swanlab_cfg = train_cfg.get("swanlab")
    if not swanlab_cfg or not bool(swanlab_cfg.get("enabled", False)):
        return None

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "train.swanlab.enabled=true 但当前环境未安装 swanlab；"
            "请先执行 uv sync --extra walkie"
        ) from exc

    mode = str(swanlab_cfg.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"train.swanlab.mode 只支持 online/offline/disabled，得到 {mode}")
    if mode == "disabled":
        return None

    api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key=api_key)
    elif mode == "online":
        try:
            from swanlab.package import has_api_key
        except ImportError:
            has_saved_api_key = False
        else:
            has_saved_api_key = has_api_key()
        if not has_saved_api_key:
            print(
                "[walkie/swanlab] 未检测到 SwanLab 登录凭据；请先执行 "
                "`uv run --extra walkie swanlab login` 完成登录，或设置环境变量 SWANLAB_API_KEY。"
            )

    config_payload = _plain_cfg_dict(train_cfg)
    if isinstance(config_payload.get("swanlab"), dict):
        config_payload["swanlab"].pop("api_key", None)

    run_id = str(swanlab_cfg.get("run_id") or resume_run_id or uuid4().hex[:8])
    resume_mode = swanlab_cfg.get("resume", None)
    if resume_mode is None and resume_run_id is not None:
        resume_mode = "allow"
    run = swanlab.init(
        project=str(swanlab_cfg.get("project", "walkie")),
        workspace=swanlab_cfg.get("workspace") or swanlab_cfg.get("entity"),
        experiment_name=swanlab_cfg.get("experiment_name") or swanlab_cfg.get("name"),
        description=swanlab_cfg.get("description") or swanlab_cfg.get("notes"),
        tags=list(swanlab_cfg.get("tags", [])),
        log_dir=str(out_dir),
        mode=mode,
        id=run_id,
        resume=resume_mode,
        config={
            "model": model_cfg.to_dict(),
            "train": config_payload,
            "runtime": {
                "out_dir": str(out_dir),
                "resume_from": str(resume_path) if resume_path is not None else None,
            },
        },
    )
    if run is None:
        raise RuntimeError("swanlab.init 返回 None，无法建立训练日志 run")

    print(
        f"[walkie/swanlab] project={swanlab_cfg.get('project', 'walkie')} "
        f"experiment={swanlab_cfg.get('experiment_name') or swanlab_cfg.get('name') or run_id} "
        f"mode={mode} run_id={run_id}"
    )
    return run


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def train(cfg, resume: str | None = None, init_from: str | None = None) -> None:
    t0 = time.time()

    def _log(message: str) -> None:
        """打印带 ``[+HH:MM:SS]`` 相对开始时间戳前缀的训练日志。"""
        print(f"{_fmt_elapsed(time.time() - t0)} {message}")

    dist_info = setup_distributed(cfg.distributed.backend)

    train_cfg = cfg.train
    model_cfg_dict = dict(cfg.model)

    torch.manual_seed(int(train_cfg.seed) + dist_info.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(train_cfg.seed) + dist_info.rank)

    device = select_device(train_cfg.device)
    dtype = select_dtype(train_cfg.dtype, device)
    use_amp = amp_enabled(device, dtype, bool(train_cfg.amp))

    # 训练吞吐优化：开启 TF32、matmul high、cudnn benchmark。仅在 CUDA 上生效。
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if dist_info.is_main:
        _log(
            f"[walkie/setup] device={device} dtype={dtype} amp={use_amp} "
            f"world_size={dist_info.world_size}"
        )

    # ----- 模型 -----
    model_cfg_dict.setdefault("block_size", int(train_cfg.block_size))
    if train_cfg.get("gradient_checkpointing", None) is not None:
        model_cfg_dict["gradient_checkpointing"] = bool(train_cfg.gradient_checkpointing)
    model_cfg = WalkieConfig.from_dict(model_cfg_dict)
    model = WalkieForCausalLM(model_cfg).to(device)

    # DDP 优先包装，再 compile：让 compile 能感知 DDP bucket，命中更优融合。
    # tied weights + grad_accum 的 no_sync 组合下 static_graph=True 不再总是安全，默认关闭，
    # 可通过 train.ddp_static_graph=true 显式开启。
    if dist_info.enabled:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            static_graph=bool(getattr(train_cfg, "ddp_static_graph", False)),
            
        )

    if bool(getattr(train_cfg, "compile", False)):
        compile_mode = str(getattr(train_cfg, "compile_mode", "default"))
        model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]

    raw_model = unwrap_model(model)
    if dist_info.is_main:
        n_params = raw_model.num_parameters()
        _log(
            f"[walkie/setup] model={model_cfg.model_name} params={n_params/1e6:.2f}M "
            f"gradient_checkpointing={model_cfg.gradient_checkpointing} "
            f"loss_chunk_size={model_cfg.loss_chunk_size}"
        )

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
    best_metric: float | None = None
    resume_extra: dict[str, object] | None = None
    if init_from is not None:
        if dist_info.is_main:
            _log(f"[walkie/init-from] {init_from}")
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
            _log(f"[walkie/resume] {resume_path}")
        payload = load_walkie_checkpoint(
            resume_path,
            map_location="cpu",
            strict_arch=True,
            expected_model_cfg=model_cfg.to_dict(),
        )
        info = apply_walkie_checkpoint(
            payload, model=model, optimizers=optimizers, scaler=scaler, restore_rng=True
        )
        start_step = int(info.get("step", 0))
        best_metric = info.get("best_metric")
        _restore_schedule_state(
            schedule,
            payload.get("schedule"),
            current_total_steps=int(train_cfg.total_steps),
            expected_step=start_step,
        )
        resume_extra = payload.get("extra")

    # ----- 数据：两阶段（训练 + 验证） -----
    stages_cfg = cfg.data.stages
    main_train_data = _load_stage_data(
        stages_cfg.main,
        model_cfg.vocab_size,
        int(train_cfg.block_size),
        field="bin",
    )
    anneal_train_data = _load_stage_data(
        stages_cfg.anneal,
        model_cfg.vocab_size,
        int(train_cfg.block_size),
        field="bin",
    )
    main_val_data = _load_stage_data(
        stages_cfg.main,
        model_cfg.vocab_size,
        int(train_cfg.block_size),
        field="val_bin",
        fallback_data=main_train_data,
    )
    anneal_val_data = _load_stage_data(
        stages_cfg.anneal,
        model_cfg.vocab_size,
        int(train_cfg.block_size),
        field="val_bin",
        fallback_data=anneal_train_data,
    )

    # ----- 训练循环 -----
    out_dir = Path(train_cfg.out_dir)
    if dist_info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    resume_swanlab_run_id = None
    if resume_extra is not None:
        if resume_extra.get("swanlab_run_id") is not None:
            resume_swanlab_run_id = str(resume_extra["swanlab_run_id"])
        elif resume_extra.get("wandb_run_id") is not None:
            resume_swanlab_run_id = str(resume_extra["wandb_run_id"])
    swanlab_run = None
    if dist_info.is_main:
        swanlab_run = _init_swanlab_run(
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            out_dir=out_dir,
            resume_path=resume,
            resume_run_id=resume_swanlab_run_id,
        )

    model.train()
    total_steps = int(train_cfg.total_steps)
    log_interval = int(train_cfg.log_interval)
    eval_interval = int(train_cfg.eval_interval)
    ckpt_interval = int(getattr(train_cfg, "ckpt_interval", 0)) or eval_interval
    grad_accum = int(train_cfg.grad_accum_steps)
    save_step_ckpts = bool(getattr(train_cfg, "save_step_checkpoints", True))
    keep_step_ckpts = int(getattr(train_cfg, "keep_step_checkpoints", 3))
    pin_memory = device.type == "cuda"
    tokens_per_step = (
        int(train_cfg.batch_size)
        * int(train_cfg.block_size)
        * grad_accum
        * dist_info.world_size
    )
    sampling_mode = _train_sampling_mode(train_cfg)
    if sampling_mode not in {"random", "shuffled_sequential"}:
        raise ValueError(
            "train.sampling.mode 只支持 random 或 shuffled_sequential，"
            f"得到 {sampling_mode}"
        )
    sampler_resume_policy = _train_sampler_resume_policy(train_cfg)

    batch_rng = torch.Generator(device="cpu")
    eval_rng = torch.Generator(device="cpu")
    _restore_or_seed_data_generators(
        batch_rng,
        eval_rng,
        seed=int(train_cfg.seed),
        rank=dist_info.rank,
        step=start_step,
        extra=resume_extra,
    )
    train_samplers: dict[str, ShuffledBlockSampler] | None = None
    if sampling_mode == "shuffled_sequential":
        train_samplers = {
            "main": ShuffledBlockSampler(
                main_train_data,
                int(train_cfg.block_size),
                int(train_cfg.batch_size),
                device,
                seed=int(train_cfg.seed) + 31_337,
                rank=dist_info.rank,
                world_size=dist_info.world_size,
                name="main",
                pin_memory=pin_memory,
            ),
            "anneal": ShuffledBlockSampler(
                anneal_train_data,
                int(train_cfg.block_size),
                int(train_cfg.batch_size),
                device,
                seed=int(train_cfg.seed) + 97_531,
                rank=dist_info.rank,
                world_size=dist_info.world_size,
                name="anneal",
                pin_memory=pin_memory,
            ),
        }
        sampler_resume_message = _restore_or_fast_forward_train_samplers(
            train_samplers,
            schedule=schedule,
            start_step=start_step,
            grad_accum=grad_accum,
            extra=resume_extra,
            resume_policy=sampler_resume_policy,
        )
        if dist_info.is_main and resume is not None:
            _log(
                f"[walkie/sampler-resume] policy={sampler_resume_policy} "
                f"{sampler_resume_message}"
            )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint_extra() -> dict[str, Any]:
        extra = _checkpoint_extra(batch_rng, eval_rng, train_samplers)
        if swanlab_run is not None:
            extra["swanlab_run_id"] = getattr(swanlab_run, "id", None) or getattr(swanlab_run, "run_id", None)
        elif resume_swanlab_run_id is not None:
            extra["swanlab_run_id"] = resume_swanlab_run_id
        return extra

    t0_loop = time.time()
    last_log_time = t0_loop
    last_log_step = start_step
    last_stage = schedule.current_stage(start_step)
    if dist_info.is_main:
        _log(
            f"[walkie/train] start_step={start_step} stage={last_stage} "
            f"total_steps={total_steps} global_tokens/step={tokens_per_step:,} "
            f"sampling={sampling_mode}"
        )
        main_val_source = "main.val_bin" if main_val_data is not main_train_data else "main.bin (fallback)"
        anneal_val_source = (
            "anneal.val_bin" if anneal_val_data is not anneal_train_data else "anneal.bin (fallback)"
        )
        _log(
            f"[walkie/data] main_train={len(main_train_data):,} main_val={len(main_val_data):,}"
            f" ({main_val_source}) anneal_train={len(anneal_train_data):,}"
            f" anneal_val={len(anneal_val_data):,} ({anneal_val_source})"
        )
        if train_samplers is not None:
            _log(
                f"[walkie/sampler] main_blocks={train_samplers['main'].num_samples:,} "
                f"main_tokens/epoch={train_samplers['main'].tokens_per_epoch:,} "
                f"anneal_blocks={train_samplers['anneal'].num_samples:,} "
                f"anneal_tokens/epoch={train_samplers['anneal'].tokens_per_epoch:,}"
            )

    for step in range(start_step, total_steps + 1):
        # 1) 学习率
        lrs = schedule.step_to(step)
        apply_lrs_to_optimizers(optimizers, lrs)

        # 2) 阶段切换日志
        stage = schedule.current_stage(step)
        if stage != last_stage and dist_info.is_main:
            _log(
                f"[walkie/stage] step={step} switch {last_stage} -> {stage} "
                f"(lrs={lrs})"
            )
            last_stage = stage
        cur_train_data = anneal_train_data if stage == "anneal" else main_train_data
        cur_eval_data = anneal_val_data if stage == "anneal" else main_val_data
        eval_split_name = "anneal_val" if stage == "anneal" else "main_val"

        # 3) eval / ckpt
        if step % eval_interval == 0 and step > start_step:
            raw_model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(int(train_cfg.eval_iters)):
                    x, y = get_batch(
                        cur_eval_data,
                        int(train_cfg.block_size),
                        int(train_cfg.batch_size),
                        device,
                        generator=eval_rng,
                        pin_memory=pin_memory,
                    )
                    if use_amp:
                        with torch.autocast(device_type=device.type, dtype=dtype):
                            _, loss = raw_model(x, y, return_logits=False)
                    else:
                        _, loss = raw_model(x, y, return_logits=False)
                    losses.append(loss.item())
            loss_tensor = torch.tensor(float(np.mean(losses)), device=device)
            if dist_info.enabled:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= dist_info.world_size
            val = float(loss_tensor.item())
            if dist_info.is_main:
                _log(
                    f"[walkie/eval] step={step} stage={stage} split={eval_split_name} "
                    f"val_loss={val:.4f} lrs={lrs}"
                )
            raw_model.train()

            is_best = best_metric is None or val < best_metric
            if is_best:
                best_metric = val
                _dist_barrier(dist_info.enabled)
                if dist_info.is_main:
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
                        train_cfg=_plain_cfg_dict(train_cfg),
                        extra=checkpoint_extra(),
                        format="best",
                    )
                _dist_barrier(dist_info.enabled)
            if dist_info.is_main and swanlab_run is not None:
                swanlab_run.log(
                    {
                        "train/step": step,
                        "eval/loss": val,
                        "eval/best_loss": float(best_metric) if best_metric is not None else val,
                        "eval/is_best": int(is_best),
                        "eval/is_anneal_split": int(stage == "anneal"),
                        "lr/adamw": float(lrs["adamw"]),
                        "lr/muon": float(lrs["muon"]),
                        "train/is_anneal": int(stage == "anneal"),
                    },
                    step=step,
                )

        if step % ckpt_interval == 0 and step > start_step:
            _dist_barrier(dist_info.enabled)
            if dist_info.is_main:
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
                    train_cfg=_plain_cfg_dict(train_cfg),
                    extra=checkpoint_extra(),
                    format="latest",
                )
                if save_step_ckpts:
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
                        train_cfg=_plain_cfg_dict(train_cfg),
                        extra=checkpoint_extra(),
                        format="step",
                        tag=step,
                    )
                    prune_step_checkpoints(out_dir, keep_step_ckpts)
            _dist_barrier(dist_info.enabled)

        if step == total_steps:
            break

        # 4) 一个 step（梯度累积）
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        # 用 GPU 端 tensor 累积 loss，避免每个 microstep 触发 GPU->CPU 同步。
        loss_accum = torch.zeros((), device=device)
        for micro_step in range(grad_accum):
            if train_samplers is not None:
                x, y = train_samplers[stage].next_batch()
            else:
                x, y = get_batch(
                    cur_train_data,
                    int(train_cfg.block_size),
                    int(train_cfg.batch_size),
                    device,
                    generator=batch_rng,
                    pin_memory=pin_memory,
                )
            sync_context = (
                model.no_sync()
                if dist_info.enabled and micro_step < grad_accum - 1
                else nullcontext()
            )
            with sync_context:
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        _, loss = model(x, y, return_logits=False)
                    loss = loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    _, loss = model(x, y, return_logits=False)
                    loss = loss / grad_accum
                    loss.backward()
                loss_accum = loss_accum + loss.detach()

        grad_norm_tensor: torch.Tensor | None = None
        if float(train_cfg.grad_clip) > 0:
            if use_amp and dtype == torch.float16:
                for opt in optimizers.values():
                    scaler.unscale_(opt)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), float(train_cfg.grad_clip)
            )

        if use_amp and dtype == torch.float16:
            for opt in optimizers.values():
                scaler.step(opt)
            scaler.update()
        else:
            for opt in optimizers.values():
                opt.step()

        completed_step = step + 1
        if completed_step % log_interval == 0 and dist_info.is_main:
            now = time.time()
            dt = now - last_log_time
            elapsed = now - t0_loop
            logged_steps = max(1, completed_step - last_log_step)
            tokens_sec = tokens_per_step * logged_steps / max(dt, 1e-9)
            sec_per_step = dt / logged_steps
            remaining_steps = max(0, total_steps - completed_step)
            eta_seconds = remaining_steps * sec_per_step
            eta_str = _fmt_eta(eta_seconds)
            progress = completed_step / total_steps if total_steps > 0 else 0.0
            last_log_time = now
            last_log_step = completed_step
            # 命中日志时再做一次 GPU->CPU 同步，把 loss/grad_norm 取出来。
            # loss_accum 已是各 microstep 累加后的 grad_accum-平均 loss。
            loss_value = float(loss_accum.item())
            grad_norm = float(grad_norm_tensor.item()) if grad_norm_tensor is not None else None
            peak_gb = None
            mem = ""
            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                mem = f" mem_peak={peak_gb:.2f}GiB"
            grad = "" if grad_norm is None else f" grad_norm={grad_norm:.2f}"
            _log(
                f"[walkie/step] step={completed_step} stage={stage} "
                f"loss={loss_value:.4f} "
                f"lr_adamw={lrs['adamw']:.2e} lr_muon={lrs['muon']:.2e} "
                f"tok/s={tokens_sec:,.0f} eta={eta_str}{grad}{mem}"
            )
            if swanlab_run is not None:
                payload = {
                    "train/step": completed_step,
                    "train/loss": loss_value,
                    "train/progress": float(progress),
                    "lr/adamw": float(lrs["adamw"]),
                    "lr/muon": float(lrs["muon"]),
                    "perf/tokens_per_sec": float(tokens_sec),
                    "perf/elapsed_sec": float(elapsed),
                    "perf/sec_per_step": float(sec_per_step),
                    "perf/eta_sec": float(eta_seconds),
                    "train/is_anneal": int(stage == "anneal"),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = float(grad_norm)
                if peak_gb is not None:
                    payload["system/mem_peak_gib"] = float(peak_gb)
                swanlab_run.log(payload, step=completed_step)

    # 收尾：写一次 latest，确保最后一步状态落盘
    _dist_barrier(dist_info.enabled)
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
            train_cfg=_plain_cfg_dict(train_cfg),
            extra=checkpoint_extra(),
            format="latest",
        )
        if save_step_ckpts:
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
                train_cfg=_plain_cfg_dict(train_cfg),
                extra=checkpoint_extra(),
                format="step",
                tag=total_steps,
            )
            prune_step_checkpoints(out_dir, keep_step_ckpts)
        if swanlab_run is not None:
            final_payload = {
                "train/step": int(total_steps),
                "train/final_step": int(total_steps),
                "train/final_stage": schedule.current_stage(total_steps),
                "train/latest_checkpoint": str(out_dir / "latest.pt"),
            }
            if best_metric is not None:
                final_payload["eval/best_loss"] = float(best_metric)
            swanlab_run.log(final_payload, step=total_steps)
            swanlab_run.finish()
        _log(f"[walkie/done] saved to {out_dir}")
    _dist_barrier(dist_info.enabled)

    cleanup_distributed()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    try:
        train(cfg, resume=args.resume, init_from=args.init_from)
    finally:
        # 即使训练过程异常退出，也保证 NCCL/Gloo process group 被优雅释放。
        cleanup_distributed()


if __name__ == "__main__":
    main()
