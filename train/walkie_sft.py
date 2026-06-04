"""Walkie SFT entrypoint with resumable checkpointing and SwanLab logging.

Single GPU:
    CUDA_VISIBLE_DEVICES=1 python -m train.walkie_sft --config configs/train/sft_walkie.yaml \
        --init-from runs/walkie_code_0.5b/latest.pt

DDP:
    torchrun --nproc-per-node=2 -m train.walkie_sft --config configs/train/sft_walkie.yaml ...
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

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
from posttrain.data.chat_template import ChatTemplate
from posttrain.data.sft_dataset import SFTIterableDataset, collate_sft_batch
from posttrain.utils.schedule import WarmupDecaySchedule, apply_lrs


class TokenizerAdapter:
    def __init__(self, tokenizer: Any, *, eos_token: str, pad_token: str) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = int(tokenizer.token_to_id(eos_token))
        self.pad_token_id = int(tokenizer.token_to_id(pad_token))

    def encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        return [int(item) for item in encoded.ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--init-from", default=None, type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    dist_info = setup_distributed(str(getattr(cfg.distributed, "backend", "none")))
    try:
        train(cfg, resume=args.resume, init_from=args.init_from, dist_info=dist_info)
    finally:
        cleanup_distributed()


def train(cfg, *, resume: str | None, init_from: str | None, dist_info) -> None:
    started_at = time.time()

    def log(message: str) -> None:
        print(f"{_fmt_elapsed(time.time() - started_at)} {message}")

    train_cfg = cfg.train
    data_cfg = cfg.data
    seed = int(getattr(train_cfg, "seed", 42)) + int(dist_info.rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(str(train_cfg.device))
    if device.type == "cuda":
        cuda_index = dist_info.local_rank if dist_info.enabled else (device.index or 0)
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    dtype = select_dtype(str(train_cfg.dtype), device)
    use_amp = amp_enabled(device, dtype, bool(train_cfg.amp))

    tokenizer = _load_tokenizer(data_cfg.tokenizer_path, data_cfg.eos_token, data_cfg.pad_token)
    template = _resolve_template(tokenizer, data_cfg)
    paths = _data_paths(data_cfg)
    if not paths:
        raise ValueError("data.paths must point to JSONL/JSON/Parquet SFT data")

    model_cfg_dict = _plain_cfg_dict(cfg.model)
    model_cfg_dict.setdefault("block_size", int(train_cfg.block_size))
    if train_cfg.get("gradient_checkpointing", None) is not None:
        model_cfg_dict["gradient_checkpointing"] = bool(train_cfg.gradient_checkpointing)
    model_cfg = WalkieConfig.from_dict(model_cfg_dict)
    model = WalkieForCausalLM(model_cfg).to(device)

    if dist_info.enabled:
        model = DDP(
            model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            static_graph=bool(getattr(train_cfg, "ddp_static_graph", False)),
        )
    if bool(getattr(train_cfg, "compile", False)):
        model = torch.compile(model, mode=str(getattr(train_cfg, "compile_mode", "default")))  # type: ignore[assignment]
    raw_model = unwrap_model(model)

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
    schedule = WarmupDecaySchedule.from_config(
        total_steps=int(train_cfg.total_steps),
        warmup_steps=int(train_cfg.warmup_steps),
        decay_shape=str(train_cfg.decay_shape),
        tracks={
            "adamw": {"peak_lr": float(train_cfg.adamw.peak_lr), "final_lr": float(train_cfg.adamw.final_lr)},
            "muon": {"peak_lr": float(train_cfg.muon.peak_lr), "final_lr": float(train_cfg.muon.final_lr)},
        },
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))
    start_step = 0
    best_metric: float | None = None
    resume_extra: dict[str, Any] | None = None
    global_sample_index = int(getattr(train_cfg, "resume_sample_index", 0))

    if init_from is not None:
        if dist_info.is_main:
            log(f"[walkie/sft/init-from] {resolve_resume_path(init_from)}")
        payload = load_walkie_checkpoint(
            resolve_resume_path(init_from),
            map_location="cpu",
            strict_arch=True,
            expected_model_cfg=model_cfg.to_dict(),
        )
        apply_walkie_checkpoint(payload, model=model, optimizers=None, scaler=None, restore_rng=False)
    resume_path: Path | None = None
    if resume is not None:
        resume_path = resolve_resume_path(resume)
        if dist_info.is_main:
            log(f"[walkie/sft/resume] {resume_path}")
        payload = load_walkie_checkpoint(
            resume_path,
            map_location="cpu",
            strict_arch=True,
            expected_model_cfg=model_cfg.to_dict(),
        )
        info = apply_walkie_checkpoint(payload, model=model, optimizers=optimizers, scaler=scaler, restore_rng=True)
        start_step = int(info.get("step", 0))
        best_metric = info.get("best_metric")
        _restore_schedule_state(
            schedule,
            payload.get("schedule"),
            current_total_steps=int(train_cfg.total_steps),
            expected_step=start_step,
        )
        resume_extra = payload.get("extra", {})
        global_sample_index = int(extra_data_state(resume_extra).get("global_sample_index", global_sample_index))

    out_dir = Path(train_cfg.out_dir)
    if dist_info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    _dist_barrier(dist_info.enabled)

    resume_swanlab_run_id = _swanlab_resume_run_id(resume_extra)
    swanlab_run = None
    if dist_info.is_main:
        swanlab_run = _init_swanlab_run(
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            out_dir=out_dir,
            resume_path=str(resume_path) if resume_path is not None else resume,
            resume_run_id=resume_swanlab_run_id,
        )

    dataset = SFTIterableDataset(
        paths,
        tokenizer=tokenizer,
        template=template,
        max_length=int(train_cfg.block_size),
        rank=dist_info.rank,
        world_size=dist_info.world_size,
        start_index=global_sample_index,
    )
    loader = _make_loader(dataset, tokenizer, train_cfg)
    iterator = iter(loader)

    total_steps = int(train_cfg.total_steps)
    stop_step = _resolve_stop_step(train_cfg, total_steps=total_steps)
    if stop_step < start_step:
        raise RuntimeError(f"train.stop_step={stop_step} is smaller than resumed step={start_step}")
    grad_accum = int(train_cfg.grad_accum_steps)
    log_interval = int(train_cfg.log_interval)
    ckpt_interval = int(getattr(train_cfg, "ckpt_interval", 0)) or max(1, log_interval)
    save_step_ckpts = bool(getattr(train_cfg, "save_step_checkpoints", False))
    keep_step_ckpts = int(getattr(train_cfg, "keep_step_checkpoints", 3))
    tokens_per_step = int(train_cfg.batch_size) * int(train_cfg.block_size) * grad_accum * dist_info.world_size

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if dist_info.is_main:
        log(
            f"[walkie/sft/start] start_step={start_step} stop_step={stop_step} "
            f"total_steps={total_steps} device={device} dtype={dtype} amp={use_amp} "
            f"world_size={dist_info.world_size} data_paths={paths}"
        )

    def checkpoint_extra() -> dict[str, Any]:
        return _build_checkpoint_extra(
            global_sample_index=global_sample_index,
            swanlab_run=swanlab_run,
            resume_swanlab_run_id=resume_swanlab_run_id,
        )

    model.train()
    last_log_time = time.time()
    last_log_step = start_step
    for step in range(start_step, stop_step):
        lrs = schedule.step_to(step)
        apply_lrs(optimizers, lrs)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), device=device)
        grad_norm_tensor: torch.Tensor | None = None
        for micro_step in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                global_sample_index = 0
                dataset = SFTIterableDataset(
                    paths,
                    tokenizer=tokenizer,
                    template=template,
                    max_length=int(train_cfg.block_size),
                    rank=dist_info.rank,
                    world_size=dist_info.world_size,
                    start_index=0,
                )
                loader = _make_loader(dataset, tokenizer, train_cfg)
                iterator = iter(loader)
                batch = next(iterator)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            sync_context = _no_sync_context(model, enabled=dist_info.enabled and micro_step < grad_accum - 1)
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                    _, loss = model(input_ids, labels, return_logits=False)
                    assert loss is not None
                    loss = loss / grad_accum
                if use_amp and dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            loss_accum = loss_accum + loss.detach()
            global_sample_index += int(input_ids.shape[0]) * dist_info.world_size

        if float(train_cfg.grad_clip) > 0:
            if use_amp and dtype == torch.float16:
                for optimizer in optimizers.values():
                    scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(train_cfg.grad_clip))
        if use_amp and dtype == torch.float16:
            for optimizer in optimizers.values():
                scaler.step(optimizer)
            scaler.update()
        else:
            for optimizer in optimizers.values():
                optimizer.step()

        completed_step = step + 1
        if dist_info.enabled:
            loss_tensor = loss_accum.detach().clone()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist_info.world_size
        else:
            loss_tensor = loss_accum

        if dist_info.is_main and completed_step % log_interval == 0:
            now = time.time()
            elapsed = now - started_at
            interval_steps = max(1, completed_step - last_log_step)
            interval_seconds = max(now - last_log_time, 1e-9)
            tokens_sec = tokens_per_step * interval_steps / interval_seconds
            sec_per_step = interval_seconds / interval_steps
            eta = _fmt_eta(max(0, stop_step - completed_step) * sec_per_step)
            last_log_time = now
            last_log_step = completed_step
            loss_value = float(loss_tensor.item())
            grad_norm = float(grad_norm_tensor.item()) if grad_norm_tensor is not None else None
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None
            grad_text = "" if grad_norm is None else f" grad_norm={grad_norm:.2f}"
            mem_text = "" if peak_gb is None else f" mem_peak={peak_gb:.2f}GiB"
            log(
                f"[walkie/sft/step] step={completed_step} loss={loss_value:.4f} "
                f"lr_adamw={lrs['adamw']:.2e} lr_muon={lrs['muon']:.2e} "
                f"tok/s={tokens_sec:,.0f} eta={eta}{grad_text}{mem_text}"
            )
            if swanlab_run is not None:
                payload = {
                    "train/step": completed_step,
                    "train/loss": loss_value,
                    "train/progress": completed_step / max(1, total_steps),
                    "lr/adamw": float(lrs["adamw"]),
                    "lr/muon": float(lrs["muon"]),
                    "perf/tokens_per_sec": float(tokens_sec),
                    "perf/elapsed_sec": float(elapsed),
                    "perf/sec_per_step": float(sec_per_step),
                    "train/global_sample_index": int(global_sample_index),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = float(grad_norm)
                if peak_gb is not None:
                    payload["system/mem_peak_gib"] = float(peak_gb)
                swanlab_run.log(payload, step=completed_step)

        if completed_step % ckpt_interval == 0:
            _save_checkpoint(
                out_dir,
                model,
                optimizers,
                scaler,
                schedule,
                completed_step,
                model_cfg,
                train_cfg,
                best_metric,
                checkpoint_extra(),
                dist_info,
                save_step=save_step_ckpts,
                keep_step_checkpoints=keep_step_ckpts,
            )

    _save_checkpoint(
        out_dir,
        model,
        optimizers,
        scaler,
        schedule,
        stop_step,
        model_cfg,
        train_cfg,
        best_metric,
        checkpoint_extra(),
        dist_info,
        save_step=save_step_ckpts,
        keep_step_checkpoints=keep_step_ckpts,
    )
    if dist_info.is_main and swanlab_run is not None:
        swanlab_run.log(
            {
                "train/step": int(stop_step),
                "train/segment_stop_step": int(stop_step),
                "train/total_steps": int(total_steps),
            },
            step=int(stop_step),
        )
        swanlab_run.finish()
    if dist_info.is_main:
        log(f"[walkie/sft/done] saved to {out_dir} at step={stop_step}")


def _save_checkpoint(
    out_dir: Path,
    model,
    optimizers,
    scaler,
    schedule,
    step: int,
    model_cfg: WalkieConfig,
    train_cfg,
    best_metric: float | None,
    extra: dict[str, Any],
    dist_info,
    *,
    save_step: bool,
    keep_step_checkpoints: int,
) -> None:
    _dist_barrier(dist_info.enabled)
    if dist_info.is_main:
        schedule_state = schedule.state_dict()
        schedule_state["step"] = int(step)
        save_walkie_checkpoint(
            out_dir,
            model=model,
            optimizers=optimizers,
            scaler=scaler,
            schedule_state=schedule_state,
            step=step,
            stage="sft",
            best_metric=best_metric,
            model_cfg=model_cfg.to_dict(),
            train_cfg=_plain_cfg_dict(train_cfg),
            extra=extra,
            format="latest",
        )
        if save_step:
            save_walkie_checkpoint(
                out_dir,
                model=model,
                optimizers=optimizers,
                scaler=scaler,
                schedule_state=schedule.state_dict(),
                step=step,
                stage="sft",
                best_metric=best_metric,
                model_cfg=model_cfg.to_dict(),
                train_cfg=_plain_cfg_dict(train_cfg),
                extra=extra,
                format="step",
                tag=step,
            )
            prune_step_checkpoints(out_dir, int(keep_step_checkpoints))
    _dist_barrier(dist_info.enabled)


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
        raise RuntimeError("train.swanlab.enabled=true but swanlab is not installed") from exc

    mode = str(swanlab_cfg.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"train.swanlab.mode must be online/offline/disabled, got {mode}")
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
            print("[walkie/swanlab] no SwanLab credential detected; run `uv run swanlab login` or set SWANLAB_API_KEY")

    config_payload = _plain_cfg_dict(train_cfg)
    if isinstance(config_payload.get("swanlab"), dict):
        config_payload["swanlab"].pop("api_key", None)

    run_id = str(swanlab_cfg.get("run_id") or resume_run_id or uuid4().hex[:8])
    resume_mode = swanlab_cfg.get("resume", None)
    if resume_mode is None:
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
        raise RuntimeError("swanlab.init returned None")
    try:
        setattr(run, "_walkie_run_id", run_id)
    except Exception:
        pass
    print(
        f"[walkie/swanlab] project={swanlab_cfg.get('project', 'walkie')} "
        f"experiment={swanlab_cfg.get('experiment_name') or swanlab_cfg.get('name') or run_id} "
        f"mode={mode} run_id={run_id}"
    )
    return run


def _restore_schedule_state(
    schedule: WarmupDecaySchedule,
    payload_schedule: dict[str, Any] | None,
    *,
    current_total_steps: int,
    expected_step: int,
) -> None:
    if payload_schedule is None:
        return
    saved_step = int(payload_schedule.get("step", expected_step))
    if saved_step != expected_step:
        raise RuntimeError(f"checkpoint step mismatch: schedule.step={saved_step} vs step={expected_step}")
    current_state = schedule.state_dict()
    for field in ("warmup_steps", "decay_shape", "tracks"):
        if payload_schedule.get(field) != current_state[field]:
            raise RuntimeError(f"resume schedule field {field} mismatch: ckpt={payload_schedule.get(field)} vs cfg={current_state[field]}")
    if int(current_total_steps) < saved_step:
        raise RuntimeError(f"train.total_steps={current_total_steps} is smaller than resumed step={saved_step}")
    schedule.load_state_dict(payload_schedule)
    schedule.total_steps = int(current_total_steps)
    schedule._step = saved_step


def _load_tokenizer(path: str | None, eos_token: str, pad_token: str) -> TokenizerAdapter:
    if path is None:
        raise ValueError("data.tokenizer_path is required for SFT")
    from tokenizers import Tokenizer

    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    return TokenizerAdapter(Tokenizer.from_file(str(tokenizer_path)), eos_token=eos_token, pad_token=pad_token)


def _resolve_template(tokenizer: TokenizerAdapter, data_cfg) -> ChatTemplate:
    requested = str(data_cfg.template)
    if requested == "chatml_lowfreq_alias":
        has_chatml = tokenizer.tokenizer.token_to_id("<|im_start|>") is not None and tokenizer.tokenizer.token_to_id("<|im_end|>") is not None
        if not has_chatml:
            print("[walkie/sft] tokenizer lacks ChatML alias tokens; falling back to plain_eot template")
            requested = "plain_eot"
    return ChatTemplate(kind=requested, eos_token=str(data_cfg.eos_token))


def _data_paths(data_cfg) -> list[str]:
    if data_cfg.get("paths") is not None:
        return [str(path) for path in data_cfg.paths]
    if data_cfg.get("path") is not None:
        return [str(data_cfg.path)]
    return []


def _make_loader(dataset: SFTIterableDataset, tokenizer: TokenizerAdapter, train_cfg) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        collate_fn=partial(collate_sft_batch, pad_token_id=tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )


def _no_sync_context(model, *, enabled: bool):
    if not enabled:
        return nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if no_sync is None:
        return nullcontext()
    return no_sync()


def _resolve_stop_step(train_cfg, *, total_steps: int) -> int:
    stop_step = getattr(train_cfg, "stop_step", None)
    if stop_step is None:
        return int(total_steps)
    return min(int(total_steps), int(stop_step))


def _build_checkpoint_extra(
    *,
    global_sample_index: int,
    swanlab_run: Any | None,
    resume_swanlab_run_id: str | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {"data_state": {"global_sample_index": int(global_sample_index)}}
    if swanlab_run is not None:
        run_id = getattr(swanlab_run, "id", None) or getattr(swanlab_run, "run_id", None) or getattr(swanlab_run, "_walkie_run_id", None)
        if run_id is not None:
            extra["swanlab_run_id"] = str(run_id)
    elif resume_swanlab_run_id is not None:
        extra["swanlab_run_id"] = str(resume_swanlab_run_id)
    return extra


def _swanlab_resume_run_id(extra: dict[str, Any] | None) -> str | None:
    if not extra:
        return None
    if extra.get("swanlab_run_id") is not None:
        return str(extra["swanlab_run_id"])
    if extra.get("wandb_run_id") is not None:
        return str(extra["wandb_run_id"])
    return None


def extra_data_state(extra: dict[str, Any] | None) -> dict[str, Any]:
    if not extra:
        return {}
    data_state = extra.get("data_state")
    return data_state if isinstance(data_state, dict) else {}


def _plain_cfg_dict(cfg) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _dist_barrier(enabled: bool) -> None:
    if enabled:
        dist.barrier()


def _fmt_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"[+{hours:02d}:{minutes:02d}:{secs:02d}]"


def _fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "?"
    total = int(seconds)
    if total >= 3600:
        hours, rem = divmod(total, 3600)
        return f"{hours}h{rem // 60:02d}m"
    if total >= 60:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m{secs:02d}s"
    return f"{total}s"


if __name__ == "__main__":
    main()