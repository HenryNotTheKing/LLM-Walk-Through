"""Segmented RL training loop with full HumanEval evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from core.utils.config import load_config
from core.utils.walkie_checkpoint import load_walkie_checkpoint, resolve_resume_path
from scripts.run_sft_bench_loop import (
    append_jsonl,
    build_export_command,
    checkpoint_step,
    checkpoint_swanlab_run_id,
    choose_checkpoint_args,
    log_bench_to_swanlab,
    next_stop_step,
    parse_bench_summary,
    run_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmented Walkie DAPO/GRPO with full HumanEval evaluation")
    parser.add_argument("--config", default="configs/train/rl_walkie_dapo_deepcoder.yaml")
    parser.add_argument("--init-from", default="runs/walkie_code_0.5b_sft_kodcode_bench/latest.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--segment-steps", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--sandbox-url", action="append", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-timeout", type=float, default=None)
    parser.add_argument("--eval-max-concurrency", type=int, default=None)
    parser.add_argument("--skip-sandbox-smoke", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-override", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    loop_cfg = cfg.get("bench_loop", {}) or {}
    eval_cfg = loop_cfg.get("eval", {}) or {}

    out_dir = Path(args.out_dir or cfg.train.out_dir)
    total_steps = int(args.total_steps or cfg.train.total_steps)
    segment_steps = int(args.segment_steps or loop_cfg.get("segment_steps", 200))
    eval_interval = int(args.eval_interval if args.eval_interval is not None else loop_cfg.get("eval_interval", 200))
    dataset = str(args.dataset or eval_cfg.get("dataset", "openai_humaneval"))
    bench_root = str(args.bench_root or eval_cfg.get("bench_root", "data/bench"))
    sandbox_urls = list(args.sandbox_url or eval_cfg.get("sandbox_urls", ["http://127.0.0.1:18901"]))
    eval_batch_size = int(args.eval_batch_size or eval_cfg.get("batch_size", 1024))
    eval_max_tokens = int(args.eval_max_tokens or eval_cfg.get("max_tokens", 512))
    eval_temperature = float(args.eval_temperature if args.eval_temperature is not None else eval_cfg.get("temperature", 0.2))
    eval_top_p = float(args.eval_top_p if args.eval_top_p is not None else eval_cfg.get("top_p", 0.95))
    eval_timeout = float(args.eval_timeout if args.eval_timeout is not None else eval_cfg.get("timeout", 10.0))
    eval_max_concurrency = int(args.eval_max_concurrency or eval_cfg.get("max_concurrency", 32))

    out_dir.mkdir(parents=True, exist_ok=True)
    preserve_base_checkpoint(out_dir, init_from=args.init_from)
    history_path = out_dir / "humaneval_history.jsonl"
    best_state = load_best_state(out_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    train_processes = visible_gpu_count(str(args.gpu)) if effective_distributed_backend(cfg, args.train_override) != "none" else 1

    while checkpoint_step(out_dir) < total_steps:
        current_step = checkpoint_step(out_dir)
        stop_step = next_stop_step(current_step, total_steps=total_steps, segment_steps=segment_steps)
        train_cmd = build_train_command(
            config=args.config,
            out_dir=out_dir,
            init_from=args.init_from,
            total_steps=total_steps,
            stop_step=stop_step,
            train_overrides=args.train_override,
            nproc_per_node=train_processes,
        )
        run_command(train_cmd, env=env, dry_run=args.dry_run)
        if args.dry_run:
            break

        completed_step = checkpoint_step(out_dir)
        if completed_step < stop_step:
            raise RuntimeError(f"training stopped at step={completed_step}, expected at least {stop_step}")
        if args.no_eval or not should_run_eval(completed_step, total_steps=total_steps, eval_interval=eval_interval):
            continue

        export_dir = out_dir / "hf_exports" / f"step_{completed_step:08d}"
        eval_output = out_dir / "humaneval_eval" / f"step_{completed_step:08d}_full"
        export_cmd = build_export_command(out_dir / "latest.pt", export_dir, tokenizer_path=str(cfg.data.tokenizer_path))
        run_command(export_cmd, env=env, dry_run=args.dry_run)
        eval_cmd = build_humaneval_eval_command(
            model_dir=export_dir,
            output_dir=eval_output,
            dataset=dataset,
            bench_root=bench_root,
            sandbox_urls=sandbox_urls,
            batch_size=eval_batch_size,
            max_tokens=eval_max_tokens,
            temperature=eval_temperature,
            top_p=eval_top_p,
            timeout=eval_timeout,
            max_concurrency=eval_max_concurrency,
            skip_sandbox_smoke=args.skip_sandbox_smoke,
        )
        eval_started = time.time()
        run_command(eval_cmd, env=env, dry_run=args.dry_run)
        eval_seconds = time.time() - eval_started

        summary = parse_bench_summary(eval_output / "summary.json")
        pass_at_1 = dataset_pass_at_1(summary, dataset)
        row = {
            "step": completed_step,
            "macro_pass@1": pass_at_1,
            "dataset": dataset,
            "is_full_eval": True,
            "limit": None,
            "eval_seconds": eval_seconds,
            "export_dir": str(export_dir),
            "output_dir": str(eval_output),
            "datasets": summary.get("datasets", {}),
        }
        append_jsonl(history_path, row)
        run_id = checkpoint_swanlab_run_id(out_dir / "latest.pt")
        log_bench_to_swanlab(cfg.train, run_id=run_id, row=row, out_dir=out_dir)
        if pass_at_1 is not None and (best_state.get("pass@1") is None or pass_at_1 > float(best_state["pass@1"])):
            best_state = update_best_humaneval_checkpoint(out_dir, metric=pass_at_1, summary=row)

    return 0


def build_train_command(*, config: str, out_dir: Path, init_from: str, total_steps: int, stop_step: int, train_overrides: Sequence[str] | None = None, nproc_per_node: int = 1) -> list[str]:
    if int(nproc_per_node) > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={int(nproc_per_node)}", "-m", "train.walkie_rl", "--config", str(config)]
    else:
        cmd = [sys.executable, "-m", "train.walkie_rl", "--config", str(config)]
    cmd.extend(choose_checkpoint_args(out_dir, init_from=init_from))
    cmd.extend(
        [
            f"train.out_dir={out_dir}",
            f"train.total_steps={int(total_steps)}",
            f"train.stop_step={int(stop_step)}",
        ]
    )
    cmd.extend(train_overrides or [])
    return cmd


def visible_gpu_count(gpu_arg: str) -> int:
    return max(1, len([item for item in str(gpu_arg).split(",") if item.strip()]))


def effective_distributed_backend(cfg, train_overrides: Sequence[str]) -> str:
    for override in reversed(list(train_overrides or [])):
        if override.startswith("distributed.backend="):
            return override.split("=", 1)[1]
    return str(cfg.get("distributed", {}).get("backend", "none"))


def build_humaneval_eval_command(
    *,
    model_dir: Path,
    output_dir: Path,
    dataset: str,
    bench_root: str,
    sandbox_urls: Sequence[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    max_concurrency: int,
    skip_sandbox_smoke: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_code_bench",
        "--model",
        str(model_dir),
        "--backend",
        "hf",
        "--device",
        "cuda:0",
        "--attn-implementation",
        "flash_attention_2",
        "--dtype",
        "bfloat16",
        "--batch-size",
        str(batch_size),
        "--dataset",
        str(dataset),
        "--bench-root",
        bench_root,
        "--output",
        str(output_dir),
        "--prompt-style",
        "plain_dialog",
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--n",
        "1",
        "--pass-at",
        "1",
        "--timeout",
        str(timeout),
        "--max-concurrency",
        str(max_concurrency),
    ]
    for sandbox_url in sandbox_urls:
        cmd.extend(["--sandbox-url", str(sandbox_url)])
    if skip_sandbox_smoke:
        cmd.append("--skip-sandbox-smoke")
    return cmd


def should_run_eval(step: int, *, total_steps: int, eval_interval: int) -> bool:
    if int(step) >= int(total_steps):
        return True
    return bool(eval_interval > 0 and int(step) % int(eval_interval) == 0)


def dataset_pass_at_1(summary: dict[str, Any], dataset: str) -> float | None:
    dataset_summary = summary.get("datasets", {}).get(dataset)
    if not isinstance(dataset_summary, dict) or dataset_summary.get("pass@1") is None:
        return None
    return float(dataset_summary["pass@1"])


def preserve_base_checkpoint(out_dir: Path, *, init_from: str) -> Path | None:
    target = out_dir / "base_model_latest.pt"
    if target.exists():
        return target
    try:
        source = resolve_resume_path(init_from)
    except FileNotFoundError:
        return None
    shutil.copy2(source, target)
    (out_dir / "base_model_source.txt").write_text(str(source) + "\n", encoding="utf-8")
    return target


def load_best_state(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "best_humaneval.json"
    if not path.exists():
        return {"pass@1": None}
    return json.loads(path.read_text(encoding="utf-8"))


def update_best_humaneval_checkpoint(out_dir: Path, *, metric: float, summary: dict[str, Any]) -> dict[str, Any]:
    latest = resolve_resume_path(out_dir / "latest.pt")
    payload = load_walkie_checkpoint(latest, map_location="cpu", strict_arch=False)
    payload["best_metric"] = float(metric)
    extra = payload.setdefault("extra", {})
    if isinstance(extra, dict):
        extra["humaneval_best"] = {"pass@1": float(metric), "step": int(summary["step"])}
    tmp = out_dir / "best_humaneval.pt.tmp"
    best_humaneval = out_dir / "best_humaneval.pt"
    torch.save(payload, tmp)
    os.replace(tmp, best_humaneval)
    shutil.copy2(best_humaneval, out_dir / "best.pt")
    state = {"pass@1": float(metric), "step": int(summary["step"]), "summary": summary}
    (out_dir / "best_humaneval.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


if __name__ == "__main__":
    raise SystemExit(main())