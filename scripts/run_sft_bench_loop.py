"""Segmented SFT training loop with HF code-bench evaluation.

The loop intentionally runs training and evaluation as separate subprocesses so a
single GPU can be fully released between SFT and HF generation.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmented Walkie SFT with bench evaluation after each segment")
    parser.add_argument("--config", default="configs/train/sft_walkie_kodcode_bench.yaml")
    parser.add_argument("--init-from", default="runs/walkie_code_0.5b/latest.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--eval-gpu", default=None)
    parser.add_argument("--segment-steps", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--full-eval-interval", type=int, default=None)
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--full-limit", type=int, default=None)
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
    segment_steps = int(args.segment_steps or loop_cfg.get("segment_steps", 1000))
    full_eval_interval = int(args.full_eval_interval if args.full_eval_interval is not None else loop_cfg.get("full_eval_interval", 2000))
    smoke_limit = int(args.smoke_limit if args.smoke_limit is not None else loop_cfg.get("smoke_limit", 64))
    full_limit = args.full_limit if args.full_limit is not None else loop_cfg.get("full_limit", None)
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
    history_path = out_dir / "bench_history.jsonl"
    best_state = load_best_state(out_dir)
    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval_env = os.environ.copy()
    eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu if args.eval_gpu is not None else args.gpu)
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
        run_command(train_cmd, env=train_env, dry_run=args.dry_run)
        if args.dry_run:
            break

        completed_step = checkpoint_step(out_dir)
        if completed_step < stop_step:
            raise RuntimeError(f"training stopped at step={completed_step}, expected at least {stop_step}")
        if args.no_eval:
            continue

        is_full_eval = should_run_full_eval(completed_step, total_steps=total_steps, full_eval_interval=full_eval_interval)
        eval_limit = full_limit if is_full_eval else smoke_limit
        eval_tag = "full" if is_full_eval else f"limit{eval_limit}"
        export_dir = out_dir / "hf_exports" / f"step_{completed_step:08d}"
        eval_output = out_dir / "bench_eval" / f"step_{completed_step:08d}_{eval_tag}"

        export_cmd = build_export_command(out_dir / "latest.pt", export_dir, tokenizer_path=str(cfg.data.tokenizer_path))
        run_command(export_cmd, env=eval_env, dry_run=args.dry_run)
        eval_cmd = build_eval_command(
            model_dir=export_dir,
            output_dir=eval_output,
            bench_root=bench_root,
            sandbox_urls=sandbox_urls,
            limit=eval_limit,
            batch_size=eval_batch_size,
            max_tokens=eval_max_tokens,
            temperature=eval_temperature,
            top_p=eval_top_p,
            timeout=eval_timeout,
            max_concurrency=eval_max_concurrency,
            skip_sandbox_smoke=args.skip_sandbox_smoke,
        )
        eval_started = time.time()
        run_command(eval_cmd, env=eval_env, dry_run=args.dry_run)
        eval_seconds = time.time() - eval_started

        summary = parse_bench_summary(eval_output / "summary.json")
        macro = macro_pass_at_1(summary)
        row = {
            "step": completed_step,
            "macro_pass@1": macro,
            "is_full_eval": is_full_eval,
            "limit": eval_limit,
            "eval_seconds": eval_seconds,
            "export_dir": str(export_dir),
            "output_dir": str(eval_output),
            "datasets": summary.get("datasets", {}),
        }
        append_jsonl(history_path, row)
        run_id = checkpoint_swanlab_run_id(out_dir / "latest.pt")
        log_bench_to_swanlab(cfg.train, run_id=run_id, row=row, out_dir=out_dir)
        if macro is not None and (best_state.get("macro_pass@1") is None or macro > float(best_state["macro_pass@1"])):
            best_state = update_best_checkpoint(out_dir, metric=macro, summary=row)

    return 0


def build_train_command(
    *,
    config: str,
    out_dir: Path,
    init_from: str,
    total_steps: int,
    stop_step: int,
    train_overrides: Sequence[str] | None = None,
    nproc_per_node: int = 1,
) -> list[str]:
    if int(nproc_per_node) > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={int(nproc_per_node)}",
            "-m",
            "train.walkie_sft",
            "--config",
            str(config),
        ]
    else:
        cmd = [sys.executable, "-m", "train.walkie_sft", "--config", str(config)]
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
    return len([item for item in str(gpu_arg).split(",") if item.strip()])


def effective_distributed_backend(cfg, train_overrides: Sequence[str]) -> str:
    backend = str(cfg.get("distributed", {}).get("backend", "none"))
    for override in train_overrides:
        if str(override).startswith("distributed.backend="):
            backend = str(override).split("=", 1)[1]
    return backend


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


def choose_checkpoint_args(out_dir: Path, *, init_from: str) -> list[str]:
    if (out_dir / "latest.pt").exists():
        return ["--resume", str(out_dir)]
    return ["--init-from", str(init_from)]


def build_export_command(checkpoint: Path, output_dir: Path, *, tokenizer_path: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.export_walkie_to_hf",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--tokenizer",
        tokenizer_path,
    ]


def build_eval_command(
    *,
    model_dir: Path,
    output_dir: Path,
    bench_root: str,
    sandbox_urls: Sequence[str],
    limit: int | None,
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
        "bf16",
        "--batch-size",
        str(batch_size),
        "--dataset",
        "all",
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
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if skip_sandbox_smoke:
        cmd.append("--skip-sandbox-smoke")
    return cmd


def run_command(cmd: Sequence[str], *, env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[sft-loop] $ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=REPO_ROOT, env=env, check=True)


def checkpoint_step(out_dir: Path) -> int:
    latest = out_dir / "latest.pt"
    if not latest.exists():
        return 0
    payload = load_walkie_checkpoint(latest, map_location="cpu", strict_arch=False)
    return int(payload.get("step", 0))


def checkpoint_swanlab_run_id(checkpoint: Path) -> str | None:
    if not checkpoint.exists():
        return None
    payload = load_walkie_checkpoint(checkpoint, map_location="cpu", strict_arch=False)
    extra = payload.get("extra", {})
    run_id = extra.get("swanlab_run_id") if isinstance(extra, dict) else None
    return str(run_id) if run_id is not None else None


def next_stop_step(current_step: int, *, total_steps: int, segment_steps: int) -> int:
    if segment_steps <= 0:
        raise ValueError("segment_steps must be positive")
    return min(int(total_steps), int(current_step) + int(segment_steps))


def should_run_full_eval(step: int, *, total_steps: int, full_eval_interval: int) -> bool:
    if int(step) >= int(total_steps):
        return True
    return bool(full_eval_interval > 0 and step % full_eval_interval == 0)


def parse_bench_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"bench summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"invalid bench summary: {path}")
    return payload


def macro_pass_at_1(summary: dict[str, Any]) -> float | None:
    values: list[float] = []
    for dataset_summary in summary.get("datasets", {}).values():
        if isinstance(dataset_summary, dict) and dataset_summary.get("pass@1") is not None:
            values.append(float(dataset_summary["pass@1"]))
    if not values:
        return None
    return sum(values) / len(values)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_best_state(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "best_bench.json"
    if not path.exists():
        return {"macro_pass@1": None}
    return json.loads(path.read_text(encoding="utf-8"))


def update_best_checkpoint(out_dir: Path, *, metric: float, summary: dict[str, Any]) -> dict[str, Any]:
    latest = resolve_resume_path(out_dir / "latest.pt")
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    payload["best_metric"] = float(metric)
    extra = payload.setdefault("extra", {})
    if isinstance(extra, dict):
        extra["bench_best"] = {"macro_pass@1": float(metric), "step": int(summary["step"])}
    tmp = out_dir / "best_bench.pt.tmp"
    best_bench = out_dir / "best_bench.pt"
    torch.save(payload, tmp)
    os.replace(tmp, best_bench)
    shutil.copy2(best_bench, out_dir / "best.pt")
    state = {"macro_pass@1": float(metric), "step": int(summary["step"]), "summary": summary}
    (out_dir / "best_bench.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def log_bench_to_swanlab(train_cfg, *, run_id: str | None, row: dict[str, Any], out_dir: Path) -> None:
    swanlab_cfg = train_cfg.get("swanlab")
    if not swanlab_cfg or not bool(swanlab_cfg.get("enabled", False)) or run_id is None:
        return
    mode = str(swanlab_cfg.get("mode", "online"))
    if mode == "disabled":
        return
    try:
        import swanlab
    except ImportError:
        return
    run = swanlab.init(
        project=str(swanlab_cfg.get("project", "walkie")),
        workspace=swanlab_cfg.get("workspace") or swanlab_cfg.get("entity"),
        experiment_name=swanlab_cfg.get("experiment_name") or swanlab_cfg.get("name"),
        tags=list(swanlab_cfg.get("tags", [])),
        log_dir=str(out_dir),
        mode=mode,
        id=str(run_id),
        resume="allow",
    )
    payload: dict[str, Any] = {
        "bench/macro_pass@1": row.get("macro_pass@1"),
        "bench/is_full_eval": int(bool(row.get("is_full_eval"))),
        "bench/limit": -1 if row.get("limit") is None else int(row["limit"]),
        "bench/eval_seconds": float(row.get("eval_seconds", 0.0)),
    }
    for dataset, dataset_summary in row.get("datasets", {}).items():
        if isinstance(dataset_summary, dict) and dataset_summary.get("pass@1") is not None:
            payload[f"bench/{dataset}/pass@1"] = float(dataset_summary["pass@1"])
    run.log(payload, step=int(row["step"]))
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())