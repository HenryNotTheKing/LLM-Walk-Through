"""Evaluate a custom code-generation JSONL dataset with vLLM + sandbox."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from posttrain.eval.humaneval import (
    build_candidates,
    evaluate_candidates_async,
    load_custom_jsonl,
    summarize_pass_at_k,
    write_jsonl,
)
from posttrain.eval.ray_sandbox import evaluate_candidates_ray
from posttrain.eval.vllm_runner import EvalSamplingConfig, generate_with_vllm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sandbox-url", action="append", default=None)
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--test-field", default=None)
    parser.add_argument("--task-id-field", default=None)
    parser.add_argument("--entry-point-field", default=None)
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument("--ray-workers", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--pass-at", default=None)
    return parser.parse_args()


def main() -> None:
    args = _merge_config(parse_args())
    samples = load_custom_jsonl(
        args.data,
        prompt_field=args.prompt_field,
        test_field=args.test_field,
        task_id_field=args.task_id_field,
        entry_point_field=args.entry_point_field,
    )
    completions = generate_with_vllm(
        args.model,
        [sample.prompt for sample in samples],
        sampling=EvalSamplingConfig(n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    candidates = build_candidates(samples, completions)
    if args.use_ray:
        rows = evaluate_candidates_ray(candidates, sandbox_urls=args.sandbox_url, timeout=args.timeout, num_workers=args.ray_workers)
    else:
        rows = asyncio.run(evaluate_candidates_async(candidates, sandbox_urls=args.sandbox_url, timeout=args.timeout, max_concurrency=args.max_concurrency))
    write_jsonl(args.output, rows)
    ks = [int(item) for item in args.pass_at.split(",") if item.strip()]
    summary = summarize_pass_at_k(rows, ks=ks)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    config: dict[str, Any] = {}
    if args.config is not None:
        config = dict(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True))
    defaults = {
        "prompt_field": "prompt",
        "test_field": "test",
        "task_id_field": "task_id",
        "entry_point_field": "entry_point",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "n": 1,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 512,
        "timeout": 10.0,
        "max_concurrency": 32,
        "pass_at": "1",
    }
    for attr, default in defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, config.get(attr, default))
    for attr in ("model", "data", "output", "ray_workers"):
        if getattr(args, attr) is None:
            setattr(args, attr, config.get(attr))
    if args.sandbox_url is None:
        args.sandbox_url = config.get("sandbox_urls")
    if isinstance(args.pass_at, list):
        args.pass_at = ",".join(str(item) for item in args.pass_at)
    args.use_ray = bool(args.use_ray or config.get("use_ray", False))
    missing = [name for name in ("model", "data", "output", "sandbox_url") if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing required arguments/config keys: {', '.join(missing)}")
    return args


if __name__ == "__main__":
    main()
