"""Evaluate Walkie checkpoints on local code benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from posttrain.eval.code_bench import (
    SUPPORTED_CODE_BENCH_DATASETS,
    build_code_bench_candidates,
    load_code_bench_samples,
    normalize_dataset_name,
    render_code_bench_prompt,
)
from posttrain.eval.humaneval import evaluate_candidates_async, summarize_pass_at_k, write_jsonl
from posttrain.eval.vllm_runner import EvalSamplingConfig, generate_with_vllm
from posttrain.sandbox.jupyter_client import JupyterSandboxClient


def parse_args(argv: Sequence[str] | None = None, *, default_dataset: str = "all") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local code benchmarks with model generation and sandbox execution")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, choices=[*SUPPORTED_CODE_BENCH_DATASETS, "all"])
    parser.add_argument("--bench-root", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default=None, choices=["vllm", "hf"])
    parser.add_argument("--sandbox-url", dest="sandbox_urls", action="append", default=None)
    parser.add_argument("--use-ray", dest="use_ray", action="store_true", default=None)
    parser.add_argument("--no-use-ray", dest="use_ray", action="store_false")
    parser.add_argument("--ray-workers", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--pass-at", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-style", default=None, choices=["plain_dialog", "dialog", "raw"])
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--skip-sandbox-smoke", action="store_true", default=None)
    parser.add_argument("--omit-test-programs", action="store_true", default=None)
    args = parser.parse_args(argv)
    args.default_dataset = default_dataset
    return args


def main(argv: Sequence[str] | None = None, *, default_dataset: str = "all") -> int:
    args = parse_args(argv, default_dataset=default_dataset)
    config = _load_config(args.config)

    dataset_arg = _coalesce(args, config, "dataset", args.default_dataset)
    dataset_names = list(SUPPORTED_CODE_BENCH_DATASETS) if dataset_arg == "all" else [normalize_dataset_name(str(dataset_arg))]
    data = _coalesce(args, config, "data", None)
    if data is not None and len(dataset_names) != 1:
        raise ValueError("--data can only be used when evaluating a single dataset")

    samples = []
    bench_root = _coalesce(args, config, "bench_root", "data/bench")
    limit = _coalesce(args, config, "limit", None)
    for dataset in dataset_names:
        samples.extend(load_code_bench_samples(dataset, bench_root=bench_root, data=data, limit=limit))
    if not samples:
        raise RuntimeError("no benchmark samples loaded")

    sandbox_urls = _as_list(_coalesce(args, config, "sandbox_urls", None)) or ["http://127.0.0.1:18901"]
    timeout = float(_coalesce(args, config, "timeout", 10.0))
    skip_sandbox_smoke = bool(_coalesce(args, config, "skip_sandbox_smoke", False))
    if not skip_sandbox_smoke:
        asyncio.run(_check_sandbox(sandbox_urls, timeout=timeout))

    prompt_style = str(_coalesce(args, config, "prompt_style", "plain_dialog"))
    prompts = [render_code_bench_prompt(sample, prompt_style=prompt_style) for sample in samples]
    pass_at = _parse_pass_at(_coalesce(args, config, "pass_at", [1]))
    n = _resolve_num_samples(
        int(_coalesce(args, config, "n", 1)),
        pass_at=pass_at,
    )
    sampling = EvalSamplingConfig(
        n=n,
        temperature=float(_coalesce(args, config, "temperature", 0.2)),
        top_p=float(_coalesce(args, config, "top_p", 0.95)),
        max_tokens=int(_coalesce(args, config, "max_tokens", 512)),
        stop=_as_list(_coalesce(args, config, "stop", None)),
        seed=_coalesce(args, config, "seed", None),
    )
    model_path = str(_coalesce(args, config, "model", "runs/walkie_code_0.5b_vllm_hf"))
    backend = str(_coalesce(args, config, "backend", "vllm")).lower()
    if backend == "vllm":
        completions = generate_with_vllm(
            model_path,
            prompts,
            sampling=sampling,
            tensor_parallel_size=int(_coalesce(args, config, "tensor_parallel_size", 1)),
            dtype=str(_coalesce(args, config, "dtype", "auto")),
        )
    elif backend == "hf":
        from posttrain.eval.hf_runner import generate_with_hf

        completions = generate_with_hf(
            model_path,
            prompts,
            sampling=sampling,
            batch_size=int(_coalesce(args, config, "batch_size", 4)),
            device=str(_coalesce(args, config, "device", "auto")),
            dtype=str(_coalesce(args, config, "dtype", "auto")),
            attn_implementation=str(_coalesce(args, config, "attn_implementation", "auto")),
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    candidates = build_code_bench_candidates(samples, completions, prompts=prompts)

    use_ray = bool(_coalesce(args, config, "use_ray", False))
    if use_ray:
        from posttrain.eval.ray_sandbox import evaluate_candidates_ray

        rows = evaluate_candidates_ray(
            candidates,
            sandbox_urls=sandbox_urls,
            timeout=timeout,
            num_workers=_coalesce(args, config, "ray_workers", None),
        )
    else:
        rows = asyncio.run(
            evaluate_candidates_async(
                candidates,
                sandbox_urls=sandbox_urls,
                timeout=timeout,
                max_concurrency=int(_coalesce(args, config, "max_concurrency", 32)),
            )
        )

    if bool(_coalesce(args, config, "omit_test_programs", False)):
        for row in rows:
            row.pop("test_program", None)

    summary = _write_outputs(
        rows,
        output=Path(str(_coalesce(args, config, "output", "runs/eval/walkie_code_0.5b"))),
        pass_at=pass_at,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


async def _check_sandbox(sandbox_urls: Sequence[str], *, timeout: float) -> None:
    client = JupyterSandboxClient(sandbox_urls, timeout=timeout)
    result = await client.run_code("print('ALL TESTS PASSED')")
    if result.status != "success" or "ALL TESTS PASSED" not in result.stdout:
        raise RuntimeError(f"sandbox smoke failed: status={result.status} stderr={result.stderr!r}")


def _write_outputs(rows: list[dict[str, Any]], *, output: Path, pass_at: Sequence[int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset", "unknown"))].append(row)

    file_mode = output.suffix == ".jsonl" and len(grouped) == 1
    if file_mode:
        dataset, dataset_rows = next(iter(grouped.items()))
        write_jsonl(output, dataset_rows)
        summary = _dataset_summary(dataset, dataset_rows, pass_at=pass_at)
        summary_path = output.with_name(f"{output.stem}.summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"datasets": {dataset: summary}}

    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for dataset, dataset_rows in sorted(grouped.items()):
        dataset_dir = output / dataset
        write_jsonl(dataset_dir / "results.jsonl", dataset_rows)
        summary = _dataset_summary(dataset, dataset_rows, pass_at=pass_at)
        (dataset_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries[dataset] = summary
    aggregate = {"datasets": summaries, "macro": _macro_pass_at_k(summaries, pass_at=pass_at)}
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate


def _dataset_summary(dataset: str, rows: list[dict[str, Any]], *, pass_at: Sequence[int]) -> dict[str, Any]:
    summary = summarize_pass_at_k(rows, ks=pass_at)
    sandbox_errors = sum(1 for row in rows if row.get("status") != "success")
    summary.update(
        {
            "dataset": dataset,
            "num_completions": len(rows),
            "sandbox_errors": sandbox_errors,
            "sandbox_error_rate": float(sandbox_errors / len(rows)) if rows else 0.0,
        }
    )
    return summary


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return dict(data)


def _coalesce(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _parse_pass_at(value: Any) -> list[int]:
    if value is None:
        return [1]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _resolve_num_samples(n: int, *, pass_at: Sequence[int]) -> int:
    """Ensure enough completions are generated for the requested pass@k metrics."""
    if not pass_at:
        return max(1, int(n))
    required = max(int(k) for k in pass_at)
    resolved = max(1, int(n))
    if resolved < required:
        print(f"[eval] bumping n from {resolved} to {required} to support pass@{required}")
        resolved = required
    return resolved


def _macro_pass_at_k(summaries: dict[str, dict[str, Any]], *, pass_at: Sequence[int]) -> dict[str, float]:
    macro: dict[str, float] = {}
    for k in pass_at:
        key = f"pass@{k}"
        values = [float(summary[key]) for summary in summaries.values() if key in summary]
        macro[key] = float(sum(values) / len(values)) if values else 0.0
    return macro


if __name__ == "__main__":
    raise SystemExit(main())