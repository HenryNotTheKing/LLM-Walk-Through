"""vLLM generation helpers for code evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalSamplingConfig:
    n: int = 1
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


def generate_with_vllm(
    model_path: str,
    prompts: list[str],
    *,
    sampling: EvalSamplingConfig,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
) -> list[list[str]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - Linux/CUDA optional dependency
        raise RuntimeError("vLLM evaluation requires `uv sync --extra posttrain` on Linux/CUDA") from exc

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    params = SamplingParams(
        n=sampling.n,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        stop=sampling.stop or None,
        seed=sampling.seed,
    )
    outputs = llm.generate(prompts, params)
    return [[completion.text for completion in output.outputs] for output in outputs]
