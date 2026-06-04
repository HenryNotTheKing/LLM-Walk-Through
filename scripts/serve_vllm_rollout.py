"""Persistent vLLM rollout server for Walkie RL."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a persistent vLLM rollout engine")
    parser.add_argument("--model", default=None, help="Initial HF/vLLM model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


class RolloutServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.llm: Any | None = None
        self.model_path: str | None = None

    def load(self, model_path: str) -> dict[str, Any]:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        self.llm = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": str(model_dir),
            "tensor_parallel_size": int(self.args.tensor_parallel_size),
            "dtype": str(self.args.dtype),
            "gpu_memory_utilization": float(self.args.gpu_memory_utilization),
            "enforce_eager": bool(self.args.enforce_eager),
        }
        if self.args.max_model_len is not None:
            kwargs["max_model_len"] = int(self.args.max_model_len)
        self.llm = LLM(**kwargs)
        self.model_path = str(model_dir)
        return {"ready": True, "model_path": self.model_path}

    def generate(self, prompts: list[str], sampling: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("vLLM model is not loaded; call /reload first")
        from vllm import SamplingParams

        params = SamplingParams(
            n=int(sampling.get("num_generations", 1)),
            temperature=float(sampling.get("temperature", 0.8)),
            top_p=float(sampling.get("top_p", 0.95)),
            max_tokens=int(sampling.get("max_tokens", 512)),
            stop=list(sampling.get("stop") or []) or None,
            seed=sampling.get("seed"),
        )
        raw_outputs = self.llm.generate(prompts, params)
        outputs: list[dict[str, Any]] = []
        for prompt_index, request_output in enumerate(raw_outputs):
            for generation_index, completion in enumerate(request_output.outputs):
                outputs.append(
                    {
                        "prompt_index": prompt_index,
                        "generation_index": generation_index,
                        "text": completion.text,
                        "metadata": {"finish_reason": getattr(completion, "finish_reason", None)},
                    }
                )
        return {"model_path": self.model_path, "outputs": outputs}


def main() -> int:
    args = parse_args()
    try:
        from fastapi import Body, FastAPI, HTTPException
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("serve_vllm_rollout requires fastapi and uvicorn") from exc

    state = RolloutServerState(args)
    app = FastAPI(title="Walkie vLLM rollout server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ready": state.llm is not None, "model_path": state.model_path}

    @app.post("/reload")
    def reload_model(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            model_path = payload.get("model_path")
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("model_path must be a non-empty string")
            return state.load(model_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate")
    def generate(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            prompts = payload.get("prompts")
            sampling = payload.get("sampling", {})
            if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("prompts must be a list of strings")
            if not isinstance(sampling, dict):
                raise ValueError("sampling must be an object")
            return state.generate(prompts, sampling)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if args.model is not None:
        state.load(args.model)
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
