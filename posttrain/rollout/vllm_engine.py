"""vLLM rollout backend.

The import is lazy so Windows development and CPU tests do not need vLLM
installed. Real training should run on Linux/CUDA with `uv sync --extra posttrain`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from collections.abc import Sequence
from typing import Any

from .base import RolloutOutput, SamplingConfig


class VLLMRolloutEngine:
    def __init__(self, model_path: str, *, tensor_parallel_size: int = 1, dtype: str = "auto") -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - optional Linux dependency
            raise RuntimeError("vLLM rollout requires the posttrain extra on Linux/CUDA") from exc
        self.model_path = model_path
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, dtype=dtype)

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[RolloutOutput]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=sampling.num_generations,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            stop=sampling.stop or None,
            seed=sampling.seed,
        )
        raw_outputs = self.llm.generate(prompts, params)
        outputs: list[RolloutOutput] = []
        for prompt_index, request_output in enumerate(raw_outputs):
            for generation_index, completion in enumerate(request_output.outputs):
                outputs.append(
                    RolloutOutput(
                        prompt=prompts[prompt_index],
                        response=completion.text,
                        prompt_index=prompt_index,
                        generation_index=generation_index,
                        metadata={"finish_reason": getattr(completion, "finish_reason", None)},
                    )
                )
        return outputs


class RemoteVLLMRolloutEngine:
    def __init__(
        self,
        server_url: str | Sequence[str],
        *,
        request_timeout: float = 120.0,
        reload_timeout: float = 300.0,
        max_retries: int = 2,
        request_shards: int = 1,
        max_concurrent_requests: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.server_urls = self._normalize_server_urls(server_url)
        self.server_url = self.server_urls[0]
        self.request_timeout = float(request_timeout)
        self.reload_timeout = float(reload_timeout)
        self.max_retries = int(max_retries)
        self.request_shards = max(1, int(request_shards))
        if max_concurrent_requests is None:
            max_concurrent_requests = max(len(self.server_urls), self.request_shards)
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional posttrain dependency
                raise RuntimeError("remote vLLM rollout requires httpx from the posttrain extra") from exc
            self.clients = [httpx.Client() for _ in self.server_urls]
        else:
            self.clients = [client for _ in self.server_urls]
        self.client = self.clients[0]
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests) if self.max_concurrent_requests > 1 else None

    @staticmethod
    def _normalize_server_urls(server_url: str | Sequence[str]) -> list[str]:
        if isinstance(server_url, str):
            urls = [item.strip() for item in server_url.split(",")]
        else:
            urls = [str(item).strip() for item in server_url]
        urls = [url.rstrip("/") for url in urls if url]
        if not urls:
            raise ValueError("remote vLLM rollout requires at least one server URL")
        return urls

    def health(self) -> dict[str, Any]:
        response = self.client.get(f"{self.server_url}/health", timeout=self.request_timeout)
        response.raise_for_status()
        return dict(response.json())

    def reload(self, model_path: str) -> dict[str, Any]:
        payload = {"model_path": str(model_path)}
        if len(self.server_urls) == 1:
            return self._post_to(0, "/reload", payload, timeout=self.reload_timeout)
        futures = [self._submit(self._post_to, index, "/reload", payload, self.reload_timeout) for index in range(len(self.server_urls))]
        replicas = [future.result() for future in futures]
        return {"ready": all(bool(item.get("ready", False)) for item in replicas), "model_path": str(model_path), "replicas": replicas}

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[RolloutOutput]:
        if not prompts:
            return []
        shard_count = min(len(prompts), max(self.request_shards, len(self.server_urls)))
        if shard_count == 1:
            data = self._generate_shard(0, 0, prompts, sampling)
            return self._decode_outputs(data, prompts, prompt_offset=0)

        prompt_shards = self._split_prompts(prompts, shard_count)
        futures = []
        for shard_index, (prompt_offset, shard_prompts) in enumerate(prompt_shards):
            client_index = shard_index % len(self.server_urls)
            futures.append((prompt_offset, shard_prompts, self._submit(self._generate_shard, client_index, prompt_offset, shard_prompts, sampling)))
        outputs: list[RolloutOutput] = []
        for prompt_offset, shard_prompts, future in futures:
            outputs.extend(self._decode_outputs(future.result(), shard_prompts, prompt_offset=prompt_offset))
        return outputs

    def _generate_shard(self, client_index: int, prompt_offset: int, prompts: list[str], sampling: SamplingConfig) -> dict[str, Any]:
        payload = {
            "prompts": prompts,
            "sampling": {
                "num_generations": int(sampling.num_generations),
                "temperature": float(sampling.temperature),
                "top_p": float(sampling.top_p),
                "max_tokens": int(sampling.max_tokens),
                "stop": list(sampling.stop or []),
                "seed": sampling.seed,
            },
        }
        return self._post_to(client_index, "/generate", payload, timeout=self.request_timeout)

    @staticmethod
    def _split_prompts(prompts: list[str], shard_count: int) -> list[tuple[int, list[str]]]:
        shards: list[tuple[int, list[str]]] = []
        for shard_index in range(shard_count):
            start = shard_index * len(prompts) // shard_count
            end = (shard_index + 1) * len(prompts) // shard_count
            if start < end:
                shards.append((start, prompts[start:end]))
        return shards

    def _decode_outputs(self, data: dict[str, Any], prompts: list[str], *, prompt_offset: int) -> list[RolloutOutput]:
        outputs: list[RolloutOutput] = []
        for item in data.get("outputs", []):
            local_prompt_index = int(item["prompt_index"])
            prompt_index = prompt_offset + local_prompt_index
            outputs.append(
                RolloutOutput(
                    prompt=prompts[local_prompt_index],
                    response=str(item.get("text", "")),
                    prompt_index=prompt_index,
                    generation_index=int(item["generation_index"]),
                    metadata={
                        key: value
                        for key, value in dict(item.get("metadata", {})).items()
                        if value is not None
                    },
                )
            )
        return outputs

    def _post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        return self._post_to(0, path, payload, timeout=timeout)

    def _post_to(self, client_index: int, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        last_error: Exception | None = None
        client = self.clients[client_index]
        server_url = self.server_urls[client_index]
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(f"{server_url}{path}", json=payload, timeout=timeout)
                response.raise_for_status()
                return dict(response.json())
            except Exception as exc:  # pragma: no cover - exercised with real HTTP failures
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(1.0 * (attempt + 1), 5.0))
        assert last_error is not None
        raise RuntimeError(f"remote vLLM request failed: {server_url}{path}") from last_error

    def _submit(self, fn, *args):
        if self.executor is None:
            class ImmediateFuture:
                def __init__(self, value) -> None:
                    self.value = value

                def result(self):
                    return self.value

            return ImmediateFuture(fn(*args))
        return self.executor.submit(fn, *args)