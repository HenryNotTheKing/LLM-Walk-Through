from __future__ import annotations

from posttrain.rollout.base import SamplingConfig
from posttrain.rollout.vllm_engine import RemoteVLLMRolloutEngine


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict, float]] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        return FakeResponse({"ready": True, "model_path": "model-a"})

    def post(self, url: str, *, json: dict, timeout: float) -> FakeResponse:
        self.posts.append((url, json, timeout))
        if url.endswith("/reload"):
            return FakeResponse({"ready": True, "model_path": json["model_path"]})
        return FakeResponse(
            {
                "outputs": [
                    {
                        "prompt_index": 0,
                        "generation_index": 0,
                        "text": "one",
                        "metadata": {"finish_reason": "stop"},
                    },
                    {
                        "prompt_index": 0,
                        "generation_index": 1,
                        "text": "two",
                        "metadata": {"finish_reason": None},
                    },
                ]
            }
        )


def test_remote_vllm_rollout_engine_reload_and_generate() -> None:
    client = FakeClient()
    engine = RemoteVLLMRolloutEngine(
        "http://127.0.0.1:18080/",
        request_timeout=12.0,
        reload_timeout=34.0,
        client=client,
    )

    assert engine.health()["ready"] is True
    assert engine.reload("/tmp/model")["model_path"] == "/tmp/model"

    outputs = engine.generate(
        ["prompt"],
        SamplingConfig(num_generations=2, temperature=0.7, top_p=0.9, max_tokens=16, stop=["\n"], seed=123),
    )

    assert [output.response for output in outputs] == ["one", "two"]
    assert [output.generation_index for output in outputs] == [0, 1]
    assert all(output.prompt == "prompt" for output in outputs)
    assert outputs[0].metadata["finish_reason"] == "stop"
    assert "finish_reason" not in outputs[1].metadata
    reload_call, generate_call = client.posts
    assert reload_call == ("http://127.0.0.1:18080/reload", {"model_path": "/tmp/model"}, 34.0)
    assert generate_call[0] == "http://127.0.0.1:18080/generate"
    assert generate_call[1]["sampling"]["num_generations"] == 2
    assert generate_call[1]["sampling"]["seed"] == 123


class FakeShardedClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict, float]] = []

    def post(self, url: str, *, json: dict, timeout: float) -> FakeResponse:
        self.posts.append((url, json, timeout))
        if url.endswith("/reload"):
            return FakeResponse({"ready": True, "model_path": json["model_path"]})
        return FakeResponse(
            {
                "outputs": [
                    {
                        "prompt_index": prompt_index,
                        "generation_index": 0,
                        "text": prompt,
                        "metadata": {},
                    }
                    for prompt_index, prompt in enumerate(json["prompts"])
                ]
            }
        )


def test_remote_vllm_rollout_engine_shards_generate_across_urls() -> None:
    client = FakeShardedClient()
    engine = RemoteVLLMRolloutEngine(
        ["http://127.0.0.1:18080", "http://127.0.0.1:18081"],
        request_timeout=12.0,
        reload_timeout=34.0,
        request_shards=2,
        max_concurrent_requests=2,
        client=client,
    )

    reload_result = engine.reload("/tmp/model")
    outputs = engine.generate(
        ["prompt-0", "prompt-1", "prompt-2", "prompt-3"],
        SamplingConfig(num_generations=1, max_tokens=16),
    )

    assert reload_result["ready"] is True
    assert [output.prompt_index for output in outputs] == [0, 1, 2, 3]
    assert [output.prompt for output in outputs] == ["prompt-0", "prompt-1", "prompt-2", "prompt-3"]
    generate_calls = [post for post in client.posts if post[0].endswith("/generate")]
    assert [call[0] for call in generate_calls] == ["http://127.0.0.1:18080/generate", "http://127.0.0.1:18081/generate"]
    assert [call[1]["prompts"] for call in generate_calls] == [["prompt-0", "prompt-1"], ["prompt-2", "prompt-3"]]