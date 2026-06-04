"""Ray-accelerated sandbox execution for code evaluation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from .humaneval import CodeEvalCandidate


def evaluate_candidates_ray(
    candidates: Sequence[CodeEvalCandidate],
    *,
    sandbox_urls: Sequence[str],
    timeout: float = 10.0,
    num_workers: int | None = None,
) -> list[dict]:
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Ray evaluation requires `uv sync --extra posttrain` on Linux") from exc

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_workers)
    futures = [
        _evaluate_one_remote.remote(asdict(candidate), list(sandbox_urls), float(timeout), index)
        for index, candidate in enumerate(candidates)
    ]
    return list(ray.get(futures))


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    import requests

    response = requests.post(url, json=payload, timeout=timeout + 2.0)
    response.raise_for_status()
    return response.json()


try:
    import ray

    @ray.remote
    def _evaluate_one_remote(candidate: dict, sandbox_urls: list[str], timeout: float, index: int) -> dict:
        import uuid

        base_url = sandbox_urls[index % len(sandbox_urls)].rstrip("/")
        session_id = str(uuid.uuid4())
        try:
            payload = _post_json(
                f"{base_url}/run_jupyter",
                {"session_id": session_id, "code": candidate["test_program"], "timeout": timeout},
                timeout,
            )
            output = payload.get("output", {}) if isinstance(payload.get("output", {}), dict) else {}
            stdout = str(output.get("stdout", "") or "")
            stderr = str(output.get("stderr", "") or "")
            status = str(payload.get("status", "error"))
            return {
                **candidate,
                "passed": status == "success" and "ALL TESTS PASSED" in stdout,
                "status": status,
                "stdout": stdout,
                "stderr": stderr,
                "result": output.get("result"),
                "execution_time": float(payload.get("execution_time", 0.0) or 0.0),
            }
        except Exception as exc:
            return {
                **candidate,
                "passed": False,
                "status": "error",
                "stdout": "",
                "stderr": str(exc),
                "result": None,
                "execution_time": 0.0,
            }
        finally:
            try:
                _post_json(f"{base_url}/clear_session", {"session_id": session_id}, timeout)
            except Exception:
                pass
except ImportError:  # pragma: no cover - ray optional at import time
    def _evaluate_one_remote(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Ray is not installed")
