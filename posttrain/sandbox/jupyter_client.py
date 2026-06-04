"""Client adapter for ChenShawn/MultiModal-Jupyter-Sandbox."""

from __future__ import annotations

import asyncio
import itertools
import uuid
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class JupyterExecutionResult:
    status: str
    execution_time: float
    stdout: str
    stderr: str
    result: Any = None
    images: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success" and not self.stderr


def parse_jupyter_response(payload: dict[str, Any]) -> JupyterExecutionResult:
    output = payload.get("output", {})
    if not isinstance(output, dict):
        output = {"stderr": str(output)}
    images = output.get("images", [])
    if images is None:
        images = []
    return JupyterExecutionResult(
        status=str(payload.get("status", "error")),
        execution_time=float(payload.get("execution_time", 0.0) or 0.0),
        stdout=str(output.get("stdout", "") or ""),
        stderr=str(output.get("stderr", "") or ""),
        result=output.get("result"),
        images=[str(item) for item in images] if isinstance(images, list) else [],
    )


class JupyterSandboxClient:
    """Async HTTP client for `/run_jupyter` and `/clear_session`.

    `base_urls` may point to a port pool such as `http://127.0.0.1:18901` ...
    `http://127.0.0.1:18904`. The client imports `httpx` lazily so unit tests do
    not require post-training extras to be installed.
    """

    def __init__(
        self,
        base_urls: Iterable[str],
        *,
        timeout: float = 10.0,
        retries: int = 3,
        clear_session: bool = True,
    ) -> None:
        urls = [url.rstrip("/") for url in base_urls]
        if not urls:
            raise ValueError("base_urls must not be empty")
        self.base_urls = urls
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.clear_session_enabled = bool(clear_session)
        self._cycle = itertools.cycle(self.base_urls)

    async def run_code(self, code: str, *, session_id: str | None = None) -> JupyterExecutionResult:
        import httpx

        sid = session_id or str(uuid.uuid4())
        last_error: Exception | None = None
        for _ in range(max(1, self.retries + 1)):
            base_url = next(self._cycle)
            try:
                async with httpx.AsyncClient(timeout=self.timeout + 2.0) as client:
                    response = await client.post(
                        f"{base_url}/run_jupyter",
                        json={"session_id": sid, "code": code, "timeout": self.timeout},
                    )
                    if response.status_code == 404:
                        response = await client.post(
                            f"{base_url}/jupyter_sandbox",
                            json={"session_id": sid, "code": code, "timeout": self.timeout},
                        )
                    response.raise_for_status()
                    result = parse_jupyter_response(response.json())
                    if self.clear_session_enabled:
                        try:
                            await client.post(f"{base_url}/clear_session", json={"session_id": sid})
                        except Exception:
                            pass
                    return result
            except Exception as exc:  # pragma: no cover - network branch
                last_error = exc
                await asyncio.sleep(0)
        return JupyterExecutionResult(
            status="error",
            execution_time=0.0,
            stdout="",
            stderr=str(last_error) if last_error is not None else "sandbox request failed",
            images=[],
        )
