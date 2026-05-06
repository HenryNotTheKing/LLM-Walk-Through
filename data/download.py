"""Hugging Face 资源下载工具。

从 Hugging Face Hub 下载数据集、模型权重、tokenizer 等资源到本地。

公开接口：
- ``download(repo_id, local_dir, ...)``：从 HF 下载指定仓库内容到 ``local_dir/hf_snapshot/``。
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def _resolve_subset_shards(
    repo_id: str, subset_name: str, token: str | None = None
) -> tuple[str, list[str]]:
    """探测 ``subset_name`` 在远端的实际路径前缀，并返回排序后的文件列表。

    返回 ``(matched_prefix, shard_files)``。``shard_files`` 中的路径是相对于 repo 根的，
    可直接喂给 :func:`huggingface_hub.snapshot_download` 的 ``allow_patterns``。
    """
    all_files = list_repo_files(repo_id, repo_type="dataset", token=token)
    data_files = [f for f in all_files if not f.startswith(".")]

    possible_prefixes = [f"data/{subset_name}/", f"{subset_name}/"]
    alt_name = subset_name.replace("-", "/")
    if alt_name != subset_name:
        possible_prefixes.extend([f"data/{alt_name}/", f"{alt_name}/"])

    for prefix in possible_prefixes:
        matches = sorted(
            f for f in data_files if f.startswith(prefix) and f.endswith(".parquet")
        )
        if matches:
            return prefix, matches

    raise ValueError(
        f"在 {repo_id} 中未找到 subset_name={subset_name!r} 对应的 parquet 文件。"
        f"尝试了以下路径前缀: {possible_prefixes}\n"
        f"请检查 subset_name 拼写，或通过 allow_patterns 直接指定文件路径。"
    )


def download(
    repo_id: str,
    local_dir: str | Path,
    *,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    subset_name: str | None = None,
    num_shards: int | None = None,
    hf_endpoint: str | None = None,
    token: str | None = None,
    max_workers: int = 8,
) -> Path:
    """从 Hugging Face Hub 下载资源到 ``local_dir/hf_snapshot/``。

    Args:
        repo_id: HF 仓库 ID，如 ``"HuggingFaceFW/fineweb-edu"``。
        local_dir: 本地保存根目录。下载内容会放在 ``local_dir/hf_snapshot/`` 下。
        repo_type: 仓库类型。``"dataset"`` | ``"model"`` | ``"space"``。
        allow_patterns: 显式文件过滤规则，优先级高于 ``subset_name`` 自动推导。
        ignore_patterns: 排除规则。
        subset_name: 子集/配置名称（仅 ``repo_type="dataset"`` 时有效）。
            自动探测 ``data/{subset_name}/`` 等前缀并列出 shard。
        num_shards: 仅 ``repo_type="dataset"`` 且 ``subset_name`` 已指定时有效。
            只保证本地存在 N 个 shard——若已有 M 个，则只下载剩余 ``max(0, N - M)`` 个；
            多余的 shard 不会被删除。
        hf_endpoint: 自定义 HF 端点，如 ``"https://hf-mirror.com"``。
        token: HF access token。
        max_workers: 下载并发数。

    Returns:
        本地 snapshot 目录 ``local_dir/hf_snapshot/`` 的 :class:`~pathlib.Path`。

    调用示例：
        .. code-block:: python

            from data.download import download

            # 下载完整数据集
            download("HuggingFaceFW/fineweb-edu", "data/cache/fineweb")

            # 下载指定子集的前 5 个 shard
            download(
                "HuggingFaceFW/fineweb-edu",
                "data/cache/fineweb",
                subset_name="sample-10BT",
                num_shards=5,
            )

            # 下载模型权重（仅关键文件）
            download(
                "openai-community/gpt2",
                "models/gpt2",
                repo_type="model",
                allow_patterns=["pytorch_model.bin", "config.json", "tokenizer.json"],
            )
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = local_dir / "hf_snapshot"

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    skip_download = False

    # num_shards 仅在 dataset + subset_name 时生效
    if (
        repo_type == "dataset"
        and allow_patterns is None
        and subset_name is not None
        and num_shards is not None
    ):
        if snapshot_dir.exists():
            local_parquet_count = sum(1 for _ in snapshot_dir.rglob("*.parquet"))
            if local_parquet_count >= num_shards:
                print(
                    f"[download] 本地已有 {local_parquet_count} 个 parquet shard "
                    f">= 目标 {num_shards}，跳过远程探测与下载"
                )
                skip_download = True

    if not skip_download:
        print(
            f"[download] 准备 HF {repo_type} {repo_id} "
            f"(subset={subset_name!r}) -> {snapshot_dir}"
        )

        if allow_patterns is None and repo_type == "dataset" and subset_name is not None:
            matched_prefix, shard_files = _resolve_subset_shards(
                repo_id, subset_name, token=token
            )

            if num_shards is not None:
                locally_present = [
                    s for s in shard_files if (snapshot_dir / s).exists()
                ]
                if len(locally_present) >= num_shards:
                    print(
                        f"[download] 路径前缀 {matched_prefix} 下已有 "
                        f"{len(locally_present)} 个 shard >= 目标 {num_shards}，跳过下载"
                    )
                    skip_download = True
                else:
                    remaining = num_shards - len(locally_present)
                    candidates = [s for s in shard_files if s not in set(locally_present)]
                    to_download = candidates[:remaining]
                    print(
                        f"[download] 路径前缀 {matched_prefix} 下已有 "
                        f"{len(locally_present)} 个 shard，"
                        f"还需下载 {len(to_download)} 个: "
                        f"{[Path(p).name for p in to_download]}"
                    )
                    allow_patterns = to_download + ["README.md"]
            else:
                print(
                    f"[download] 检测到数据路径前缀: {matched_prefix}（下载全部 shard）"
                )
                allow_patterns = [
                    f"{matched_prefix}*.parquet",
                    f"{matched_prefix}*.jsonl",
                    "README.md",
                ]

    if not skip_download:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=snapshot_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=max_workers,
                token=token,
            )
        )

    if repo_type == "dataset":
        parquet_count = sum(1 for _ in snapshot_dir.rglob("*.parquet"))
        if parquet_count == 0:
            raise RuntimeError(
                f"下载结束后 {snapshot_dir} 下仍未找到任何 parquet 文件。"
                f" allow_patterns={allow_patterns}"
            )
        print(f"[download] 完成：本地 {parquet_count} 个 parquet shard")
    else:
        print(f"[download] 完成：{snapshot_dir}")

    return snapshot_dir