"""DDP 工具：从环境变量解析 rank/world_size，初始化 process group。

约定：
    - 单进程脚本无需调用本模块；多进程通过 ``torchrun`` 启动。
    - ``setup_distributed`` 是幂等的：未启动多进程时返回单进程占位。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = "none") -> DistInfo:
    """根据配置和环境变量决定是否初始化 DDP。"""
    if backend == "none" or "RANK" not in os.environ:
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist_backend = "nccl"
    else:
        dist_backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=dist_backend, init_method="env://")

    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
