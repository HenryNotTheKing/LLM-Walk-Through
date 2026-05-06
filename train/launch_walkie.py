"""跨平台 Walkie 训练启动器：包装 ``torchrun``。

用法示例：

    # 单机 8 卡
    python -m train.launch_walkie --nproc 8 \
        --config configs/train/pretrain_walkie.yaml

    # 单机单卡 + 覆写 step 数
    python -m train.launch_walkie --nproc 1 \
        --config configs/train/pretrain_walkie_tiny.yaml \
        train.total_steps=20

    # 恢复
    python -m train.launch_walkie --nproc 8 \
        --config configs/train/pretrain_walkie.yaml \
        --resume runs/walkie_code_1b/latest.pt
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--nproc", type=int, default=1, help="每节点进程数（GPU 数）")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29500")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="ddp",
        help="distributed.backend 覆写值（设为 none 则单进程跑）",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist 覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inner_args: list[str] = ["--config", args.config]
    if args.resume:
        inner_args += ["--resume", args.resume]
    if args.init_from:
        inner_args += ["--init-from", args.init_from]
    inner_args += [f"distributed.backend={args.backend}"] + list(args.overrides)

    if args.nproc <= 1 and args.nnodes == 1:
        # 直接 python -m，无需 torchrun
        cmd = [sys.executable, "-m", "train.walkie_pretrain", *inner_args]
    else:
        torchrun = shutil.which("torchrun") or shutil.which("torchrun.exe")
        if torchrun is None:
            # 退化为 python -m torch.distributed.run
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={args.nproc}",
                f"--nnodes={args.nnodes}",
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
                "-m",
                "train.walkie_pretrain",
                *inner_args,
            ]
        else:
            cmd = [
                torchrun,
                f"--nproc_per_node={args.nproc}",
                f"--nnodes={args.nnodes}",
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
                "-m",
                "train.walkie_pretrain",
                *inner_args,
            ]

    print("[launch_walkie] " + " ".join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
