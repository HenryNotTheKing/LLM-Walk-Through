"""自动检测 GPU 环境并安装对应版本的 PyTorch。

用法:
    python scripts/install_torch.py          # 检测并提示安装命令
    python scripts/install_torch.py --run    # 直接执行安装
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_cuda_capability() -> tuple[bool, str | None, list[str]]:
    """检测是否有 NVIDIA GPU 及推荐的 CUDA wheel 版本。

    返回 (has_gpu, cuda_tag, warnings)。
    """
    warnings: list[str] = []

    # 检查 Python 版本 —— PyTorch Windows CUDA wheel 目前不支持 3.13+
    py_major, py_minor = sys.version_info[:2]
    if sys.platform == "win32" and (py_major, py_minor) >= (3, 13):
        warnings.append(
            f"当前 Python {py_major}.{py_minor}："
            "PyTorch Windows CUDA 版暂不支持 Python 3.13+，"
            "请使用 Python 3.10~3.12（推荐 3.12）。"
        )

    try:
        import torch

        if torch.cuda.is_available():
            return True, f"cu{torch.version.cuda.replace('.', '')}", warnings
    except ImportError:
        pass

    # 没有 torch 或 torch 无 CUDA，尝试用 nvidia-smi 推断
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        driver = result.stdout.strip()
        if driver:
            # 驱动版本与 CUDA 兼容性参考: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
            major = int(driver.split(".")[0])
            if major >= 545:
                return True, "cu124", warnings
            if major >= 525:
                return True, "cu121", warnings
            return True, "cu118", warnings
    except Exception:
        pass

    return False, None, warnings


def build_install_command(cuda_tag: str | None) -> list[str]:
    """构造 uv pip install 命令。"""
    cmd = [sys.executable, "-m", "uv", "pip", "install", "torch"]
    if cuda_tag:
        cmd += ["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-detect GPU and install matching PyTorch")
    parser.add_argument("--run", action="store_true", help="直接执行安装命令")
    args = parser.parse_args()

    has_gpu, cuda_tag, warnings = detect_cuda_capability()

    for w in warnings:
        print(f"⚠️  {w}")

    if has_gpu:
        print(f"检测到 NVIDIA GPU，推荐安装 PyTorch + {cuda_tag}")
    else:
        print("未检测到 NVIDIA GPU，将安装 CPU 版本 PyTorch")
        cuda_tag = None

    cmd = build_install_command(cuda_tag)
    print(f"\n安装命令:\n  {' '.join(cmd)}\n")

    # 检查当前已安装的 torch
    try:
        import torch

        print(f"当前 PyTorch 版本: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
    except ImportError:
        print("当前环境未安装 PyTorch")

    if args.run:
        print("开始安装...")
        _run(cmd)
        print("安装完成，请验证: python -c 'import torch; print(torch.cuda.is_available())'")
    else:
        print("提示: 加上 --run 参数可直接执行上述命令")


if __name__ == "__main__":
    main()
