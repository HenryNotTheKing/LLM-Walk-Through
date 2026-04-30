"""加载 checkpoint 并做文本生成。

示例：
    python -m scripts.generate \
        --checkpoint runs/tiny_shakespeare/ckpt.pt \
        --tokenizer data/cache/tiny_shakespeare/tokenizer.json \
        --prompt "ROMEO:" --max-new-tokens 200 --top-k 50 --top-p 0.95 --temperature 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.model import GPT2Config, GPT2LMHeadModel
from core.tokenizer import build_tokenizer, load_tokenizer
from core.utils.device import select_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="自训分词器 JSON 路径（kind 自动从文件读出）；不指定时退回到 GPT-2 官方 BPE",
    )
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = GPT2Config(**ckpt["model_cfg"])
    model = GPT2LMHeadModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.tokenizer:
        tok = load_tokenizer(args.tokenizer)
    else:
        tok = build_tokenizer("gpt2")

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    if ids.size(1) == 0:
        # 至少有一个起始 token
        ids = torch.tensor([[tok.eos_token_id]], dtype=torch.long, device=device)

    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
