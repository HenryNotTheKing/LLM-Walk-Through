"""GPT-2 主体：``Block`` / ``GPT2Model`` / ``GPT2LMHeadModel``。

设计要点：
    - 通过 ``GPT2Config`` 数据类承接 OmegaConf 解出来的字段。
    - 各子模块从 ``core/`` 下对应位置导入，方便后续按消融实验整体替换。
    - ``from_pretrained_hf`` 用于加载 HuggingFace ``transformers`` 的官方 GPT-2 权重，
      做 logits 对齐 sanity check；首版仅在用户显式调用时才依赖 ``transformers``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention import CausalSelfAttention
from core.ffn import GeluMLP
from core.norm import LayerNorm
from core.position import LearnedPositionalEmbedding


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    attn_impl: str = "sdpa"
    init_std: float = 0.02

    @classmethod
    def from_dict(cls, d) -> "GPT2Config":  # accepts DictConfig or dict
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in fields})


class Block(nn.Module):
    """GPT-2 Pre-LN 残差块: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.bias,
            attn_impl=cfg.attn_impl,
        )
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = GeluMLP(cfg.n_embd, dropout=cfg.dropout, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LMHeadModel(nn.Module):
    """GPT-2 + 语言建模头。

    forward 返回 ``(logits, loss)``；当 ``targets`` 为 None 时只返回最后一步 logits 以省显存。
    """

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = LearnedPositionalEmbedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 论文：对残差路径上的投影做 1/sqrt(2*n_layer) 缩放
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ----- forward / loss -----
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"输入序列长度 {T} 超过 block_size {self.cfg.block_size}"
            )

        tok_emb = self.wte(idx)                              # (B, T, C)
        pos_emb = self.wpe(T, device=idx.device)             # (T, C)
        x = self.drop(tok_emb + pos_emb)
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        # 推理时仅算最后一步 logits 节省算力
        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    # ----- generate -----
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """简单生成接口（首版无 KV cache）。

        - ``temperature <= 0`` 退化为贪心采样。
        - ``top_k`` 与 ``top_p`` 可同时启用：先做 top-k 截断再做 top-p 截断。
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # (B, V)

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, -1, None]
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    mask = cum > top_p
                    # 至少保留 1 个 token
                    mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_idx, sorted_logits
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return idx

    # ----- HF 权重加载（用于对齐 sanity check） -----
    @classmethod
    def from_pretrained_hf(cls, model_name: str = "gpt2") -> "GPT2LMHeadModel":
        """加载 HuggingFace ``transformers`` 的 GPT-2 权重并转换到本项目模型。

        仅在调用时导入 ``transformers``，避免成为强依赖。
        """
        from transformers import GPT2LMHeadModel as HFGPT2  # 延迟导入
        from transformers import GPT2Config as HFGPT2Config

        hf_cfg = HFGPT2Config.from_pretrained(model_name)
        cfg = GPT2Config(
            vocab_size=hf_cfg.vocab_size,
            n_layer=hf_cfg.n_layer,
            n_head=hf_cfg.n_head,
            n_embd=hf_cfg.n_embd,
            block_size=hf_cfg.n_positions,
            dropout=0.0,
            bias=True,
            tie_weights=True,
            attn_impl="sdpa",
        )
        model = cls(cfg)
        hf_model = HFGPT2.from_pretrained(model_name)
        _copy_hf_gpt2_weights(model, hf_model)
        return model


def _copy_hf_gpt2_weights(dst: GPT2LMHeadModel, hf_model) -> None:
    """把 HF GPT-2 的权重拷到本项目实现。

    要点：HF GPT-2 用 ``Conv1D``，其 weight 形状为 ``(in, out)``；
    我们用 ``nn.Linear``，weight 形状为 ``(out, in)``，需要转置。
    """
    sd_hf = hf_model.state_dict()
    sd = dst.state_dict()

    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    # HF 的 key 命名: transformer.wte.weight / transformer.h.{i}.attn.c_attn.weight ...
    mapping = {}
    mapping["wte.weight"] = "transformer.wte.weight"
    mapping["wpe.weight"] = "transformer.wpe.weight"
    mapping["ln_f.weight"] = "transformer.ln_f.weight"
    mapping["ln_f.bias"] = "transformer.ln_f.bias"
    for i in range(dst.cfg.n_layer):
        for sub in [
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]:
            mapping[f"h.{i}.{sub}"] = f"transformer.h.{i}.{sub}"

    for our_key, hf_key in mapping.items():
        if our_key not in sd:
            raise KeyError(f"目标模型缺少键: {our_key}")
        if hf_key not in sd_hf:
            raise KeyError(f"HF 模型缺少键: {hf_key}")
        w = sd_hf[hf_key]
        if any(our_key.endswith(t) for t in transposed):
            w = w.t().contiguous()
        sd[our_key].copy_(w)

    # tied weights：lm_head.weight 与 wte.weight 共享，无需单独拷贝
    if not dst.cfg.tie_weights:
        sd["lm_head.weight"].copy_(sd_hf["lm_head.weight"])  # 一般用不到这条
