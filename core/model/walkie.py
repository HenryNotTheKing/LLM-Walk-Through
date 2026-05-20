"""Walkie-Code-1B 模型主体。

特点（与 GPT-2 教学实现的差异）：
    - Pre-norm + RMSNorm（替代 LayerNorm）。
    - SwiGLU FFN（替代 GELU MLP）。
    - GQA + QK-Norm + RoPE 的因果自注意力。
    - 默认无 bias，可配置 weight tying（默认 ``True`` 让 1B 总参数量更紧凑）。
    - 显式预留 FIM/repo/file 等 special token 元数据，便于后续数据流水线接入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.attention.walkie_attention import WalkieCausalSelfAttention
from core.ffn.swiglu import SwiGLUMLP
from core.norm.rmsnorm import RMSNorm
from core.position.rope import RotaryPositionalEmbedding


def _default_special_tokens() -> dict[str, str]:
    return {
        "endoftext": "<|endoftext|>",
        "pad": "<|pad|>",
    }


@dataclass
class WalkieConfig:
    model_name: str = "Walkie-Code-1B"

    # 词表 / 上下文
    vocab_size: int = 65536
    block_size: int = 16384

    # 主体维度
    n_embd: int = 1536
    n_layer: int = 36
    n_head: int = 24
    n_head_kv: int = 8
    head_dim: int = 64
    d_ffn: int = 3840

    # 归一化 / FFN / 位置
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1.0e6
    rope_scaling_factor: float = 1.0
    qk_norm: bool = True

    # 其它训练相关
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    attn_impl: str = "sdpa"
    init_std: float = 0.02
    gradient_checkpointing: bool = False
    loss_chunk_size: int | None = 1024

    # 数据管线侧使用的 special tokens，连续 token bin 只写 endoftext。
    special_tokens: dict[str, str] = field(default_factory=_default_special_tokens)

    @classmethod
    def from_dict(cls, d) -> "WalkieConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in fields})

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class WalkieBlock(nn.Module):
    """Pre-norm 残差块: x = x + Attn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))."""

    def __init__(self, cfg: WalkieConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.attn = WalkieCausalSelfAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_head_kv=cfg.n_head_kv,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.block_size,
            dropout=cfg.dropout,
            bias=cfg.bias,
            attn_impl=cfg.attn_impl,
            qk_norm=cfg.qk_norm,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
            rms_norm_eps=cfg.rms_norm_eps,
            rope=rope,
        )
        self.norm_ffn = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(
            n_embd=cfg.n_embd,
            d_ffn=cfg.d_ffn,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_ffn(x))
        return x


class WalkieForCausalLM(nn.Module):
    """Walkie-Code-1B causal language model。

    forward 返回 ``(logits, loss)``；当 ``targets`` 为 None 时只返回最后一步 logits 以省显存。
    """

    def __init__(self, cfg: WalkieConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        # 全模型共享一个 RoPE 模块，避免每层重复缓存 cos/sin。
        self.rope = RotaryPositionalEmbedding(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.block_size,
            base=cfg.rope_theta,
            scaling_factor=cfg.rope_scaling_factor,
        )

        self.layers = nn.ModuleList(
            [WalkieBlock(cfg, rope=self.rope) for _ in range(cfg.n_layer)]
        )
        self.norm_out = RMSNorm(cfg.n_embd, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        # 残差路径上的输出投影做 1/sqrt(2*n_layer) 缩放（与 GPT-2 一致）
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _chunked_cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = self.cfg.loss_chunk_size
        if chunk_size is None or chunk_size <= 0 or hidden.size(1) <= chunk_size:
            logits = self.lm_head(hidden)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        total_loss = hidden.new_zeros(())
        total_tokens = targets.ne(-1).sum()
        for start in range(0, hidden.size(1), chunk_size):
            stop = min(start + chunk_size, hidden.size(1))
            logits = self.lm_head(hidden[:, start:stop, :])
            total_loss = total_loss + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, start:stop].reshape(-1),
                ignore_index=-1,
                reduction="sum",
            )

        denom = total_tokens.clamp_min(1).to(total_loss.dtype)
        return total_loss / denom

    # ----- 参数统计辅助 -----
    def num_parameters(self, only_trainable: bool = False) -> int:
        params = (p for p in self.parameters() if (not only_trainable) or p.requires_grad)
        # tied weights 在 nn.Module.parameters() 里只会被算一次，直接 sum 即可
        return sum(p.numel() for p in params)

    # ----- forward / loss -----
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"输入序列长度 {T} 超过 block_size {self.cfg.block_size}")

        x = self.drop(self.tok_embeddings(idx))
        if self.cfg.gradient_checkpointing and self.training:
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x)
        x = self.norm_out(x)

        if targets is not None:
            if not return_logits:
                return None, self._chunked_cross_entropy(x, targets)
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    # ----- generate（首版无 KV cache，逻辑参考现有 GPT2LMHeadModel） -----
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
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k).values[:, -1, None]
                    logits = torch.where(
                        logits < kth, torch.full_like(logits, float("-inf")), logits
                    )
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    mask = cum > top_p
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
