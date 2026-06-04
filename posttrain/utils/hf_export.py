"""Export Walkie checkpoints to a Qwen3-compatible HF/vLLM directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.walkie_checkpoint import load_walkie_checkpoint, resolve_resume_path, unwrap_model


def export_walkie_to_hf(
    checkpoint_or_model: str | Path | WalkieForCausalLM,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> Path:
    """Write weights/config/tokenizer for vLLM rollout.

    The exported config uses `model_type=qwen3` because Qwen3 shares Walkie's
    RMSNorm + SwiGLU + GQA + QK-norm shape. No files are downloaded.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if isinstance(checkpoint_or_model, WalkieForCausalLM):
        model = unwrap_model(checkpoint_or_model)
        cfg = model.cfg.to_dict()
        state = model.state_dict()
    else:
        payload = load_walkie_checkpoint(resolve_resume_path(checkpoint_or_model), map_location="cpu", weights_only=False)
        cfg = dict(payload.get("model_cfg") or model_cfg or {})
        if not cfg:
            raise ValueError("checkpoint does not contain model_cfg; pass model_cfg explicitly")
        state = payload["model"]
    walkie_cfg = WalkieConfig.from_dict(cfg)
    hf_state = map_walkie_state_to_qwen3(state, walkie_cfg)

    weights_path = output / "model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(_clone_shared_storage_tensors(hf_state), weights_path)
    except ImportError:  # pragma: no cover - optional dependency fallback
        weights_path = output / "pytorch_model.bin"
        torch.save(hf_state, weights_path)

    (output / "config.json").write_text(json.dumps(_qwen3_config(walkie_cfg), indent=2), encoding="utf-8")
    (output / "generation_config.json").write_text(
        json.dumps({"bos_token_id": None, "eos_token_id": 0, "pad_token_id": 1}, indent=2),
        encoding="utf-8",
    )
    if tokenizer_path is not None:
        src = Path(tokenizer_path)
        if src.is_dir():
            for child in src.iterdir():
                if child.is_file() and child.name in {"tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"}:
                    shutil.copy2(child, output / child.name)
        elif src.is_file():
            shutil.copy2(src, output / "tokenizer.json")
    return output


def map_walkie_state_to_qwen3(state: dict[str, torch.Tensor], cfg: WalkieConfig) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": state["tok_embeddings.weight"],
        "model.norm.weight": state["norm_out.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for layer_index in range(cfg.n_layer):
        prefix = f"layers.{layer_index}"
        hf_prefix = f"model.layers.{layer_index}"
        mapped[f"{hf_prefix}.input_layernorm.weight"] = state[f"{prefix}.norm_attn.weight"]
        mapped[f"{hf_prefix}.post_attention_layernorm.weight"] = state[f"{prefix}.norm_ffn.weight"]
        mapped[f"{hf_prefix}.self_attn.q_proj.weight"] = state[f"{prefix}.attn.q_proj.weight"]
        mapped[f"{hf_prefix}.self_attn.k_proj.weight"] = state[f"{prefix}.attn.k_proj.weight"]
        mapped[f"{hf_prefix}.self_attn.v_proj.weight"] = state[f"{prefix}.attn.v_proj.weight"]
        mapped[f"{hf_prefix}.self_attn.o_proj.weight"] = state[f"{prefix}.attn.o_proj.weight"]
        if cfg.qk_norm:
            mapped[f"{hf_prefix}.self_attn.q_norm.weight"] = state[f"{prefix}.attn.q_norm.weight"]
            mapped[f"{hf_prefix}.self_attn.k_norm.weight"] = state[f"{prefix}.attn.k_norm.weight"]
        mapped[f"{hf_prefix}.mlp.gate_proj.weight"] = state[f"{prefix}.mlp.gate_proj.weight"]
        mapped[f"{hf_prefix}.mlp.up_proj.weight"] = state[f"{prefix}.mlp.up_proj.weight"]
        mapped[f"{hf_prefix}.mlp.down_proj.weight"] = state[f"{prefix}.mlp.down_proj.weight"]
    return mapped


def _clone_shared_storage_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    seen_storage_ptrs: set[int] = set()
    for key, tensor in state.items():
        value = tensor.detach().contiguous()
        storage_ptr = value.untyped_storage().data_ptr()
        if storage_ptr in seen_storage_ptrs:
            value = value.clone()
            storage_ptr = value.untyped_storage().data_ptr()
        seen_storage_ptrs.add(storage_ptr)
        cloned[key] = value
    return cloned


def _qwen3_config(cfg: WalkieConfig) -> dict[str, Any]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.n_embd,
        "intermediate_size": cfg.d_ffn,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "num_key_value_heads": cfg.n_head_kv,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.block_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "attention_bias": cfg.bias,
        "tie_word_embeddings": cfg.tie_weights,
        "hidden_act": "silu",
        "torch_dtype": "bfloat16",
        "bos_token_id": None,
        "eos_token_id": 0,
        "pad_token_id": 1,
        "use_cache": True,
    }