"""Transformers generation helpers for code evaluation."""

from __future__ import annotations

from typing import Any

from posttrain.eval.vllm_runner import EvalSamplingConfig


def generate_with_hf(
    model_path: str,
    prompts: list[str],
    *,
    sampling: EvalSamplingConfig,
    batch_size: int = 4,
    device: str = "auto",
    dtype: str = "auto",
    attn_implementation: str = "auto",
) -> list[list[str]]:
    try:
        import torch
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("HF evaluation requires the posttrain dependencies") from exc

    if sampling.seed is not None:
        torch.manual_seed(int(sampling.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sampling.seed))

    resolved_device = _resolve_device(device, torch)
    resolved_dtype = _resolve_dtype(dtype, resolved_device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if resolved_dtype is not None:
        model_kwargs["dtype"] = resolved_dtype
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    model.to(resolved_device)

    do_sample = sampling.temperature > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": sampling.max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update({"temperature": sampling.temperature, "top_p": sampling.top_p})
    if sampling.n > 1:
        generate_kwargs["num_return_sequences"] = sampling.n
        if not do_sample:
            generate_kwargs["num_beams"] = sampling.n

    completions: list[list[str]] = []
    effective_batch_size = max(1, int(batch_size))
    try:
        for start in tqdm(range(0, len(prompts), effective_batch_size), desc="HF generate", unit="batch"):
            batch_prompts = prompts[start : start + effective_batch_size]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_width = int(encoded.input_ids.shape[-1])
            with torch.inference_mode():
                generated = model.generate(**encoded, **generate_kwargs)
            batch_texts = [
                _trim_stop(tokenizer.decode(row[prompt_width:], skip_special_tokens=True), sampling.stop)
                for row in generated
            ]
            for index in range(0, len(batch_texts), sampling.n):
                completions.append(batch_texts[index : index + sampling.n])
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return completions


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: Any, torch: Any) -> Any | None:
    normalized = dtype.lower()
    if normalized == "auto":
        if getattr(device, "type", str(device)) == "cuda":
            return torch.bfloat16
        return None
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported HF dtype: {dtype}")
    return aliases[normalized]


def _trim_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for marker in stop:
        if not marker:
            continue
        position = text.find(marker)
        if position >= 0:
            cut = min(cut, position)
    return text[:cut]