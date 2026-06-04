"""Torch rollout backend using the in-process Walkie actor."""

from __future__ import annotations

from typing import Any

import torch

from .base import RolloutOutput, SamplingConfig


class TorchRolloutEngine:
    def __init__(self, model: Any, tokenizer: Any, *, device: torch.device, dtype: torch.dtype, use_amp: bool) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.use_amp = bool(use_amp)

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[RolloutOutput]:
        outputs: list[RolloutOutput] = []
        was_training = bool(getattr(self.model, "training", False))
        try:
            self.model.eval()
            if sampling.seed is not None:
                torch.manual_seed(int(sampling.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(sampling.seed))
            for prompt_index, prompt in enumerate(prompts):
                prompt_ids = self.tokenizer.encode(prompt)
                if not prompt_ids:
                    continue
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                for generation_index in range(int(sampling.num_generations)):
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                        generated = self.model.generate(
                            input_ids.clone(),
                            max_new_tokens=int(sampling.max_tokens),
                            temperature=float(sampling.temperature),
                            top_p=float(sampling.top_p),
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    completion_ids = _strip_after_eos(
                        generated[0, len(prompt_ids) :].detach().cpu().tolist(),
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    text = _trim_stop(self.tokenizer.decode(completion_ids), list(sampling.stop or []))
                    outputs.append(
                        RolloutOutput(
                            prompt=prompt,
                            response=text,
                            prompt_index=prompt_index,
                            generation_index=generation_index,
                            metadata={"backend": "torch"},
                        )
                    )
        finally:
            if was_training:
                self.model.train()
        return outputs


def _strip_after_eos(token_ids: list[int], *, eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    if eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id)]
    return token_ids


def _trim_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for marker in stop:
        if not marker:
            continue
        position = text.find(marker)
        if position >= 0:
            cut = min(cut, position)
    return text[:cut]