"""Focused tests for segmented SFT and bench-loop helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from posttrain.utils.schedule import WarmupDecaySchedule
from scripts.run_sft_bench_loop import (
    build_eval_command,
    build_train_command,
    choose_checkpoint_args,
    macro_pass_at_1,
    next_stop_step,
    preserve_base_checkpoint,
    should_run_full_eval,
)
from train.walkie_sft import (
    _build_checkpoint_extra,
    _restore_schedule_state,
    _resolve_stop_step,
    _swanlab_resume_run_id,
)


def _schedule(total_steps: int = 20) -> WarmupDecaySchedule:
    return WarmupDecaySchedule.from_config(
        total_steps=total_steps,
        warmup_steps=2,
        decay_shape="cosine",
        tracks={
            "adamw": {"peak_lr": 1.0, "final_lr": 0.1},
            "muon": {"peak_lr": 2.0, "final_lr": 0.2},
        },
    )


def test_sft_schedule_restore_keeps_current_total_steps() -> None:
    saved = _schedule(total_steps=10)
    saved.step_to(5)
    current = _schedule(total_steps=20)

    _restore_schedule_state(current, saved.state_dict(), current_total_steps=20, expected_step=5)

    assert current.total_steps == 20
    assert current.state_dict()["step"] == 5


def test_sft_schedule_restore_rejects_short_total_steps() -> None:
    saved = _schedule(total_steps=10)
    saved.step_to(5)
    current = _schedule(total_steps=10)

    with pytest.raises(RuntimeError, match="smaller than resumed step"):
        _restore_schedule_state(current, saved.state_dict(), current_total_steps=4, expected_step=5)


def test_sft_checkpoint_extra_uses_swanlab_id() -> None:
    class Run:
        id = "run-123"

    extra = _build_checkpoint_extra(global_sample_index=32, swanlab_run=Run(), resume_swanlab_run_id=None)

    assert extra["data_state"]["global_sample_index"] == 32
    assert extra["swanlab_run_id"] == "run-123"


def test_sft_checkpoint_extra_uses_private_walkie_run_id() -> None:
    class Run:
        _walkie_run_id = "generated-123"

    extra = _build_checkpoint_extra(global_sample_index=7, swanlab_run=Run(), resume_swanlab_run_id=None)

    assert extra["swanlab_run_id"] == "generated-123"


def test_sft_resume_run_id_prefers_swanlab_key() -> None:
    assert _swanlab_resume_run_id({"swanlab_run_id": "swan", "wandb_run_id": "wandb"}) == "swan"
    assert _swanlab_resume_run_id({"wandb_run_id": "wandb"}) == "wandb"


def test_sft_stop_step_caps_at_total() -> None:
    class TrainCfg:
        stop_step = 12

    assert _resolve_stop_step(TrainCfg(), total_steps=10) == 10


def test_loop_checkpoint_args_switch_to_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    assert choose_checkpoint_args(out_dir, init_from="base/latest.pt") == ["--init-from", "base/latest.pt"]
    (out_dir / "latest.pt").write_bytes(b"placeholder")
    assert choose_checkpoint_args(out_dir, init_from="base/latest.pt") == ["--resume", str(out_dir)]


def test_loop_build_train_command_uses_stop_step(tmp_path: Path) -> None:
    cmd = build_train_command(
        config="configs/train/sft.yaml",
        out_dir=tmp_path,
        init_from="base/latest.pt",
        total_steps=100,
        stop_step=25,
        train_overrides=["train.swanlab.mode=offline"],
    )

    assert "-m" in cmd
    assert "train.walkie_sft" in cmd
    assert "--init-from" in cmd
    assert f"train.out_dir={tmp_path}" in cmd
    assert "train.total_steps=100" in cmd
    assert "train.stop_step=25" in cmd
    assert "train.swanlab.mode=offline" in cmd


def test_loop_preserves_base_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "base" / "latest.pt"
    source.parent.mkdir()
    source.write_bytes(b"base weights")
    out_dir = tmp_path / "sft"
    out_dir.mkdir()

    copied = preserve_base_checkpoint(out_dir, init_from=str(source.parent))

    assert copied == out_dir / "base_model_latest.pt"
    assert copied.read_bytes() == b"base weights"
    assert (out_dir / "base_model_source.txt").read_text(encoding="utf-8").strip() == str(source)


def test_loop_macro_pass_at_1() -> None:
    summary = {"datasets": {"a": {"pass@1": 0.25}, "b": {"pass@1": 0.75}}}

    assert macro_pass_at_1(summary) == pytest.approx(0.5)


def test_loop_eval_schedule() -> None:
    assert next_stop_step(900, total_steps=2500, segment_steps=1000) == 1900
    assert next_stop_step(1900, total_steps=2500, segment_steps=1000) == 2500
    assert should_run_full_eval(1000, total_steps=3000, full_eval_interval=2000) is False
    assert should_run_full_eval(2000, total_steps=3000, full_eval_interval=2000) is True
    assert should_run_full_eval(3000, total_steps=3000, full_eval_interval=2000) is True


def test_loop_eval_command_uses_hf_flash_attention(tmp_path: Path) -> None:
    cmd = build_eval_command(
        model_dir=tmp_path / "hf",
        output_dir=tmp_path / "eval",
        bench_root="data/bench",
        sandbox_urls=["http://127.0.0.1:18901"],
        limit=4,
        batch_size=8,
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        timeout=10.0,
        max_concurrency=2,
        skip_sandbox_smoke=True,
    )

    assert "--backend" in cmd and "hf" in cmd
    assert "--attn-implementation" in cmd and "flash_attention_2" in cmd
    assert "--device" in cmd and "cuda:0" in cmd
    assert "--limit" in cmd and "4" in cmd
    assert "--skip-sandbox-smoke" in cmd