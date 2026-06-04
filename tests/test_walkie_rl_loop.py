"""Focused tests for RL segmented-loop and DDP helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from posttrain.rewards.registry import RewardScore
from posttrain.rl.algorithms.dapo import dapo_group_filter
from scripts.run_rl_humaneval_loop import build_train_command, effective_distributed_backend, visible_gpu_count
from train.walkie_rl import PromptExample, _log_swanlab, _reward_component_tensor, _rollout_log_stats, _shard_prompt_batch


def test_rl_loop_build_train_command_uses_torchrun_for_multiple_processes(tmp_path: Path) -> None:
    cmd = build_train_command(
        config="configs/train/rl.yaml",
        out_dir=tmp_path,
        init_from="base/latest.pt",
        total_steps=100,
        stop_step=20,
        nproc_per_node=2,
    )

    assert "torch.distributed.run" in cmd
    assert "--nproc_per_node=2" in cmd
    assert "train.walkie_rl" in cmd
    assert f"train.out_dir={tmp_path}" in cmd
    assert "train.stop_step=20" in cmd


def test_rl_loop_gpu_count_and_backend_override() -> None:
    cfg = OmegaConf.create({"distributed": {"backend": "ddp"}})

    assert visible_gpu_count("0,1") == 2
    assert visible_gpu_count("1") == 1
    assert effective_distributed_backend(cfg, []) == "ddp"
    assert effective_distributed_backend(cfg, ["distributed.backend=none"]) == "none"


def test_rl_prompt_batch_is_sharded_by_rank() -> None:
    class DistInfo:
        enabled = True
        rank = 1
        world_size = 2

    batch = [PromptExample(text=f"prompt-{index}", metadata={}) for index in range(6)]

    shard = _shard_prompt_batch(batch, DistInfo())

    assert [item.text for item in shard] == ["prompt-1", "prompt-3", "prompt-5"]


def test_dapo_filter_can_use_binary_execution_component() -> None:
    prompt_ids = torch.tensor([0, 0])
    total_rewards = torch.tensor([0.05, -0.10])
    reward_scores = [
        RewardScore(score=0.05, components={"code_execution": 0.0, "repeat_penalty": 1.0}),
        RewardScore(score=-0.10, components={"code_execution": 0.0, "repeat_penalty": 0.0}),
    ]

    component_rewards = _reward_component_tensor(reward_scores, "code_execution", device=torch.device("cpu"))

    assert dapo_group_filter(total_rewards, prompt_ids).tolist() == [True, True]
    assert dapo_group_filter(component_rewards, prompt_ids).tolist() == [False, False]


def test_rollout_log_stats_reports_group_filter_reasons() -> None:
    class DistInfo:
        enabled = False

    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    filter_rewards = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    prompt_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    keep_mask = dapo_group_filter(filter_rewards, prompt_ids)

    stats = _rollout_log_stats(
        rewards,
        keep_mask,
        prompt_ids,
        filter_rewards=filter_rewards,
        filter_component="code_execution",
        device=torch.device("cpu"),
        dist_info=DistInfo(),
    )

    assert stats["reward_mean"] == 0.75
    assert stats["keep_fraction"] == 0.5
    assert stats["local_keep_fraction"] == 0.5
    assert stats["code_execution_mean"] == 0.75
    assert stats["code_execution_positive_fraction"] == 0.75
    assert stats["dapo_total_groups"] == 2.0
    assert stats["dapo_kept_groups"] == 1.0
    assert stats["dapo_all_positive_groups"] == 1.0
    assert stats["dapo_all_non_positive_groups"] == 0.0


def test_rl_swanlab_logging_skips_non_numeric_stats() -> None:
    class Run:
        def __init__(self) -> None:
            self.payload = None
            self.step = None

        def log(self, payload, *, step: int) -> None:
            self.payload = payload
            self.step = step

    run = Run()

    _log_swanlab(
        run,
        {"reward_mean": 0.5, "loss_normalization": "token"},
        3,
        total_steps=10,
        lrs={"adamw": 1e-6, "muon": 1e-4},
        elapsed=12.0,
    )

    assert run.step == 3
    assert run.payload["rl/reward_mean"] == 0.5
    assert "rl/loss_normalization" not in run.payload