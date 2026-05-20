from __future__ import annotations

import numpy as np
import torch

from train.walkie_pretrain import ShuffledBlockSampler


def test_shuffled_block_sampler_covers_epoch_without_replacement():
    block_size = 4
    data = np.arange(block_size * 9 + 1, dtype=np.uint16)
    sampler = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=1,
        device=torch.device("cpu"),
        seed=123,
    )

    starts = [int(sampler.next_starts()[0]) for _ in range(sampler.num_samples)]

    assert sorted(starts) == [i * block_size for i in range(sampler.num_samples)]
    assert len(set(starts)) == sampler.num_samples


def test_shuffled_block_sampler_splits_global_batch_by_rank():
    block_size = 4
    data = np.arange(block_size * 16 + 1, dtype=np.uint16)
    sampler0 = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=2,
        device=torch.device("cpu"),
        seed=456,
        rank=0,
        world_size=2,
    )
    sampler1 = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=2,
        device=torch.device("cpu"),
        seed=456,
        rank=1,
        world_size=2,
    )

    starts0 = set(map(int, sampler0.next_starts()))
    starts1 = set(map(int, sampler1.next_starts()))

    assert starts0.isdisjoint(starts1)


def test_shuffled_block_sampler_resume_allows_same_global_batch_layout_change():
    block_size = 4
    data = np.arange(block_size * 32 + 1, dtype=np.uint16)
    single_rank = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=16,
        device=torch.device("cpu"),
        seed=654,
        rank=0,
        world_size=1,
    )
    single_rank.next_starts()
    state = single_rank.state_dict()
    expected_next_global_batch = set(map(int, single_rank.next_starts()))

    rank0 = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=8,
        device=torch.device("cpu"),
        seed=654,
        rank=0,
        world_size=2,
    )
    rank1 = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=8,
        device=torch.device("cpu"),
        seed=654,
        rank=1,
        world_size=2,
    )
    rank0.load_state_dict(state)
    rank1.load_state_dict(state)

    resumed_next_global_batch = set(map(int, rank0.next_starts())) | set(map(int, rank1.next_starts()))

    assert resumed_next_global_batch == expected_next_global_batch


def test_shuffled_block_sampler_state_resume_matches_next_batch():
    block_size = 4
    data = np.arange(block_size * 12 + 1, dtype=np.uint16)
    sampler = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=2,
        device=torch.device("cpu"),
        seed=789,
    )
    sampler.next_starts()
    state = sampler.state_dict()
    expected = sampler.next_starts()

    restored = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=2,
        device=torch.device("cpu"),
        seed=789,
    )
    restored.load_state_dict(state)

    assert np.array_equal(restored.next_starts(), expected)


def test_shuffled_block_sampler_next_batch_uses_contiguous_blocks():
    block_size = 4
    data = np.arange(block_size * 8 + 1, dtype=np.uint16)
    sampler = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=1,
        device=torch.device("cpu"),
        seed=321,
    )
    start = int(sampler.next_starts()[0])
    sampler = ShuffledBlockSampler(
        data,
        block_size,
        batch_size=1,
        device=torch.device("cpu"),
        seed=321,
    )

    x, y = sampler.next_batch()

    assert torch.equal(x[0], torch.arange(start, start + block_size))
    assert torch.equal(y[0], torch.arange(start + 1, start + block_size + 1))