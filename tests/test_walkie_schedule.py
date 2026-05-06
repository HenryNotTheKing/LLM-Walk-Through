"""Walkie WSD 调度器测试：连续性、阶段切换、序列化。"""

from __future__ import annotations

import math

import pytest

from core.utils.walkie_schedule import WalkieWSDSchedule


def make_sched(total=100, warmup=10, anneal_ratio=0.8, shape="sqrt"):
    return WalkieWSDSchedule.from_config(
        total_steps=total,
        warmup_steps=warmup,
        anneal_start_ratio=anneal_ratio,
        decay_shape=shape,
        tracks={
            "adamw": {"peak_lr": 1.0, "final_lr": 0.1},
            "muon": {"peak_lr": 0.5, "final_lr": 0.05},
        },
    )


def test_warmup_starts_at_zero_and_ends_at_peak():
    s = make_sched()
    assert s.lr_at(0, "adamw") == pytest.approx(0.0)
    assert s.lr_at(10, "adamw") == pytest.approx(1.0)
    assert s.lr_at(10, "muon") == pytest.approx(0.5)


def test_stable_phase_holds_peak():
    s = make_sched()
    for step in [10, 30, 79]:
        assert s.lr_at(step, "adamw") == pytest.approx(1.0)


def test_continuity_at_anneal_start():
    """关键：进入 anneal 时 lr 必须连续，等于 peak_lr。"""
    s = make_sched()
    last_stable = s.lr_at(s.anneal_start - 1, "adamw")
    first_anneal = s.lr_at(s.anneal_start, "adamw")
    assert math.isclose(first_anneal, 1.0, abs_tol=1e-9)
    assert math.isclose(last_stable, first_anneal, rel_tol=0.0, abs_tol=1e-9)


def test_decay_reaches_final_at_total():
    s = make_sched()
    assert s.lr_at(s.total_steps, "adamw") == pytest.approx(0.1, abs=1e-6)
    assert s.lr_at(s.total_steps, "muon") == pytest.approx(0.05, abs=1e-6)


def test_current_stage_boundary():
    s = make_sched()
    assert s.current_stage(s.anneal_start - 1) == "main"
    assert s.current_stage(s.anneal_start) == "anneal"
    assert s.current_stage(s.total_steps) == "anneal"


def test_state_dict_round_trip():
    s = make_sched()
    s.step_to(42)
    blob = s.state_dict()
    s2 = make_sched(total=1, warmup=0, anneal_ratio=0.5)
    s2.load_state_dict(blob)
    assert s2.step == 42
    assert s2.total_steps == s.total_steps
    assert s2.lr_at(42, "muon") == s.lr_at(42, "muon")


def test_decay_shapes_all_continuous():
    for shape in ("sqrt", "linear", "cosine"):
        s = make_sched(shape=shape)
        assert s.lr_at(s.anneal_start, "adamw") == pytest.approx(1.0)
        assert s.lr_at(s.total_steps, "adamw") == pytest.approx(0.1, abs=1e-6)
