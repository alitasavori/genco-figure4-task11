"""Cosine LR ceiling + val plateau schedule and gradient clip."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    EPOCHS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LR_COSINE_MIN_RATIO,
    cosine_lr_at_epoch,
    current_lr,
    make_lr_schedule,
    restore_lr_schedule,
    train_epoch_loss_line,
)


def make_optimizer() -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([param], lr=LEARNING_RATE)


def test_schedule_constants():
    assert LEARNING_RATE == pytest.approx(5e-4)
    assert LR_COSINE_MIN_RATIO == pytest.approx(0.05)
    assert GRAD_CLIP_NORM > 0.0
    assert cosine_lr_at_epoch(0) == pytest.approx(LEARNING_RATE)
    assert cosine_lr_at_epoch(EPOCHS) == pytest.approx(LEARNING_RATE * LR_COSINE_MIN_RATIO)


def test_cosine_ceiling_shrinks_lr_over_epochs():
    optimizer = make_optimizer()
    schedule = make_lr_schedule(optimizer)
    assert current_lr(optimizer) == pytest.approx(LEARNING_RATE)
    mid = EPOCHS // 2
    for ep in range(1, mid + 1):
        schedule.step_epoch(ep, 0.2 - 0.001 * ep)
    mid_lr = current_lr(optimizer)
    assert mid_lr < LEARNING_RATE
    assert mid_lr > LEARNING_RATE * LR_COSINE_MIN_RATIO
    for ep in range(mid + 1, EPOCHS + 1):
        schedule.step_epoch(ep, 0.2 - 0.001 * ep)
    assert current_lr(optimizer) == pytest.approx(LEARNING_RATE * LR_COSINE_MIN_RATIO, rel=1e-5)


def test_scheduler_resume_syncs_completed_epochs():
    walked_opt = make_optimizer()
    walked = make_lr_schedule(walked_opt)
    for ep in range(1, 8):
        walked.step_epoch(ep, 0.2 - 0.001 * ep)
    resumed_opt = make_optimizer()
    resumed = restore_lr_schedule(resumed_opt, None, completed_epochs=7)
    assert current_lr(resumed_opt) == pytest.approx(current_lr(walked_opt))


def test_plateau_halves_lr_on_flat_val():
    optimizer = make_optimizer()
    schedule = make_lr_schedule(optimizer)
    stale = 0.12
    for ep in range(1, 10):
        schedule.step_epoch(ep, stale)
    assert current_lr(optimizer) < LEARNING_RATE


def test_scheduler_state_round_trips():
    optimizer = make_optimizer()
    schedule = make_lr_schedule(optimizer)
    for ep in range(1, 4):
        schedule.step_epoch(ep, 0.1 - 0.001 * ep)
    state = schedule.state_dict()
    restored_opt = make_optimizer()
    restored_opt.load_state_dict(optimizer.state_dict())
    restored = make_lr_schedule(restored_opt)
    restored.load_state_dict(state)
    assert current_lr(restored_opt) == pytest.approx(current_lr(optimizer))


def test_epoch_line_shows_lr():
    line = train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "0.123456", lr=5e-4)
    assert line.endswith("val_pbl 0.123456  lr 5.0e-04")
    assert "lr" not in train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "skipped")
