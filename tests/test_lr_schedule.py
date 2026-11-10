"""Cosine LR decay + gradient clip (anti-spike schedule)."""

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
    make_lr_scheduler,
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


def test_cosine_scheduler_shrinks_lr_over_epochs():
    optimizer = make_optimizer()
    scheduler = make_lr_scheduler(optimizer)
    assert current_lr(optimizer) == pytest.approx(LEARNING_RATE)
    mid = EPOCHS // 2
    for _ in range(mid):
        optimizer.step()
        scheduler.step()
    mid_lr = current_lr(optimizer)
    assert mid_lr < LEARNING_RATE
    assert mid_lr > LEARNING_RATE * LR_COSINE_MIN_RATIO
    for _ in range(EPOCHS - mid):
        optimizer.step()
        scheduler.step()
    assert current_lr(optimizer) == pytest.approx(LEARNING_RATE * LR_COSINE_MIN_RATIO, rel=1e-5)


def test_scheduler_resume_syncs_completed_epochs():
    walked = make_optimizer()
    scheduler = make_lr_scheduler(walked)
    for _ in range(7):
        walked.step()
        scheduler.step()
    resumed = make_lr_scheduler(make_optimizer(), completed_epochs=7)
    assert current_lr(resumed.optimizer) == pytest.approx(current_lr(walked))


def test_scheduler_state_round_trips():
    optimizer = make_optimizer()
    scheduler = make_lr_scheduler(optimizer)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    restored_opt = make_optimizer()
    restored_opt.load_state_dict(optimizer.state_dict())
    restored = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_opt,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * LR_COSINE_MIN_RATIO,
    )
    restored.load_state_dict(scheduler.state_dict())
    assert restored.last_epoch == scheduler.last_epoch
    assert current_lr(restored_opt) == pytest.approx(current_lr(optimizer))


def test_epoch_line_shows_lr():
    line = train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "0.123456", lr=5e-4)
    assert line.endswith("val_pbl 0.123456  lr 5.0e-04")
    assert "lr" not in train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "skipped")
