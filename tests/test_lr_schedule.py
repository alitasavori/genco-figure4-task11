"""GENCO-style LR schedule: Adam 5e-4, ReduceLROnPlateau x0.7 after patience 5."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    LEARNING_RATE,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
    current_lr,
    make_lr_scheduler,
    train_epoch_loss_line,
)


def make_optimizer() -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([param], lr=LEARNING_RATE)


def test_schedule_constants_follow_genco():
    assert LEARNING_RATE == pytest.approx(5e-4)
    assert LR_PLATEAU_FACTOR == pytest.approx(0.7)
    assert LR_PLATEAU_PATIENCE == 5


def test_plateau_cuts_lr_after_six_non_improving_validations():
    optimizer = make_optimizer()
    scheduler = make_lr_scheduler(optimizer)
    scheduler.step(1.0)
    for _ in range(LR_PLATEAU_PATIENCE):
        scheduler.step(1.0)
        assert current_lr(optimizer) == pytest.approx(5e-4)
    scheduler.step(1.0)
    assert current_lr(optimizer) == pytest.approx(5e-4 * 0.7)


def test_scheduler_state_round_trips():
    optimizer = make_optimizer()
    scheduler = make_lr_scheduler(optimizer)
    for value in (1.0, 1.0, 1.0):
        scheduler.step(value)
    restored = make_lr_scheduler(make_optimizer())
    restored.load_state_dict(scheduler.state_dict())
    assert restored.num_bad_epochs == scheduler.num_bad_epochs
    assert restored.best == scheduler.best


def test_epoch_line_shows_lr():
    line = train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "0.123456", lr=5e-4)
    assert line.endswith("val_pbl 0.123456  lr 5.0e-04")
    assert "lr" not in train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "skipped")
