"""GENCO-style per-stage physics weights: stage i of K weighted 0.5**(K-i)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    DEPTH,
    LAMBDA_PHYS,
    LAMBDA_STATE,
    _weighted_loss_breakdown,
    physics_stage_weights,
    task_loss,
    train_epoch_loss_line,
)


def test_weights_depth_4():
    assert physics_stage_weights(4) == pytest.approx((0.125, 0.25, 0.5, 1.0))
    assert physics_stage_weights(1) == pytest.approx((1.0,))


def synthetic_inputs(seed: int):
    gen = torch.Generator().manual_seed(seed)
    batch, buses = 3, 5
    prediction = torch.randn(batch, buses, 2, generator=gen)
    target = torch.randn(batch, buses, 2, generator=gen)
    types = torch.tensor([0, 0, 1, 1, 2])
    residuals = [torch.randn(batch, buses, 2, generator=gen) * (i + 1) for i in range(DEPTH)]
    return prediction, target, residuals, types


def test_task_loss_uses_weighted_stage_sum():
    prediction, target, residuals, types = synthetic_inputs(0)
    breakdown = task_loss(prediction, target, residuals, types)
    raw = [torch.mean(torch.sum(r * r, dim=-1)) for r in residuals]
    for logged, expected in zip(breakdown.physics_stages, raw):
        assert float(logged) == pytest.approx(float(expected))
    weighted = sum(w * float(r) for w, r in zip(physics_stage_weights(DEPTH), raw))
    assert float(breakdown.physics_sum) == pytest.approx(weighted)
    assert float(breakdown.total) == pytest.approx(
        LAMBDA_STATE * float(breakdown.state_mse) + LAMBDA_PHYS * weighted
    )


def test_micro_batch_average_stays_weighted():
    parts = []
    for seed, weight in ((1, 2.0), (2, 1.0)):
        prediction, target, residuals, types = synthetic_inputs(seed)
        parts.append((task_loss(prediction, target, residuals, types), weight))
    merged = _weighted_loss_breakdown(parts)
    weighted = sum(w * float(s) for w, s in zip(physics_stage_weights(DEPTH), merged.physics_stages))
    assert float(merged.physics_sum) == pytest.approx(weighted)


def test_epoch_line_labels_weighted_physics():
    line = train_epoch_loss_line(0, 1, 2, 0.2, 0.4, [0.1, 0.1], 0.6, "0.1")
    assert "phys_w 0.2000" in line
