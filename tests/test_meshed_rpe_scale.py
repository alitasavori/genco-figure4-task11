"""Meshed RPE admittance scaling and component attention masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    component_admittance_scale,
    meshed_rpe,
    type_pair_masks,
)


def test_admittance_scale_ignores_zero_off_diagonal():
    component = np.array([0, 0, 0], dtype=np.int64)
    y_abs = np.zeros((3, 3), dtype=np.float64)
    y_abs[0, 1] = 2.0
    y_abs[1, 0] = 2.0
    scale = component_admittance_scale(y_abs, component, 0)
    assert scale == 2.0
    mean_including_zeros = float(y_abs[np.ix_([0, 1, 2], [0, 1, 2])][~np.eye(3, dtype=bool)].mean())
    assert mean_including_zeros < scale


def test_meshed_rpe_uses_nonzero_only_admittance_scale():
    branches = [(0, 1, 0.1, 0.1), (1, 2, 0.1, 0.1), (0, 2, 0.1, 0.1)]
    ybus = np.zeros((3, 3), dtype=np.complex64)
    ybus[0, 1] = ybus[1, 0] = 3.0 + 4.0j
    features, component = meshed_rpe(3, branches, ybus)
    scale = component_admittance_scale(np.abs(ybus), component, 0)
    assert abs(features[0, 1, 0] - 3.0 / scale) < 1e-5
    assert abs(features[0, 1, 1] - 4.0 / scale) < 1e-5


def test_type_pair_masks_block_cross_component_attention():
    types = torch.tensor([0, 1, 0], dtype=torch.long)
    component = torch.tensor([0, 0, 1], dtype=torch.long)
    masks = type_pair_masks(types, component)
    assert masks.shape == (9, 3, 3)
    cross = (component.view(3, 1) != component.view(1, 3))
    assert not bool((masks & cross.unsqueeze(0)).any())
