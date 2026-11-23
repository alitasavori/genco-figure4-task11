"""Newton logger smoke on toy 3-bus."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jsgc_physics import TYPE_PQ, TYPE_PV, TYPE_SLACK  # noqa: E402
from jmat.pf.newton_logger import newton_trace, verify_linear_residual  # noqa: E402


def test_newton_trace_linear_residual():
    y_s = 1.0 / (0.01 + 1j * 0.1)
    ybus = torch.zeros(3, 3, dtype=torch.complex128)
    for i, j in ((0, 1), (1, 2)):
        ybus[i, j] = ybus[j, i] = -y_s
        ybus[i, i] += y_s
        ybus[j, j] += y_s
    types = torch.tensor([TYPE_PQ, TYPE_PV, TYPE_SLACK], dtype=torch.long)
    g = ybus.real.to(torch.float64)
    b = ybus.imag.to(torch.float64)
    p_spec = torch.tensor([-0.5, 0.8, 0.0], dtype=torch.float64)
    q_spec = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64)
    steps, status = newton_trace(g, b, types, p_spec, q_spec, max_iter=15)
    assert steps
    for step in steps:
        assert step.lin_residual_rel < 1e-6
    assert status["iterations"] >= 1