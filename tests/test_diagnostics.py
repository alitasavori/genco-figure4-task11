"""Forward diagnostics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    RPE_DIM,
    TPGT,
    Network,
    audit_rpe_correctness,
    format_rpe_effectiveness_diagnostics,
    meshed_rpe,
    pair_encoder_input_weight_norms,
    pbl_stats,
    summarize_rpe,
)


def test_pbl_stats_on_zero_residual():
    residual = torch.zeros(4, 118, 2)
    stats = pbl_stats(residual)
    assert stats["mean"] == 0.0
    assert stats["max"] == 0.0


def test_summarize_rpe_shape():
    rpe = torch.randn(5, 5, 10)
    rows = summarize_rpe(rpe)
    assert len(rows) == 10
    assert rows[0]["off_mean"] == rows[0]["off_mean"]  # finite


def test_pair_encoder_weight_norms():
    model = TPGT()
    rows = pair_encoder_input_weight_norms(model)
    assert len(rows) == RPE_DIM
    assert all(row["weight_l2"] >= 0.0 for row in rows)


def test_format_rpe_effectiveness_without_ablation():
    weights = [{"name": "adjacency", "weight_l2": 1.0}]
    text = format_rpe_effectiveness_diagnostics(weights, None, None)
    assert "not run" in text
    assert "adjacency" in text


def test_audit_rpe_correctness_small_graph():
    branches = [(0, 1, 0.1, 0.1), (1, 2, 0.1, 0.1)]
    ybus = np.zeros((3, 3), dtype=np.complex64)
    ybus[0, 1] = ybus[1, 0] = 2.0 + 1.0j
    features, component = meshed_rpe(3, branches, ybus)
    network = Network(
        ybus=torch.from_numpy(ybus),
        rpe=torch.from_numpy(features),
        component=torch.from_numpy(component.astype(np.int64)),
        types=torch.tensor([0, 1, 2], dtype=torch.long),
        branches=tuple(branches),
    )
    checks = audit_rpe_correctness(network)
    assert all(check.passed for check in checks)
