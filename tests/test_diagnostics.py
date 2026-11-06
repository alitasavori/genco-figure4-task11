"""Forward diagnostics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import pbl_stats, summarize_rpe  # noqa: E402


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
