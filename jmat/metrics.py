"""Aggregate benchmark metrics (Phase 2 Task 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricSummary:
    count: int
    success_rate: float
    er_x_median: float
    er_x_p95: float
    er_x_max: float
    er_r_median: float
    er_r_p95: float
    er_r_max: float
    tol_hit_rates: dict[str, float]


def relative_errors(dx_hat: np.ndarray, dx_ref: np.ndarray, j, r: np.ndarray) -> tuple[float, float]:
    er_x = float(np.linalg.norm(dx_hat - dx_ref) / (np.linalg.norm(dx_ref) + 1e-30))
    res = j @ dx_hat - r
    er_r = float(np.linalg.norm(res) / (np.linalg.norm(r) + 1e-30))
    return er_x, er_r


def summarize_errors(
    er_x: list[float],
    er_r: list[float],
    *,
    rtol_levels: tuple[str, ...] = ("1e-3", "1e-4", "1e-5", "1e-6"),
) -> MetricSummary:
    ex = np.asarray(er_x, dtype=np.float64)
    er = np.asarray(er_r, dtype=np.float64)
    tol_hit = {}
    for tag in rtol_levels:
        t = float(tag)
        tol_hit[tag] = float(np.mean(er <= t))
    return MetricSummary(
        count=int(ex.size),
        success_rate=float(np.mean(er <= 1e-6)),
        er_x_median=float(np.median(ex)),
        er_x_p95=float(np.percentile(ex, 95)),
        er_x_max=float(np.max(ex)) if ex.size else 0.0,
        er_r_median=float(np.median(er)),
        er_r_p95=float(np.percentile(er, 95)),
        er_r_max=float(np.max(er)) if er.size else 0.0,
        tol_hit_rates=tol_hit,
    )