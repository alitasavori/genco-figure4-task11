"""Aggregate the per-seed test scores written by the authors' evaluation code."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def sample_std(values: Sequence[float]) -> float:
    """Unbiased sample standard deviation, matching torch.Tensor.std."""
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        raise ValueError("Standard deviation over seeds needs at least two values.")
    if not np.isfinite(array).all():
        raise ValueError("Seed metrics contain a non-finite value.")
    return float(np.std(array, ddof=1))


def sample_mean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Seed metrics must be finite and non-empty.")
    return float(array.mean())


def aggregate_c2i(category_means: dict[str, float], counts: dict[str, int] | None = None) -> float:
    """Mean of the N, N-1, and N-2 close-to-infeasible scores.

    scripts/test_error.py averages the three topology means. Equal counts make
    that the same as a sample-weighted mean. Differing counts use the weights.
    """
    keys = ("C2I-N", "C2I-N-1", "C2I-N-2")
    missing = [key for key in keys if key not in category_means]
    if missing:
        raise ValueError(f"C2I is incomplete, missing {missing}.")
    values = [float(category_means[key]) for key in keys]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("C2I category mean is non-finite.")
    if not counts:
        return float(np.mean(values))
    weights = [int(counts[key]) for key in keys]
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"C2I sample counts must be positive, got {weights}.")
    if len(set(weights)) == 1:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def assert_no_overlap(train_ids: Iterable[str], test_ids: Iterable[str]) -> None:
    overlap = set(map(str, train_ids)) & set(map(str, test_ids))
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(f"Training and test ids overlap, for example {sample}.")
