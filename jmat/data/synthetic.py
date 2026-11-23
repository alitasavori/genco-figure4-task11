"""Synthetic nonsymmetric sparse systems for JMAT Phase 2 (Task 2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random


class SyntheticFamily(str, Enum):
    DIAG_DOMINANT = "diagonal_dominant"
    ILL_CONDITIONED = "ill_conditioned"
    RADIAL = "radial"
    MESHED = "meshed"


@dataclass(frozen=True)
class SyntheticMeta:
    family: SyntheticFamily
    n: int
    cond_hint: float
    matrix_id: int


def _finalize(j: np.ndarray, *, min_diag: float = 0.5) -> np.ndarray:
    j = np.asarray(j, dtype=np.float64)
    n = j.shape[0]
    for i in range(n):
        row_sum = np.sum(np.abs(j[i])) - abs(j[i, i])
        if j[i, i] <= row_sum + min_diag:
            j[i, i] = row_sum + min_diag + 0.1
    return j


def diagonal_dominant(n: int, rng: np.random.Generator, *, density: float = 0.15) -> np.ndarray:
    j = sparse_random(
        n,
        n,
        density=density,
        random_state=rng,
        data_rvs=rng.standard_normal,
    ).toarray()
    j += 0.1 * rng.standard_normal((n, n))
    return _finalize(j)


def ill_conditioned(n: int, rng: np.random.Generator) -> np.ndarray:
  # Ill-conditioned nonsymmetric via random A, scale σ_max/σ_min
    a = rng.standard_normal((n, n))
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    smin, smax = 1.0, 1e4
    s = np.geomspace(smax, smin, n)
    j = (u * s) @ vh
    return _finalize(j, min_diag=1.0)


def radial_chain(n: int, rng: np.random.Generator) -> np.ndarray:
    j = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j[i, i] = 3.0 + 0.1 * rng.random()
        if i + 1 < n:
            c = -(1.0 + 0.2 * rng.random())
            j[i, i + 1] = c
            j[i + 1, i] = 0.5 * c
    return _finalize(j)


def meshed_laplacian_like(n: int, rng: np.random.Generator, *, degree: int = 4) -> np.ndarray:
    """Small-world style connections + random skew."""
    j = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for d in range(1, degree + 1):
            k = i + d
            if k < n:
                w = rng.uniform(-0.8, 0.8)
                j[i, k] = w
                j[k, i] = -0.3 * w
    j += 0.05 * rng.standard_normal((n, n))
    return _finalize(j)


def sample_matrix(
    family: SyntheticFamily,
    n: int,
    matrix_id: int,
    *,
    seed: int = 0,
) -> tuple[csr_matrix, SyntheticMeta]:
    rng = np.random.default_rng(seed + matrix_id * 10007 + n * 13 + hash(family.value) % 1000)
    if family == SyntheticFamily.DIAG_DOMINANT:
        dense = diagonal_dominant(n, rng)
    elif family == SyntheticFamily.ILL_CONDITIONED:
        dense = ill_conditioned(n, rng)
    elif family == SyntheticFamily.RADIAL:
        dense = radial_chain(n, rng)
    else:
        dense = meshed_laplacian_like(n, rng)
    j = csr_matrix(dense)
    cond = float(np.linalg.cond(dense))
    meta = SyntheticMeta(family=family, n=n, cond_hint=cond, matrix_id=matrix_id)
    return j, meta


def sample_rhs_ensemble(
    j: csr_matrix,
    rng: np.random.Generator,
    n_rhs: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (r, x_star) with r = J x_star."""
    n = j.shape[0]
    j_dense = j.toarray()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_rhs):
        x = rng.standard_normal(n)
        r = j_dense @ x
        pairs.append((r.astype(np.float64), x.astype(np.float64)))
    return pairs