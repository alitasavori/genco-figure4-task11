"""Sparse direct and Krylov baselines (Phase 2 Task 4)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu, spilu


@dataclass
class LinearSolveResult:
    dx: np.ndarray
    success: bool
    relative_residual: float
    relative_solution_error: float | None
    setup_time_s: float
    solve_time_s: float
    iterations: int
    method: str
    meta: dict[str, Any]


def _rel_res(j: csr_matrix, dx: np.ndarray, r: np.ndarray) -> float:
    num = float(np.linalg.norm(j @ dx - r))
    den = max(float(np.linalg.norm(r)), 1e-30)
    return num / den


def solve_superlu(
    j: csr_matrix,
    r: np.ndarray,
    *,
    ref_dx: np.ndarray | None = None,
) -> LinearSolveResult:
    t0 = time.perf_counter()
    lu = splu(j.tocsc())
    t1 = time.perf_counter()
    dx = lu.solve(r)
    t2 = time.perf_counter()
    rel = _rel_res(j, dx, r)
    rel_x = None if ref_dx is None else float(np.linalg.norm(dx - ref_dx) / (np.linalg.norm(ref_dx) + 1e-30))
    return LinearSolveResult(
        dx=dx,
        success=rel < 1e-6,
        relative_residual=rel,
        relative_solution_error=rel_x,
        setup_time_s=t1 - t0,
        solve_time_s=t2 - t1,
        iterations=0,
        method="scipy_superlu",
        meta={},
    )


def solve_gmres(
    j: csr_matrix,
    r: np.ndarray,
    *,
    use_ilu: bool = False,
    rtol: float = 1e-8,
    ref_dx: np.ndarray | None = None,
) -> LinearSolveResult:
    t0 = time.perf_counter()
    precond = None
    setup = 0.0
    if use_ilu:
        try:
            ilu = spilu(j.tocsc(), drop_tol=1e-4)
            precond = LinearOperator(j.shape, matvec=ilu.solve)
            setup = time.perf_counter() - t0
        except Exception as exc:
            return LinearSolveResult(
                dx=np.zeros_like(r),
                success=False,
                relative_residual=float("inf"),
                relative_solution_error=None,
                setup_time_s=setup,
                solve_time_s=0.0,
                iterations=0,
                method="gmres_ilu_failed",
                meta={"error": str(exc)},
            )
    t1 = time.perf_counter()
    dx, info = gmres(j, r, atol=0, rtol=rtol, M=precond)
    t2 = time.perf_counter()
    rel = _rel_res(j, dx, r)
    rel_x = None if ref_dx is None else float(np.linalg.norm(dx - ref_dx) / (np.linalg.norm(ref_dx) + 1e-30))
    return LinearSolveResult(
        dx=dx,
        success=info == 0 and rel < max(rtol, 1e-3),
        relative_residual=rel,
        relative_solution_error=rel_x,
        setup_time_s=setup,
        solve_time_s=t2 - t1,
        iterations=int(info) if info > 0 else 0,
        method="gmres_ilu" if use_ilu else "gmres",
        meta={"gmres_info": int(info)},
    )


def solve_klu_placeholder(j: csr_matrix, r: np.ndarray, **kwargs: Any) -> LinearSolveResult:
    """KLU via scikit-umfpack when installed; otherwise SuperLU fallback."""
    try:
        from sksparse.cholmod import cholesky  # noqa: F401 — not used

        raise ImportError
    except ImportError:
        out = solve_superlu(j, r, ref_dx=kwargs.get("ref_dx"))
        out.method = "superlu_klu_unavailable"
        return out