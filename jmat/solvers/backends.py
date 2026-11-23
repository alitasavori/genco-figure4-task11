"""Identify and dispatch sparse direct solvers (Phase 3 §1)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import gmres, spilu, splu, spsolve

from jmat.solvers.linear import LinearSolveResult, _rel_res

BackendName = Literal["superlu", "umfpack", "klu"]


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    detail: str


def probe_backends() -> list[BackendStatus]:
    """Report which direct solvers are actually callable in this environment."""
    statuses: list[BackendStatus] = []
    statuses.append(
        BackendStatus(
            name="superlu",
            available=True,
            detail="scipy.sparse.linalg.splu (always available in SciPy)",
        )
    )
    umf_ok = hasattr(scipy.sparse.linalg, "use_solver")
    try:
        scipy.sparse.linalg.use_solver(useUmfpack=True)
        umf_ok = True
        detail = "scipy UMFPACK via spsolve/use_solver"
    except Exception as exc:
        detail = f"UMFPACK not available: {exc}"
        umf_ok = False
    statuses.append(BackendStatus(name="umfpack", available=umf_ok, detail=detail))
    klu_ok = False
    klu_detail = "no Python KLU binding found"
    try:
        import sparseqr  # noqa: F401

        klu_detail = "sparseqr present but not KLU — still unavailable"
    except ImportError:
        pass
    try:
        from scikits import umfpack  # noqa: F401

        klu_detail = "scikits.umfpack only (not KLU); use umfpack backend"
    except ImportError:
        pass
    statuses.append(BackendStatus(name="klu", available=klu_ok, detail=klu_detail))
    return statuses


def solve_direct(
    j: csr_matrix,
    r: np.ndarray,
    backend: BackendName,
    *,
    ref_dx: np.ndarray | None = None,
) -> LinearSolveResult:
    """Run one direct solve; ``method`` string matches the backend actually used."""
    j = j.tocsr()
    if backend == "superlu":
        t0 = time.perf_counter()
        lu = splu(j.tocsc())
        t1 = time.perf_counter()
        dx = lu.solve(r)
        t2 = time.perf_counter()
        method = "scipy_superlu"
        meta: dict[str, Any] = {"library": "SuperLU", "via": "scipy.sparse.linalg.splu"}
    elif backend == "umfpack":
        t0 = time.perf_counter()
        try:
            scipy.sparse.linalg.use_solver(useUmfpack=True)
        except Exception:
            pass
        dx = spsolve(csc_matrix(j), r)
        t1 = time.perf_counter()
        t2 = t1
        method = "scipy_umfpack"
        meta = {"library": "UMFPACK", "via": "scipy.sparse.linalg.spsolve"}
    elif backend == "klu":
        raise NotImplementedError(
            "SuiteSparse KLU is not exposed in this environment; use superlu or umfpack."
        )
    else:
        raise ValueError(backend)
    rel = _rel_res(j, dx, r)
    rel_x = None if ref_dx is None else float(np.linalg.norm(dx - ref_dx) / (np.linalg.norm(ref_dx) + 1e-30))
    return LinearSolveResult(
        dx=np.asarray(dx).reshape(-1),
        success=rel < 1e-6,
        relative_residual=rel,
        relative_solution_error=rel_x,
        setup_time_s=t1 - t0,
        solve_time_s=t2 - t1,
        iterations=0,
        method=method,
        meta=meta,
    )


def solve_gmres_wrapped(
    j: csr_matrix,
    r: np.ndarray,
    *,
    use_ilu: bool = False,
    rtol: float = 1e-8,
    ref_dx: np.ndarray | None = None,
) -> LinearSolveResult:
    from jmat.solvers.linear import solve_gmres

    return solve_gmres(j, r, use_ilu=use_ilu, rtol=rtol, ref_dx=ref_dx)