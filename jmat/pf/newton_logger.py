"""Reduced polar Newton–Raphson with per-iteration (J, r, Δx*) logging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix

from jsgc_physics import (
    build_ac_jacobian,
    build_reduced_index,
    flat_start_theta_v,
    pack_reduced_x,
    reduced_ac_residual,
    unpack_reduced_x,
)


@dataclass
class NewtonLogStep:
    iteration: int
    j: csr_matrix
    r: np.ndarray
    dx_star: np.ndarray
    lin_residual_rel: float
    x: np.ndarray
    converged: bool


def verify_linear_residual(j: csr_matrix, dx: np.ndarray, r: np.ndarray, *, tol: float = 1e-8) -> float:
    res = j @ dx - r
    rn = float(np.linalg.norm(r))
    if rn == 0:
        return 0.0 if float(np.linalg.norm(res)) == 0 else float("inf")
    return float(np.linalg.norm(res) / rn)


def newton_trace(
    g: torch.Tensor,
    b_mat: torch.Tensor,
    types: torch.Tensor,
    p_spec: torch.Tensor,
    q_spec: torch.Tensor,
    *,
    v_set: torch.Tensor | None = None,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> tuple[list[NewtonLogStep], dict[str, float]]:
    """Flat-start polar Newton; mismatch r = spec − calc (same as jsgc)."""
    index = build_reduced_index(types)
    theta, vm = flat_start_theta_v(types, v_set=v_set)
    x = pack_reduced_x(theta, vm, index).detach().cpu().numpy().astype(np.float64)
    steps: list[NewtonLogStep] = []
    status = {"iterations": 0, "converged": 0.0, "final_mismatch_inf": float("inf")}

    for k in range(max_iter):
        th, vmk = unpack_reduced_x(
            torch.tensor(x, dtype=torch.float64),
            index,
            theta_base=theta,
            v_base=vm,
        )
        r_t = reduced_ac_residual(th, vmk, g, b_mat, p_spec, q_spec, index).detach().cpu().numpy()
        mis_inf = float(np.max(np.abs(r_t)))
        status["final_mismatch_inf"] = mis_inf
        j_t = build_ac_jacobian(th, vmk, g, b_mat, index).detach().cpu().numpy()
        j_csr = csr_matrix(j_t)
        try:
            dx = np.linalg.solve(j_t, r_t)
        except np.linalg.LinAlgError:
            dx = np.zeros_like(r_t)
        lin_rel = verify_linear_residual(j_csr, dx, r_t)
        converged = mis_inf < tol
        steps.append(
            NewtonLogStep(
                iteration=k,
                j=j_csr,
                r=r_t.copy(),
                dx_star=dx.copy(),
                lin_residual_rel=lin_rel,
                x=x.copy(),
                converged=converged,
            )
        )
        if converged:
            status["converged"] = 1.0
            status["iterations"] = k + 1
            break
        x = x + dx
        status["iterations"] = k + 1
    return steps, status