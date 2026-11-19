"""Unit tests for JSGC physics operators (no GPU required)."""

from __future__ import annotations

import numpy as np
import torch

from jsgc_physics import (
    TYPE_PQ,
    TYPE_PV,
    TYPE_SLACK,
    ac_power_injections,
    baseline_iterate,
    build_ac_jacobian,
    build_reduced_index,
    coarse_correct,
    jacobi_solve,
    prepare_topology,
    projected_linear_residual,
    reduced_ac_residual,
    ybus_to_gb,
)


def _toy_3bus() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple 3-bus: slack=2, PV=1, PQ=0; two lines."""
    types = torch.tensor([TYPE_PQ, TYPE_PV, TYPE_SLACK], dtype=torch.long)
    # Series y = 1/(0.01+j0.1) ≈ 0.990−j9.901
    y_s = 1.0 / (0.01 + 1j * 0.1)
    ybus = torch.zeros(3, 3, dtype=torch.complex128)
    for i, j in ((0, 1), (1, 2)):
        ybus[i, j] = ybus[j, i] = -y_s
        ybus[i, i] += y_s
        ybus[j, j] += y_s
    p_spec = torch.tensor([-0.5, 0.8, 0.0], dtype=torch.float64)  # PQ load, PV gen
    q_spec = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64)
    return ybus, types, p_spec, q_spec


def test_ac_injections_match_ybus_matmul() -> None:
    ybus, types, p_spec, q_spec = _toy_3bus()
    g, b = ybus_to_gb(ybus)
    theta = torch.zeros(3, dtype=torch.float64)
    v = torch.ones(3, dtype=torch.float64)
    p_calc, q_calc = ac_power_injections(theta, v, g, b)
    vc = v * torch.exp(1j * theta)
    flow = vc * torch.conj(ybus @ vc)
    assert torch.allclose(p_calc, flow.real, atol=1e-10)
    assert torch.allclose(q_calc, flow.imag, atol=1e-10)


def test_jacobian_finite_difference() -> None:
    ybus, types, p_spec, q_spec = _toy_3bus()
    cache = prepare_topology(ybus, types, r_theta=2, r_v=1)
    index = cache.index
    theta, v = cache.theta0.clone(), cache.v0.clone()
    j_analytic = build_ac_jacobian(theta, v, cache.g, cache.b, index)
    eps = 1e-6
    x0_r = reduced_ac_residual(theta, v, cache.g, cache.b, p_spec, q_spec, index)
    # r = spec - calc, J_newton = ∂calc/∂x = -∂r/∂x
    j_fd = torch.zeros_like(j_analytic)
    n_x = index.n_x
    for k in range(n_x):
        d = torch.zeros(n_x, dtype=torch.float64)
        d[k] = eps
        th_p, v_p = theta.clone(), v.clone()
        if index.n_theta:
            th_p[index.non_slack] = th_p[index.non_slack] + d[: index.n_theta]
        if index.n_v:
            v_p[index.pq] = v_p[index.pq] + d[index.n_theta :]
        r_p = reduced_ac_residual(th_p, v_p, cache.g, cache.b, p_spec, q_spec, index)
        # ∂r/∂x ≈ (r_p - r0)/eps → ∂calc/∂x = -∂r/∂x
        j_fd[:, k] = -(r_p - x0_r) / eps
    assert torch.allclose(j_analytic, j_fd, atol=1e-4, rtol=1e-3)


def test_coarse_projection_identity() -> None:
    ybus, types, p_spec, q_spec = _toy_3bus()
    cache = prepare_topology(ybus, types, r_theta=2, r_v=1)
    r = reduced_ac_residual(
        cache.theta0, cache.v0, cache.g, cache.b, p_spec, q_spec, cache.index
    )
    z_local = jacobi_solve(r, cache.d_blocks, cache.index)
    delta = coarse_correct(r, z_local, cache.j0, cache.p, cache.lu, cache.pivots)
    proj = projected_linear_residual(r, delta, cache.j0, cache.p)
    assert float(proj) < 1e-8


def test_fixed_point_at_solved_state() -> None:
    """At a state where residual≈0, Jacobi+coarse correction should be tiny."""
    ybus, types, p_spec, q_spec = _toy_3bus()
    cache = prepare_topology(ybus, types, r_theta=2, r_v=1)
    # Run a few Newton-like steps with full J to approach a solution (toy).
    theta, v = cache.theta0.clone(), cache.v0.clone()
    for _ in range(20):
        r = reduced_ac_residual(theta, v, cache.g, cache.b, p_spec, q_spec, cache.index)
        if torch.linalg.vector_norm(r) < 1e-10:
            break
        j = build_ac_jacobian(theta, v, cache.g, cache.b, cache.index)
        try:
            dx = torch.linalg.solve(j, r)
        except RuntimeError:
            dx = torch.linalg.lstsq(j, r.unsqueeze(-1)).solution.view(-1)
        if cache.index.n_theta:
            theta[cache.index.non_slack] = theta[cache.index.non_slack] + 0.7 * dx[: cache.index.n_theta]
        if cache.index.n_v:
            v[cache.index.pq] = (v[cache.index.pq] + 0.7 * dx[cache.index.n_theta :]).clamp(0.5, 1.5)
    r = reduced_ac_residual(theta, v, cache.g, cache.b, p_spec, q_spec, cache.index)
    z = jacobi_solve(r, cache.d_blocks, cache.index)
    delta = coarse_correct(r, z, cache.j0, cache.p, cache.lu, cache.pivots)
    # If residual is small, correction should also be small
    if float(torch.linalg.vector_norm(r)) < 1e-6:
        assert float(torch.linalg.vector_norm(delta)) < 1e-4


def test_baseline_iterate_runs() -> None:
    ybus, types, p_spec, q_spec = _toy_3bus()
    cache = prepare_topology(ybus, types, r_theta=2, r_v=1)
    theta, v, residuals = baseline_iterate(p_spec, q_spec, cache, k=4)
    assert theta.shape == (3,)
    assert len(residuals) == 4
    # Residual should not explode
    assert float(torch.linalg.vector_norm(residuals[-1])) < 10.0


if __name__ == "__main__":
    test_ac_injections_match_ybus_matmul()
    test_jacobian_finite_difference()
    test_coarse_projection_identity()
    test_fixed_point_at_solved_state()
    test_baseline_iterate_runs()
    print("test_jsgc_physics OK")
