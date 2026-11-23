"""Cross-check GENCO / jsgc reduced Jacobian against finite differences."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jsgc_physics import build_ac_jacobian, build_reduced_index, reduced_ac_residual


@dataclass
class JacobianVerifyReport:
    max_abs_err: float
    max_rel_err: float
    n_x: int
    passed: bool


def finite_difference_jacobian(
    theta: torch.Tensor,
    vm: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    p_spec: torch.Tensor,
    q_spec: torch.Tensor,
    index,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """J_FD with same sign convention as build_ac_jacobian (d calc / d x)."""
    r0 = reduced_ac_residual(theta, vm, g, b, p_spec, q_spec, index)
    n_x = index.n_x
    j_fd = torch.zeros(n_x, n_x, dtype=torch.float64)
    for k in range(n_x):
        d = torch.zeros(n_x, dtype=torch.float64)
        d[k] = eps
        th, vk = theta.clone(), vm.clone()
        if index.n_theta:
            th[index.non_slack] = th[index.non_slack] + d[: index.n_theta]
        if index.n_v:
            vk[index.pq] = vk[index.pq] + d[index.n_theta :]
        r_p = reduced_ac_residual(th, vk, g, b, p_spec, q_spec, index)
        j_fd[:, k] = -(r_p - r0) / eps
    return j_fd


def verify_toy3bus() -> JacobianVerifyReport:
    y_s = 1.0 / (0.01 + 1j * 0.1)
    ybus = torch.zeros(3, 3, dtype=torch.complex128)
    for i, j in ((0, 1), (1, 2)):
        ybus[i, j] = ybus[j, i] = -y_s
        ybus[i, i] += y_s
        ybus[j, j] += y_s
    from jsgc_physics import TYPE_PQ, TYPE_PV, TYPE_SLACK

    types = torch.tensor([TYPE_PQ, TYPE_PV, TYPE_SLACK], dtype=torch.long)
    g, b = ybus.real.to(torch.float64), ybus.imag.to(torch.float64)
    p_spec = torch.tensor([-0.5, 0.8, 0.0], dtype=torch.float64)
    q_spec = torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64)
    theta = torch.zeros(3, dtype=torch.float64)
    vm = torch.ones(3, dtype=torch.float64)
    index = build_reduced_index(types)
    j_a = build_ac_jacobian(theta, vm, g, b, index)
    j_fd = finite_difference_jacobian(theta, vm, g, b, p_spec, q_spec, index)
    err = (j_a - j_fd).abs()
    rel = err / j_fd.abs().clamp(min=1e-8)
    return JacobianVerifyReport(
        max_abs_err=float(err.max()),
        max_rel_err=float(rel.max()),
        n_x=index.n_x,
        passed=bool(torch.allclose(j_a, j_fd, rtol=1e-3, atol=1e-3)),
    )