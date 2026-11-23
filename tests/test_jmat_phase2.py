"""Phase 2 safeguards and equivariance (Task 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.complexity import complexity_report  # noqa: E402
from jmat.csr_io import apply_permutation, dense_to_csr_torch  # noqa: E402
from jmat.errors import InvalidLinearSystemError  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.normalize import normalize_linear_system  # noqa: E402

J = np.array([[4.0, -1.0, 0.0], [-2.0, 3.0, -1.0], [0.0, -1.0, 2.0]], dtype=np.float64)
R = np.array([1.0, 0.0, 0.0], dtype=np.float64)
DX = np.linalg.solve(J, R)


def test_zero_rhs_returns_zero_correction():
    crow, col, val, n = dense_to_csr_torch(J)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=1))
    dx = model.forward_csr(crow, col, val, n, torch.zeros(n))
    assert torch.allclose(dx, torch.zeros(n))


def test_zero_jacobian_nonzero_rhs_rejected():
    crow = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    col = torch.tensor([], dtype=torch.int64)
    val = torch.tensor([], dtype=torch.float32)
    r = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    with pytest.raises(InvalidLinearSystemError):
        normalize_linear_system(crow, col, val, 3, r)
    model = JMAT(JMATConfig(d_model=16, n_latent=4, n_heads=2, n_loops=1))
    with pytest.raises(InvalidLinearSystemError):
        model.forward_csr(crow, col, val, 3, r)


def test_joint_positive_scaling_invariance():
    """(αJ, αr) leaves Δx unchanged for an exact linear solve; network should match."""
    crow, col, val, n = dense_to_csr_torch(J)
    r = torch.tensor(R, dtype=torch.float32)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    torch.manual_seed(0)
    dx1 = model.forward_csr(crow, col, val, n, r)
    alpha = 2.5
    torch.manual_seed(0)
    dx2 = model.forward_csr(crow, col, val * alpha, n, r * alpha)
    np.testing.assert_allclose(dx1.detach().numpy(), dx2.detach().numpy(), rtol=2e-4, atol=2e-4)


def test_rhs_only_positive_scaling_equivariance():
    """J fixed, r → αr implies Δx → α Δx (linear physics)."""
    crow, col, val, n = dense_to_csr_torch(J)
    r = torch.tensor(R, dtype=torch.float32)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    alpha = 1.75
    torch.manual_seed(1)
    dx = model.forward_csr(crow, col, val, n, r)
    torch.manual_seed(1)
    dx_a = model.forward_csr(crow, col, val, n, r * alpha)
    np.testing.assert_allclose(dx_a.detach().numpy(), (alpha * dx).detach().numpy(), rtol=5e-2, atol=5e-2)


def test_permutation_equivariance_definition():
    """Δx'[i_new] = Δx[perm[i_new]] for J'_{i,j}=J_{perm[i],perm[j]}, r'_i=r_{perm[i]}."""
    crow, col, val, n = dense_to_csr_torch(J)
    r = torch.tensor(R, dtype=torch.float32)
    perm = torch.tensor([2, 0, 1])
    crow_p, col_p, val_p, r_p = apply_permutation(crow, col, val, n, r, perm)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    torch.manual_seed(2)
    dx = model.forward_csr(crow, col, val, n, r).detach().numpy()
    torch.manual_seed(2)
    dx_p = model.forward_csr(crow_p, col_p, val_p, n, r_p).detach().numpy()
    p = perm.numpy()
    expected = dx[p]
    np.testing.assert_allclose(dx_p, expected, rtol=2e-4, atol=2e-4)


def test_complexity_includes_dense_terms():
    crow, col, val, n = dense_to_csr_torch(J)
    nnz = int(val.numel())
    rep = complexity_report(n, nnz, JMATConfig())
    lines = rep.summary_lines()
    assert any("dense" in line.lower() for line in lines)
    assert rep.total_estimated_flops() > nnz * rep.d