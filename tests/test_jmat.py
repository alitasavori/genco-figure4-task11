"""JMAT unit tests (engineering brief §6.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.csr_io import apply_permutation, dense_to_csr_torch  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.normalize import normalize_linear_system  # noqa: E402


REFERENCE_J = np.array(
    [
        [4.0, -1.0, 0.0],
        [-2.0, 3.0, -1.0],
        [0.0, -1.0, 2.0],
    ],
    dtype=np.float64,
)
REFERENCE_R = np.array([1.0, 0.0, 0.0], dtype=np.float64)
REFERENCE_DX = np.array([0.3125, 0.25, 0.125], dtype=np.float64)


def test_reference_linear_solve():
    dx = np.linalg.solve(REFERENCE_J, REFERENCE_R)
    np.testing.assert_allclose(dx, REFERENCE_DX, rtol=0, atol=1e-12)


def test_forward_shapes_and_zero_rhs():
    crow, col, val, n = dense_to_csr_torch(REFERENCE_J)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    dx = model.forward_csr(crow, col, val, n, torch.zeros(n))
    assert dx.shape == (n,)
    assert torch.all(dx == 0)
    dx2 = model.forward_csr(crow, col, val, n, torch.as_tensor(REFERENCE_R, dtype=torch.float32))
    assert dx2.shape == (n,)


@pytest.mark.parametrize("n", [3, 106])
def test_random_csr_forward_runs(n: int):
    if n == 106:
        pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((n, n))
    crow, col, val, n_out = dense_to_csr_torch(dense)
    r = torch.as_tensor(rng.standard_normal(n), dtype=torch.float32)
    model = JMAT(JMATConfig(d_model=16, n_latent=4, n_heads=2, n_loops=1))
    dx = model.forward_csr(crow, col, val, n_out, r)
    assert dx.shape == (n,)


def test_permutation_equivariance_features():
    crow, col, val, n = dense_to_csr_torch(REFERENCE_J)
    r = torch.as_tensor(REFERENCE_R, dtype=torch.float32)
    perm = torch.tensor([2, 0, 1])
    crow_p, col_p, val_p, r_p = apply_permutation(crow, col, val, n, r, perm)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    torch.manual_seed(0)
    dx = model.forward_csr(crow, col, val, n, r)
    torch.manual_seed(0)
    dx_p = model.forward_csr(crow_p, col_p, val_p, n, r_p)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    np.testing.assert_allclose(
        dx_p[inv].detach().numpy(),
        dx.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_positive_scaling_invariance():
    crow, col, val, n = dense_to_csr_torch(REFERENCE_J)
    r = torch.as_tensor(REFERENCE_R, dtype=torch.float32)
    model = JMAT(JMATConfig(d_model=32, n_latent=8, n_heads=4, n_loops=2))
    torch.manual_seed(1)
    dx1 = model.forward_csr(crow, col, val, n, r)
    scale = 2.5
    torch.manual_seed(1)
    dx2 = model.forward_csr(crow, col, val * scale, n, r * scale)
    np.testing.assert_allclose(dx1.detach().numpy(), dx2.detach().numpy(), rtol=1e-4, atol=1e-4)


def test_tiny_overfit():
    crow, col, val, n = dense_to_csr_torch(REFERENCE_J)
    r = torch.as_tensor(REFERENCE_R, dtype=torch.float32)
    target = torch.as_tensor(REFERENCE_DX, dtype=torch.float32)
    norm = normalize_linear_system(crow, col, val, n, r)
    assert norm is not None
    model = JMAT(JMATConfig(d_model=64, n_latent=16, n_heads=4, n_loops=3))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    z_star = target / norm.beta
    for _ in range(400):
        opt.zero_grad()
        z_hat, _ = model.forward_normalized(norm)
        loss = torch.mean((z_hat - z_star) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        dx = model.forward_csr(crow, col, val, n, r)
    err = torch.linalg.vector_norm(dx - target) / torch.linalg.vector_norm(target)
    assert float(err) < 0.05